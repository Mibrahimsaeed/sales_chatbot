"""Metric-focused responses (M7).

"connects of Shehryar Abbasi" must answer with the connects, not with
the profile that happens to contain them. The distinction is not
cosmetic: a superset reply reads as authoritative and leaves the user to
find their number among team, manager and targets.

The discriminator needs no new vocabulary — the ontology already
resolves a metric from any query, and it resolves NONE for "tell me
about X" / "who is X" / "show X profile". A named metric IS the metric
intent; its absence IS the profile intent.
"""

import pytest

from app.database.models import (
    Advisor, Attendance, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services import advisor_service
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def db(db_session, monkeypatch):
    db_session.add(Advisor(wid=1, name="Shehryar Abbasi", team="Blue Area", company="Graana",
                           bm="Kaleem Ullah", rm="Kaleem Ullah", zm="Adeel Dogar", portfolio_lead="Adeel Dogar", office="Gulberg BC"))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=2, mtd_followup_connect=0,
                               mtd_new_meeting=3, mtd_followup_meeting=1,
                               mtd_booking_stored=4))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, target=1000, cleared=500))
    db_session.add(Pipeline(wid=1, pipeline=7500, overdue=2))

    db_session.add(Advisor(wid=2, name="Ahmed Khan", team="Downtown", company="Agency21",
                           bm="Nadia Rehman", rm="Nadia Rehman"))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=10, mtd_followup_connect=5,
                               mtd_new_meeting=6, mtd_followup_meeting=0))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, target=2000, cleared=900))

    # Two people share a name — a metric question must ask, never guess.
    db_session.add(Advisor(wid=3, name="Yasir Ali", team="Blue Area", company="Graana"))
    db_session.add(Advisor(wid=4, name="Yasir Ali", team="Downtown", company="Agency21"))
    db_session.commit()

    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "llm_first")
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    import app.llm.llm_client as llm_client
    monkeypatch.setattr(llm_client._client.chat.completions, "create",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("off")))
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


# ---------------------------------------------------------------------
# The four specified cases
# ---------------------------------------------------------------------

def test_1_connects_returns_only_connects(db):
    r = handle_chat_message(db, "connects of Shehryar Abbasi", session_id=None)
    assert r["type"] == "advisor_metric"
    assert r["reply"] == "Shehryar Abbasi has 2 MTD connects."


def test_2_tell_me_about_returns_the_full_profile(db):
    r = handle_chat_message(db, "tell me about Shehryar Abbasi", session_id=None)
    assert r["type"] == "advisor"
    assert "MTD connects" in r["reply"]
    assert "target" in r["reply"]


def test_3_meetings_of_another_advisor(db):
    r = handle_chat_message(db, "meetings of Ahmed Khan", session_id=None)
    assert r["type"] == "advisor_metric"
    assert r["reply"] == "Ahmed Khan has 6 MTD meetings."


def test_4_show_profile_returns_the_full_profile(db):
    r = handle_chat_message(db, "show Shehryar Abbasi profile", session_id=None)
    assert r["type"] == "advisor"


# ---------------------------------------------------------------------
# Every keyword in the spec
# ---------------------------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("connects of Shehryar Abbasi", "Shehryar Abbasi has 2 MTD connects."),
    ("meetings of Shehryar Abbasi", "Shehryar Abbasi has 4 MTD meetings."),
    ("pipeline of Shehryar Abbasi", "Shehryar Abbasi has 7,500 MTD open pipeline."),
    ("overdue of Shehryar Abbasi", "Shehryar Abbasi has 2 MTD overdue pipeline."),
    ("target of Shehryar Abbasi", "Shehryar Abbasi has 1,000 MTD target."),
    ("cleared of Shehryar Abbasi", "Shehryar Abbasi has 500 MTD revenue cleared."),
    ("cr booked of Shehryar Abbasi", "Shehryar Abbasi has 4 MTD bookings stored."),
])
def test_each_specified_metric(db, query, expected):
    r = handle_chat_message(db, query, session_id=None)
    assert r["reply"] == expected


def test_the_reply_excludes_every_profile_field(db):
    """The explicit requirement: no company/team, no reports-to, no
    CR booked, no targets, no other metric."""
    reply = handle_chat_message(db, "connects of Shehryar Abbasi", session_id=None)["reply"]
    for forbidden in ("Blue Area", "Graana", "Kaleem Ullah", "target", "cleared",
                      "booking", "pipeline", "overdue"):
        assert forbidden.lower() not in reply.lower(), forbidden
    assert reply.count(".") == 1


def test_possessive_phrasing_works_too(db):
    r = handle_chat_message(db, "Shehryar Abbasi's connects", session_id=None)
    assert r["type"] == "advisor_metric"


def test_the_response_carries_structured_data(db):
    """The four keys a consumer reads, unchanged.

    Asserted as a SUBSET since Phase 13B, which added `metrics` (the
    per-measure list, for a reply naming several) and `unavailable` (the
    measures that were asked for and could not be served). Both are
    additive — the exact-equality form would fail on any new field
    regardless of whether an existing one had changed, which is the
    opposite of what this test is for. The companion test below pins the
    new fields on their own.
    """
    r = handle_chat_message(db, "connects of Shehryar Abbasi", session_id=None)
    assert r["data"].items() >= {"wid": 1, "name": "Shehryar Abbasi",
                                 "metric": "total_connects", "value": 2.0}.items()


def test_a_single_metric_response_reports_one_metric_and_nothing_unavailable(db):
    """The Phase 13B fields on a single-measure question: one entry, and
    an empty unavailable list. A consumer can therefore read `metrics`
    uniformly instead of special-casing the single answer."""
    r = handle_chat_message(db, "connects of Shehryar Abbasi", session_id=None)
    assert r["data"]["metrics"] == [{"metric": "total_connects", "value": 2.0}]
    assert r["data"]["unavailable"] == []


# ---------------------------------------------------------------------
# Profile queries stay profile queries
# ---------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "tell me about Shehryar Abbasi",
    "show profile of Shehryar Abbasi",
    "who is Shehryar Abbasi",
    "Shehryar Abbasi",
    "info on Shehryar Abbasi",
    "how is Shehryar Abbasi doing",
    "performance of Shehryar Abbasi",
])
def test_profile_queries_are_unchanged(db, query):
    r = handle_chat_message(db, query, session_id=None)
    assert r["type"] == "advisor", query


# ---------------------------------------------------------------------
# Identity and edge cases
# ---------------------------------------------------------------------

def test_an_ambiguous_name_asks_before_reporting_a_number(db):
    """Two people are called Yasir Ali. Reporting either one's connects
    would be the wrong-person failure this codebase exists to prevent."""
    r = handle_chat_message(db, "connects of Yasir Ali", session_id=None)
    assert r["type"] == "clarification"


def test_an_unknown_person_is_not_invented(db):
    r = handle_chat_message(db, "connects of Nobody At All", session_id=None)
    assert r["type"] != "advisor_metric"


def test_a_missing_fact_row_is_said_plainly_not_reported_as_zero(db):
    """Yasir Ali (wid=3) has no SalesFunnel row. Zero is a real value and
    claiming it would be a wrong answer."""
    assert advisor_service.get_advisor_metric(db, 3, "total_connects") is None
    from app.llm.response_formatter import format_advisor_metric_reply

    assert format_advisor_metric_reply("Yasir Ali", "total_connects", None) == (
        "I don't have MTD connects on file for Yasir Ali."
    )


def test_group_metric_queries_are_untouched(db):
    """"connects of Blue Area" is a group question, not a person one."""
    r = handle_chat_message(db, "connects of Blue Area", session_id=None)
    assert r["type"] != "advisor_metric"


def test_leaderboards_are_untouched(db):
    r = handle_chat_message(db, "top 5 advisors by connects", session_id=None)
    assert r["type"] == "leaderboard"


def test_reverse_lookup_is_untouched(db):
    r = handle_chat_message(db, "who is Shehryar Abbasi's BM", session_id=None)
    assert r["type"] == "manager"


def test_the_metric_value_matches_the_profile_value(db):
    """The two paths read the same underlying data — a metric reply that
    disagreed with the profile would be worse than either."""
    profile = handle_chat_message(db, "tell me about Shehryar Abbasi", session_id=None)
    metric = handle_chat_message(db, "connects of Shehryar Abbasi", session_id=None)
    assert "2 MTD connects" in profile["reply"]
    assert "2 MTD connects" in metric["reply"]
