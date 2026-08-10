"""Phase 29 — the measures that answer together.

"connects of X" answered with a connects count alone. The number was
right and incomplete: connects only means something beside the answered
calls behind it and the rate between the two, and a meetings count only
means something beside its rate. The user had to ask three questions to
read one answer.

NOTHING HERE IS CALCULATED. Every metric in a bundle already existed with
its own binding — `answered_calls`, `answered_calls_rate`, `meeting_rate`
— and every value is read from the owner the surrounding answer already
reads from: advisor_service for a person, aggregation.metric_value for a
group. The ontology gained a declaration of WHICH measures belong in one
answer, not a definition of any of them, which is why the primary figure
is byte-identical before and after and gets its own tests below.

APPENDED, NEVER SUBSTITUTED. The sentence the user asked for stays first
and unchanged; the block follows. That is what lets the bundle carry no
window of its own — the headline states one, and a second would be the
one way these two halves could disagree.

The fixture gives connects, answered calls and meetings DIFFERENT values
per person on purpose. A bundle that silently re-read the primary, or
read a group's figure for a person, would still produce three lines —
only distinct numbers prove each line came from its own metric at the
right scope.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, conversation_memory, entity_extractor, narrative,
    semantic_parser,
)
from app.llm.metric_ontology import bundle_for
from app.services.chat_service import handle_chat_message

# Ayesha Nawaz is the Unit Head; Bilal Tahir the BCM under her.
# wid, name,           rm,             portfolio_lead,  management_lead, connects, answered, meetings
PEOPLE = [
    (1, "Ayesha Nawaz", "Ayesha Nawaz", "Ayesha Nawaz", "Ayesha Nawaz",  40, 30, 9),
    (2, "Bilal Tahir",  "Ayesha Nawaz", "Ayesha Nawaz", "Ayesha Nawaz",  60, 45, 6),
    (3, "Sadia Kamal",  "Ayesha Nawaz", "Other Zh",     "Other Bcm",    100, 25, 5),
    (4, "Owais Malik",  "Other Uh",     "Other Zh",     "Other Bcm",    999, 999, 99),
]

AYESHA_OWN_CONNECTS = 40
AYESHA_OWN_ANSWERED = 30
AYESHA_OWN_MEETINGS = 9

UNIT_CONNECTS = 40 + 60 + 100      # 200
UNIT_ANSWERED = 30 + 45 + 25       # 100
UNIT_MEETINGS = 9 + 6 + 5          # 20


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid, name, rm, pl, ml, connects, answered, meetings in PEOPLE:
        db_session.add(Advisor(wid=wid, name=name, team="Alpha", company="Graana",
                               rm=rm, portfolio_lead=pl, management_lead=ml,
                               in_master_sheet=True))
        db_session.add(Calls(wid=wid, connects_mtd=connects, connects_daily=0,
                             answered_calls_mtd=answered, answered_calls_daily=0))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=connects,
                                   mtd_followup_connect=0, mtd_cr=1,
                                   mtd_new_meeting=meetings, mtd_followup_meeting=0))
        db_session.add(Pipeline(wid=wid, pipeline=connects, overdue=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=connects))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


def _ask(db, text):
    session = f"p29-{text}"
    conversation_memory._store.pop(session, None)
    return handle_chat_message(db, text, session_id=session)


def _bundle(response):
    """{metric_key: value} from the payload — the reply's block rendered
    from exactly this, so asserting here pins the numbers themselves
    rather than their formatting."""
    data = response.get("data")
    entries = response.get("bundle")
    if entries is None and isinstance(data, dict):
        entries = data.get("bundle")
    return {b["metric"]: b["value"] for b in (entries or [])}


def _headline(response):
    return str(response["reply"]).split("\n\n")[0]


# ---------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------


def test_connects_bundles_its_answered_calls_and_rate():
    assert bundle_for("total_connects") == [
        "total_connects", "answered_calls", "answered_calls_rate"]


def test_meetings_bundles_its_rate():
    assert bundle_for("total_meetings") == ["total_meetings", "meeting_rate"]


@pytest.mark.parametrize("key", ["mtd_cleared", "pipeline", "mtd_target",
                                 "client_registrations", "achievement_pct"])
def test_every_other_measure_is_unbundled(key):
    """The bundle applies to two measures. Anything else answers exactly
    as it did, which is what keeps this a targeted change."""
    assert bundle_for(key) == []


def test_a_bundle_follows_the_window_it_was_asked_at():
    """A daily question must not pick up an MTD rate. Period resolution is
    delegated to the same authority the primary uses, so the members swap
    together or not at all."""
    assert bundle_for("total_connects", PerformancePeriod.DAILY) == [
        "daily_connects", "daily_answered_calls", "daily_answered_calls_rate"]
    assert bundle_for("daily_connects", PerformancePeriod.DAILY) == [
        "daily_connects", "daily_answered_calls", "daily_answered_calls_rate"]


def test_a_member_with_no_data_at_that_window_is_dropped_not_substituted():
    """There is no YTD answered-calls column. Reporting the MTD one beside
    a YTD count is the mismatch this omission exists to prevent."""
    assert bundle_for("total_connects", PerformancePeriod.YTD) == ["ytd_connects"]


# ---------------------------------------------------------------------
# A person's connects
# ---------------------------------------------------------------------


def test_a_person_connects_query_shows_all_three(db):
    reply = _ask(db, "connects of Ayesha Nawaz")["reply"]
    assert "Total Connects" in reply
    assert "Answered Calls" in reply
    assert "Answered Calls %" in reply


def test_a_person_connects_bundle_carries_that_persons_own_values(db):
    """Person-scoped, not promoted to the Unit Head scope she also holds."""
    values = _bundle(_ask(db, "connects of Ayesha Nawaz"))
    assert values["total_connects"] == AYESHA_OWN_CONNECTS
    assert values["answered_calls"] == AYESHA_OWN_ANSWERED
    assert values["answered_calls_rate"] is not None


def test_a_person_connects_bundle_is_not_the_teams(db):
    values = _bundle(_ask(db, "connects of Ayesha Nawaz"))
    assert values["total_connects"] != UNIT_CONNECTS
    assert values["answered_calls"] != UNIT_ANSWERED


# ---------------------------------------------------------------------
# A person's meetings
# ---------------------------------------------------------------------


def test_a_person_meetings_query_shows_both(db):
    reply = _ask(db, "meetings of Ayesha Nawaz")["reply"]
    assert "Total Meetings" in reply
    assert "Meeting %" in reply


def test_a_person_meetings_bundle_carries_that_persons_own_values(db):
    values = _bundle(_ask(db, "meetings of Ayesha Nawaz"))
    assert values["total_meetings"] == AYESHA_OWN_MEETINGS
    assert values["meeting_rate"] is not None


def test_a_meetings_bundle_names_no_call_metrics(db):
    """Two bundles, not one shared list of everything."""
    reply = _ask(db, "meetings of Ayesha Nawaz")["reply"]
    assert "Answered Calls" not in reply


# ---------------------------------------------------------------------
# A team's connects and meetings, at every manager level
# ---------------------------------------------------------------------


def test_a_team_connects_query_shows_all_three(db):
    reply = _ask(db, "connects of Ayesha Nawaz's team")["reply"]
    assert "Total Connects" in reply
    assert "Answered Calls" in reply
    assert "Answered Calls %" in reply


def test_a_team_connects_bundle_carries_the_group_values(db):
    values = _bundle(_ask(db, "connects of Ayesha Nawaz's team"))
    assert values["total_connects"] == UNIT_CONNECTS
    assert values["answered_calls"] == UNIT_ANSWERED


def test_a_team_meetings_query_shows_both(db):
    response = _ask(db, "meetings of Ayesha Nawaz's team")
    assert "Total Meetings" in response["reply"]
    assert "Meeting %" in response["reply"]
    assert _bundle(response)["total_meetings"] == UNIT_MEETINGS


def test_a_named_team_gets_the_bundle_too(db):
    """Group scope is group scope — a team named directly is not a
    different kind of answer from a manager's team."""
    response = _ask(db, "connects of Alpha")
    assert "Answered Calls" in response["reply"]


def test_nobody_outside_the_scope_reaches_the_bundle(db):
    values = _bundle(_ask(db, "connects of Ayesha Nawaz's team"))
    assert values["total_connects"] == UNIT_CONNECTS       # 999 excluded


# ---------------------------------------------------------------------
# Nothing that already worked changed
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("connects of Ayesha Nawaz", f"Ayesha Nawaz has {AYESHA_OWN_CONNECTS} MTD connects."),
    ("meetings of Ayesha Nawaz", f"Ayesha Nawaz has {AYESHA_OWN_MEETINGS} MTD meetings."),
])
def test_the_primary_sentence_is_unchanged_and_still_first(db, query, expected):
    """The whole compatibility contract: the bundle is appended, so the
    answer the user asked for is still the first thing they read and is
    still byte-identical."""
    assert _headline(_ask(db, query)) == expected


def test_the_primary_payload_fields_are_unchanged(db):
    """`metric`/`value` are the measure that was ASKED for, and `metrics`
    is Phase 13B's list of the same. A bundle is context, not a request,
    so it rides in its own key and none of these move."""
    data = _ask(db, "connects of Ayesha Nawaz")["data"]
    assert data["metric"] == "total_connects"
    assert data["value"] == AYESHA_OWN_CONNECTS
    assert data["metrics"] == [{"metric": "total_connects", "value": AYESHA_OWN_CONNECTS}]
    assert data["unavailable"] == []


def test_the_bundle_agrees_with_the_headline(db):
    """The first bundled line restates the primary. It is REUSED, not
    re-read — two fetches of one number is how a block and the sentence
    above it start to disagree."""
    response = _ask(db, "connects of Ayesha Nawaz")
    assert _bundle(response)["total_connects"] == response["data"]["value"]


@pytest.mark.parametrize("query", [
    "revenue of Ayesha Nawaz", "pipeline of Ayesha Nawaz", "target of Ayesha Nawaz",
])
def test_an_unbundled_measure_answers_exactly_as_before(db, query):
    reply = _ask(db, query)["reply"]
    assert "\n" not in reply
    assert "•" not in reply


def test_a_multi_measure_question_gets_no_bundle(db):
    """It already lists what was asked for; a block underneath would
    restate the same numbers under different headings."""
    response = _ask(db, "connects and meetings of Ayesha Nawaz")
    assert not _bundle(response)
    assert "•" not in response["reply"]


def test_a_leaderboard_gets_no_bundle(db):
    """A bundle values ONE subject. On a ranking it would need a row per
    line, which the list already is."""
    response = _ask(db, "top advisors by connects")
    assert "•" not in response["reply"] or "Answered Calls %" not in response["reply"]


# ---------------------------------------------------------------------
# Phase 20-28 guarantees still hold through the bundle
# ---------------------------------------------------------------------


def test_person_and_team_scopes_still_differ(db):
    person = _bundle(_ask(db, "connects of Ayesha Nawaz"))
    team = _bundle(_ask(db, "connects of Ayesha Nawaz's team"))
    assert person["total_connects"] == AYESHA_OWN_CONNECTS
    assert team["total_connects"] == UNIT_CONNECTS


def test_the_member_breakdown_still_follows_the_bundle(db):
    """Phase 27 and Phase 29 both append to the same reply. Both must
    survive, and the members must still sum to the headline total."""
    response = _ask(db, "connects of Ayesha Nawaz's team")
    assert "👥" in response["reply"]
    assert sum(m["value"] or 0 for m in response["members"]) == UNIT_CONNECTS


def test_the_highest_role_is_still_resolved_without_asking(db):
    """Phase 28: Ayesha grounds at four levels and is asked nothing."""
    response = _ask(db, "connects of Ayesha Nawaz's team")
    assert response["type"] != "clarification"
    assert _bundle(response)["total_connects"] == UNIT_CONNECTS
