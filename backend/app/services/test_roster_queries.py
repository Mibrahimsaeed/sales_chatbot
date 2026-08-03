"""
Roster/list intent — "who is IN this group", not "how is this group doing".

Reported failure: "All advisors in Blue Area" routed to the entity-summary
branch and answered with aggregate metrics (connects, pipeline, overdue).
That is an answer to a different question. Both readings mention the same
entity, so only the PHRASING distinguishes them — hence a dedicated
intent rather than a tweak to the summary path.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.llm.query_planner import build_query_plan
from app.services.chat_service import handle_chat_message

# Fields that only ever appear on a metric SUMMARY. The core requirement
# is that a roster query never returns these, so it is asserted directly.
_METRIC_FIELDS = ("connects", "overdue", "pipeline", "mtd_cleared", "mtd_target")


@pytest.fixture()
def roster_db(db_session, monkeypatch):
    def advisor(wid, name, team, company="Graana", **kw):
        db_session.add(Advisor(wid=wid, name=name, team=team, company=company, **kw))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=10, mtd_followup_connect=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000, cleared=500))

    # Blue Area is a TEAM in production (not an office) — the roster action
    # must be level-agnostic rather than assuming any one level.
    advisor(1, "Alice Ahmed", "Blue Area", office="F-11 Center", bm="Adeel Dogar", rm="Adeel Dogar")
    advisor(2, "Bilal Khan", "Blue Area", office="F-11 Center", bm="Adeel Dogar", rm="Adeel Dogar")
    advisor(3, "Carla Shah", "Blue Area", office="G-9 Center", bm="Fraz Khalid", rm="Fraz Khalid")
    advisor(10, "Danish Ali", "North/KPK Region", company="Agency21", bm="Fraz Khalid", rm="Fraz Khalid")
    advisor(11, "Erum Javed", "North/KPK Region", company="Agency21", zm="Zarak Khan", portfolio_lead="Zarak Khan")
    advisor(20, "Farah Iqbal", "Downtown", company="IMARAT")
    db_session.commit()

    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


def _assert_is_roster_not_summary(response, expected_names=None):
    assert response["type"] == "roster", f"expected a roster, got {response['type']}"
    data = response["data"]
    for field in _METRIC_FIELDS:
        assert field not in data, f"roster leaked the metric field {field!r}"
    if expected_names is not None:
        assert {a["name"] for a in data["advisors"]} == expected_names


# =====================================================================
# The reported cases
# =====================================================================

def test_all_advisors_in_blue_area(roster_db):
    response = handle_chat_message(roster_db, "All advisors in Blue Area", session_id=None)
    _assert_is_roster_not_summary(response, {"Alice Ahmed", "Bilal Khan", "Carla Shah"})
    assert response["data"]["level"] == "team"
    assert response["data"]["count"] == 3


def test_list_advisors_in_blue_area(roster_db):
    response = handle_chat_message(roster_db, "List advisors in Blue Area", session_id=None)
    _assert_is_roster_not_summary(response, {"Alice Ahmed", "Bilal Khan", "Carla Shah"})


def test_show_advisors_under_adeel_dogar(roster_db):
    """A relational phrasing that is ALSO a roster — the roster reading
    wins because the user said "advisors", so a nested team breakdown
    would answer a slightly different question."""
    response = handle_chat_message(roster_db, "Show advisors under Adeel Dogar", session_id=None)
    _assert_is_roster_not_summary(response, {"Alice Ahmed", "Bilal Khan"})
    assert response["data"]["level"] == "unit_head"


def test_advisors_in_north_kpk_region(roster_db):
    response = handle_chat_message(roster_db, "Advisors in North/KPK Region", session_id=None)
    _assert_is_roster_not_summary(response, {"Danish Ali", "Erum Javed"})


def test_advisors_in_a_company(roster_db):
    """"Advisors in Graana" — company-scoped roster. (In production
    "Graana" is BOTH a company and an office name, so the live query
    hits the cross-level ambiguity prompt first; that is genuine
    ambiguity in the data, not a routing failure, and it is still never
    a metric summary.)"""
    response = handle_chat_message(roster_db, "Advisors in IMARAT", session_id=None)
    _assert_is_roster_not_summary(response, {"Farah Iqbal"})
    assert response["data"]["level"] == "company"


def test_who_works_in_phrasing(roster_db):
    response = handle_chat_message(roster_db, "Who works in Blue Area", session_id=None)
    _assert_is_roster_not_summary(response)


@pytest.mark.parametrize("phrasing", [
    "all advisors in Blue Area",
    "list advisors in Blue Area",
    "show advisors in Blue Area",
    "advisors in Blue Area",
    "advisors from Blue Area",
    "employees in Blue Area",
    "staff in Blue Area",
    "who works in Blue Area",
    "give me the advisors in Blue Area",
    "name the advisors in Blue Area",
])
def test_every_required_roster_phrase_routes_to_roster(roster_db, phrasing):
    response = handle_chat_message(roster_db, phrasing, session_id=None)
    assert response["type"] == "roster", f"{phrasing!r} routed to {response['type']}"


# =====================================================================
# Entity filters are preserved
# =====================================================================

def test_roster_is_scoped_to_the_named_entity(roster_db):
    """The filter must actually apply — a roster that quietly returns
    everyone is as wrong as a summary."""
    response = handle_chat_message(roster_db, "all advisors in Downtown", session_id=None)
    assert {a["name"] for a in response["data"]["advisors"]} == {"Farah Iqbal"}


def test_roster_prefers_the_most_granular_level(roster_db):
    """A query naming a team implies the team, not its company — the
    narrower answer is the more informative one."""
    plan = build_query_plan("all advisors in blue area", {"team": "Blue Area", "company": "Graana"})
    assert plan.action == "roster"
    assert plan.level == "team"


def test_roster_carries_team_context_when_it_varies(roster_db):
    """A unit-head roster spans teams, so each line needs its team; a
    single-team roster shouldn't repeat it on every line."""
    across = handle_chat_message(roster_db, "all advisors under Fraz Khalid", session_id=None)
    assert across["type"] == "roster"
    assert "North/KPK Region" in across["reply"]


def test_unknown_entity_is_reported_not_guessed(roster_db):
    response = handle_chat_message(roster_db, "all advisors in Nonexistent Place", session_id=None)
    assert response["type"] != "roster"


# =====================================================================
# Must not regress
# =====================================================================

def test_bare_entity_mention_is_still_a_summary(roster_db):
    """"how is Blue Area doing" asks about performance — the aggregate
    answer is correct THERE. Roster detection must not swallow it."""
    response = handle_chat_message(roster_db, "how is Blue Area doing", session_id=None)
    assert response["type"] == "team"


def test_possessive_team_query_is_still_a_nested_breakdown(roster_db):
    response = handle_chat_message(roster_db, "Show Adeel Dogar's team", session_id=None)
    assert response["type"] == "breakdown"


def test_who_is_in_x_team_is_still_a_breakdown(roster_db):
    """Deliberately NOT a roster trigger: that phrasing asks about the
    team's shape and is already served by the nested breakdown. The
    distinction is what the user named — "advisors" enumerates people,
    "team" describes the team."""
    response = handle_chat_message(roster_db, "Who is in Adeel Dogar's team?", session_id=None)
    assert response["type"] == "breakdown"


def test_metric_ranking_still_wins_over_roster(roster_db):
    """"top 5 advisors in Blue Area by revenue" contains "advisors in",
    but the entity SCOPES a ranking rather than being enumerated."""
    response = handle_chat_message(roster_db, "top 5 advisors in Blue Area by revenue", session_id=None)
    assert response["type"] == "leaderboard"


def test_plain_leaderboard_unaffected(roster_db):
    assert handle_chat_message(roster_db, "top 5 advisors by revenue", session_id=None)["type"] == "leaderboard"
