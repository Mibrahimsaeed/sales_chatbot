"""
Comparison intent, end to end.

Reported failure: "Compare Graana and Agency21" returned the metric-help
message. Root cause — comparison existed DOWNSTREAM (QueryIR
intent="comparison", the compiler's subject filter,
format_ir_comparison_reply) but had no rule-based planner path, so it was
reachable only via the LLM semantic parser. With the LLM unavailable the
query degraded to "unresolved" and, with a metric named, to a plain
leaderboard that silently dropped both entities.

The hard requirement these lock in: a comparison query must NEVER fall
back to the metric-help response.
"""

import pytest

from app.database.models import (
    Advisor, Attendance, Bookings, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message
from app.services.comparison_service import DEFAULT_KPIS, get_comparison

# The exact wording of the fallback this bug produced.
_METRIC_HELP_MARKER = "I'm not tracking that one"


@pytest.fixture()
def comparison_db(db_session, monkeypatch):
    def advisor(wid, name, team, company, connects, meetings, cleared,
                bookings, pipeline, ontime, late):
        db_session.add(Advisor(wid=wid, name=name, team=team, company=company))
        db_session.add(SalesFunnel(
            wid=wid, mtd_new_connect=connects, mtd_followup_connect=0,
            mtd_new_meeting=meetings, mtd_followup_meeting=0, mtd_booking_stored=bookings,
        ))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000, cleared=cleared))
        db_session.add(Pipeline(wid=wid, pipeline=pipeline, overdue=0))
        db_session.add(Bookings(wid=wid, confirmed=bookings, expected=0, token=0))
        db_session.add(Attendance(
            wid=wid, biometric_mtd_ontime=ontime, biometric_mtd_late=late, biometric_mtd_not_marked=0,
        ))

    # Graana leads on connects; Agency21 leads on revenue. A deliberate
    # split so a comparison can't pass by picking one side for everything.
    advisor(1, "G One", "Blue Area", "Graana", 100, 10, 500, 5, 50, 18, 2)
    advisor(2, "G Two", "Blue Area", "Graana", 120, 12, 400, 6, 60, 16, 4)
    advisor(3, "A One", "Downtown", "Agency21", 40, 4, 5000, 2, 20, 10, 10)
    advisor(4, "A Two", "Downtown", "Agency21", 30, 3, 4000, 1, 10, 12, 8)
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


# =====================================================================
# The reported failure, and every required phrasing
# =====================================================================

@pytest.mark.parametrize("phrasing", [
    "Compare Graana and Agency21",
    "Compare Graana vs Agency21",
    "Compare Graana versus Agency21",
    "Difference between Graana and Agency21",
    "Which is performing better, Graana or Agency21?",
    "Who is doing better, Graana or Agency21",
    "How does Graana compare to Agency21",
    "Graana vs Agency21",
    "comparison of Graana and Agency21",
])
def test_every_comparison_phrasing_routes_to_comparison(comparison_db, phrasing):
    response = handle_chat_message(comparison_db, phrasing, session_id=None)
    assert response["type"] == "comparison", f"{phrasing!r} -> {response['type']}"


@pytest.mark.parametrize("phrasing", [
    "Compare Graana and Agency21",
    "Compare Graana vs Agency21",
    "Difference between Graana and Agency21",
    "Which is performing better, Graana or Agency21?",
    "Compare Blue Area and Downtown",
    "Compare Graana and Agency21 by revenue",
    "Compare Blue Area and Downtown attendance",
])
def test_comparison_never_falls_back_to_metric_help(comparison_db, phrasing):
    """The hard requirement — this is the exact failure that was reported."""
    response = handle_chat_message(comparison_db, phrasing, session_id=None)
    assert _METRIC_HELP_MARKER not in response["reply"], f"{phrasing!r} fell back to metric help"


def test_comparison_across_teams(comparison_db):
    response = handle_chat_message(comparison_db, "Compare Blue Area and Downtown", session_id=None)
    assert response["type"] == "comparison"
    assert {e["value"] for e in response["data"]["entities"]} == {"Blue Area", "Downtown"}
    assert all(e["level"] == "team" for e in response["data"]["entities"])


# =====================================================================
# Default KPI set vs a named metric
# =====================================================================

def test_no_metric_returns_the_default_kpi_set(comparison_db):
    """"Compare A and B" is a request for an overview, not one number."""
    response = handle_chat_message(comparison_db, "Compare Graana and Agency21", session_id=None)
    keys = {row["key"] for row in response["data"]["rows"]}
    assert "advisors" in keys                      # headcount
    for kpi in DEFAULT_KPIS:
        assert kpi in keys, f"default KPI {kpi} missing"


def test_named_metric_compares_only_that_metric(comparison_db):
    response = handle_chat_message(
        comparison_db, "Compare Graana and Agency21 by revenue", session_id=None
    )
    assert response["type"] == "comparison"
    assert [row["key"] for row in response["data"]["rows"]] == ["mtd_cleared"]
    assert response["data"]["metric"] == "mtd_cleared"


def test_named_metric_omits_the_headcount_row(comparison_db):
    """Showing headcount beside a single requested metric invites the
    reader to treat it as part of the answer."""
    response = handle_chat_message(
        comparison_db, "Compare Graana and Agency21 by revenue", session_id=None
    )
    assert "advisors" not in {row["key"] for row in response["data"]["rows"]}


def test_attendance_metric_comparison(comparison_db):
    response = handle_chat_message(
        comparison_db, "Compare Blue Area and Downtown attendance", session_id=None
    )
    assert response["type"] == "comparison"
    assert response["data"]["metric"] == "attendance_rate"


# =====================================================================
# Correctness of the numbers
# =====================================================================

def test_values_are_computed_per_entity(comparison_db):
    result = get_comparison(comparison_db, [("company", "Graana"), ("company", "Agency21")])
    by_name = {e["value"]: e for e in result["entities"]}

    assert by_name["Graana"]["advisors"] == 2
    assert by_name["Graana"]["metrics"]["total_connects"] == 220     # 100 + 120
    assert by_name["Agency21"]["metrics"]["total_connects"] == 70    # 40 + 30
    assert by_name["Agency21"]["metrics"]["mtd_cleared"] == 9000     # 5000 + 4000


def test_rate_metrics_are_averaged_not_summed(comparison_db):
    """Regression: the advisor-level binding leaves `agg` at its "sum"
    default (it describes ONE person's value), so reading it directly
    reported Graana's attendance rate as the SUM of its advisors'
    percentages — 15,098% against a real value near 60%. The rollup rule
    lives on the team binding."""
    result = get_comparison(comparison_db, [("company", "Graana"), ("company", "Agency21")])
    for entity in result["entities"]:
        rate = entity["metrics"]["attendance_rate"]
        assert 0 <= rate <= 100, f"{entity['value']} attendance rate {rate} is not a percentage"


def test_winner_is_identified_per_row(comparison_db):
    result = get_comparison(comparison_db, [("company", "Graana"), ("company", "Agency21")])
    assert result["winners"]["total_connects"] == "Graana"    # 220 vs 70
    assert result["winners"]["mtd_cleared"] == "Agency21"     # 9000 vs 900


def test_a_tie_has_no_winner(comparison_db):
    """Marking one side as leading on a tie would be a fabricated claim."""
    result = get_comparison(comparison_db, [("company", "Graana"), ("company", "Graana")])
    assert result["winners"]["total_connects"] is None


def test_unknown_entity_raises_rather_than_rendering_zeros(comparison_db):
    """A column of zeros reads as a real (very poor) result rather than
    as "I couldn't find that"."""
    from app.core.exception import NotFoundError
    with pytest.raises(NotFoundError):
        get_comparison(comparison_db, [("company", "Graana"), ("company", "Nonexistent")])


def test_fewer_than_two_targets_is_rejected(comparison_db):
    with pytest.raises(ValueError):
        get_comparison(comparison_db, [("company", "Graana")])


# =====================================================================
# Entity typing and edge cases
# =====================================================================

def test_entity_types_are_preserved(comparison_db):
    result = get_comparison(comparison_db, [("team", "Blue Area"), ("company", "Agency21")])
    levels = {e["value"]: e["level"] for e in result["entities"]}
    assert levels["Blue Area"] == "team"
    assert levels["Agency21"] == "company"


def test_one_side_missing_says_so_instead_of_answering_about_the_other(comparison_db):
    """Silently answering about the side that resolved is the wrong
    answer to the question asked."""
    response = handle_chat_message(comparison_db, "Compare Graana and Nonexistent", session_id=None)
    assert response["type"] == "clarification"
    assert "Graana" in response["reply"]
    assert _METRIC_HELP_MARKER not in response["reply"]


def test_a_name_grounding_at_two_levels_is_not_compared_with_itself(comparison_db, monkeypatch):
    """Production has a company "Graana" AND an office literally named
    "Graana", so collecting across levels made "compare Graana and
    Agency21" a THREE-way comparison of Graana against itself."""
    from app.llm.query_planner import build_query_plan
    entities = {
        "companies": ["Graana", "Agency21"], "company": "Graana",
        "business_centers": ["Graana"], "business_center": "Graana",
    }
    plan = build_query_plan("compare graana and agency21", entities)
    assert plan.action == "comparison"
    assert plan.comparison_targets == [("company", "Graana"), ("company", "Agency21")]


def test_comparison_reply_is_a_readable_table(comparison_db):
    response = handle_chat_message(comparison_db, "Compare Graana and Agency21", session_id=None)
    reply = response["reply"]
    assert "Graana" in reply and "Agency21" in reply
    assert "Total MTD Connects" in reply
    assert "←" in reply, "the leader on each row should be marked"


def test_leaderboard_is_unaffected(comparison_db):
    """A ranking with no comparison phrase must stay a leaderboard."""
    response = handle_chat_message(comparison_db, "top 3 advisors by revenue", session_id=None)
    assert response["type"] == "leaderboard"
