from app.llm.query_ir import MetricRef, QueryIR, Sort
from app.llm.response_planner import plan_response


def _ir(intent="leaderboard", **overrides):
    base = dict(
        intent=intent,
        subject_level="advisor",
        metric=MetricRef(key="mtd_cleared"),
        sort=Sort(metric="mtd_cleared", direction="desc"),
        limit=10,
    )
    base.update(overrides)
    return QueryIR(**base)


ROW = {"wid": 1, "name": "A", "value": 100.0}


def test_empty_rows_are_empty_shape_regardless_of_intent():
    plan = plan_response(_ir(), [])
    assert plan.shape == "empty"
    assert plan.show_insights is False


def test_single_leaderboard_row_is_single_value():
    plan = plan_response(_ir(), [ROW])
    assert plan.shape == "single_value"
    assert plan.show_insights is False


def test_multi_row_leaderboard_is_ranked_list():
    rows = [ROW, {"wid": 2, "name": "B", "value": 50.0}]
    plan = plan_response(_ir(), rows)
    assert plan.shape == "ranked_list"


def test_comparison_intent_is_comparison_table():
    rows = [ROW, {"wid": 2, "name": "B", "value": 50.0}]
    plan = plan_response(_ir(intent="comparison"), rows)
    assert plan.shape == "comparison_table"


def test_filtered_list_intent_is_filtered_table():
    rows = [ROW, {"wid": 2, "name": "B", "value": 50.0}]
    plan = plan_response(_ir(intent="filtered_list"), rows)
    assert plan.shape == "filtered_table"


def test_insights_only_suggested_with_at_least_three_rows():
    two_rows = [ROW, {"wid": 2, "name": "B", "value": 50.0}]
    three_rows = two_rows + [{"wid": 3, "name": "C", "value": 25.0}]
    assert plan_response(_ir(), two_rows).show_insights is False
    assert plan_response(_ir(), three_rows).show_insights is True


# =====================================================================
# Phase 3 — response MODE, capability awareness, single ownership
# =====================================================================
#
# The tests above pin the rendering SHAPE, which existed before Phase 3.
# What did not exist was a response MODE: chat_service returned
# `"type": ir.intent`, passing the QUESTION's structure through as the
# ANSWER's kind, and `single_value` shared a formatter with
# `ranked_list` so even the correct shape was rendered as a leaderboard.
#
#     "What is Downtown revenue?"
#       before: "🏆 Top 1 by MTD Revenue Cleared" / 1. Downtown — 1,100
#       after:  "Downtown has 1,100 MTD Revenue Cleared."

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import (
    Advisor, Attendance, Calls, Performance, PerformancePeriod, Pipeline,
    Portfolio, SalesFunnel, TeamTarget,
)
from app.database.session import Base
from app.llm import entity_extractor, routing
from app.services.chat_service import handle_chat_message

_ROWS = [ROW, {"wid": 2, "name": "B", "value": 50.0},
         {"wid": 3, "name": "C", "value": 25.0}]


def test_one_row_is_a_metric_value_not_a_leaderboard():
    plan = plan_response(_ir(), [ROW])
    assert plan.mode == "metric_value"
    assert any(m == "leaderboard" for m, _ in plan.rejected)


def test_several_rows_are_a_leaderboard():
    plan = plan_response(_ir(), _ROWS)
    assert plan.mode == "leaderboard"
    assert any(m == "metric_value" for m, _ in plan.rejected)


def test_no_rows_is_no_data_not_unsupported():
    """Different answers: "nothing matched" invites a rephrase, "I can't
    do that" does not."""
    plan = plan_response(_ir(), [])
    assert plan.mode == "no_data"
    assert any(m == "unsupported" for m, _ in plan.rejected)


def test_an_unsupported_intent_outranks_rows_and_emptiness():
    """A capability limit is decided first: answering the wrong question
    well is worse than saying no."""
    plan = plan_response(_ir(intent="trend"), _ROWS)
    assert plan.mode == "unsupported"
    assert plan.reason


def test_the_capability_reason_comes_from_the_registry():
    """One list of what this system cannot do, one wording each."""
    from app.llm.ir_validator import _UNSUPPORTED_INTENTS

    assert plan_response(_ir(intent="trend"), []).reason == _UNSUPPORTED_INTENTS["trend"]


def test_a_single_value_suppresses_the_narrative_explanation():
    """The explanation and the single-value sentence state the same fact;
    prepending both printed the figure twice."""
    assert plan_response(_ir(), [ROW]).show_explanation is False
    assert plan_response(_ir(), _ROWS).show_explanation is True


def test_every_plan_explains_itself():
    for intent, rows in (("leaderboard", [ROW]), ("leaderboard", _ROWS),
                         ("comparison", _ROWS), ("filtered_list", _ROWS),
                         ("leaderboard", []), ("trend", _ROWS)):
        plan = plan_response(_ir(intent=intent), rows)
        assert plan.why, f"{intent}/{len(rows)} gave no reason"
        assert plan.trace()


def test_the_planner_never_reads_user_text():
    """Inputs are QueryIR + rows + capability. Re-reading the text here
    would be a second, competing parser."""
    import inspect

    source = inspect.getsource(plan_response)
    assert "text" not in source


def test_single_value_and_ranked_list_render_differently():
    """They shared a formatter, so the planner's decision had no effect."""
    from app.llm.response_formatter import _SHAPE_FORMATTERS

    assert _SHAPE_FORMATTERS["single_value"] is not _SHAPE_FORMATTERS["ranked_list"]


def test_chat_service_does_not_derive_the_response_type_from_intent():
    """The competing owner Phase 3 removed."""
    import inspect

    from app.services import chat_service

    source = inspect.getsource(chat_service._dispatch_ir)
    assert '"type": ir.intent' not in source
    assert "response_plan.mode" in source


# ---------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------

_PEOPLE = [(1, "Yasir Ali", "Blue Area", "Graana"),
           (2, "Waqar Haider", "Blue Area", "Graana"),
           (3, "Shehryar Abbasi", "Downtown", "Graana"),
           (4, "Hina Malik", "Downtown", "IMARAT")]


@pytest.fixture(scope="module")
def _rp_engine():
    from conftest import _ADVISOR_PROFILE_VIEW

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    for wid, name, team, company in _PEOPLE:
        s.add(Advisor(wid=wid, name=name, team=team, company=company,
                      rm="Tariq Mehmood", portfolio_lead="Fawad Hafeez",
                      management_lead="Usman Ghani", office="Beverly Center",
                      region="North/KPK", unit="A", in_master_sheet=True))
        s.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000,
                          cleared=100 * wid, pct=10.0 * wid))
        s.add(Performance(wid=wid, period=PerformancePeriod.YTD, target=10000,
                          cleared=1000 * wid, pct=10.0 * wid))
        s.add(SalesFunnel(wid=wid, mtd_new_connect=10 * wid, mtd_followup_connect=0,
                          mtd_cr=5 * wid, mtd_new_meeting=wid,
                          mtd_followup_meeting=0, mtd_conversion=wid,
                          mtd_booking_stored=wid))
        s.add(Pipeline(wid=wid, pipeline=1000 * wid, overdue=wid))
        s.add(Portfolio(wid=wid, value=5000 * wid))
        s.add(Calls(wid=wid, answered_calls_mtd=20 * wid, connects_mtd=10 * wid))
        s.add(Attendance(wid=wid, biometric_mtd_ontime=15, biometric_mtd_late=5,
                         biometric_mtd_not_marked=0, login_mtd_ontime=18,
                         login_mtd_late=2, login_mtd_not_marked=0))
    for team, target, achieved in (("Blue Area", 3000, 2400), ("Downtown", 2000, 1100)):
        s.add(TeamTarget(team=team, target=target, achieved=achieved,
                         achievement_pct=round(achieved / target * 100, 1)))
    s.commit()
    with engine.begin() as conn:
        conn.exec_driver_sql(_ADVISOR_PROFILE_VIEW)
    yield engine
    s.close()


@pytest.fixture()
def org(_rp_engine, monkeypatch):
    from app.core.config import settings
    from app.llm import llm_client, semantic_parser

    monkeypatch.setattr(settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(settings, "use_llm_planner", False, raising=False)
    monkeypatch.setattr(llm_client, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)

    s = sessionmaker(bind=_rp_engine)()
    entity_extractor._cache["loaded_at"] = 0
    yield s
    s.close()


@pytest.mark.parametrize("query", [
    "What is Downtown revenue?",
    "What is Blue Area's pipeline value?",
    "Total connects for Blue Area",
])
def test_a_group_metric_question_answers_as_a_single_value(org, query):
    r = handle_chat_message(org, query, session_id=None)
    assert r["type"] == "metric_value"
    assert "🏆" not in r["reply"], "rendered as a leaderboard despite the mode"
    assert "Top 1" not in r["reply"]


def test_a_single_value_reply_states_the_figure_once(org):
    r = handle_chat_message(org, "What is Downtown revenue?", session_id=None)
    assert r["reply"].count("MTD Revenue Cleared") == 1


def test_a_ranking_answers_as_a_leaderboard(org):
    r = handle_chat_message(org, "Top advisors by revenue", session_id=None)
    assert r["type"] == "leaderboard"
    assert "🏆" in r["reply"]


@pytest.mark.parametrize("query", [
    "Show the trend of revenue",
    "Is Yasir Ali improving?",
    "revenue month over month",
    "show me the history of connects",
])
def test_trend_questions_are_refused_honestly(org, query):
    """Never silently substituted with a snapshot — before Phase 3 "show
    the trend of revenue" answered with a CURRENT ranking."""
    r = handle_chat_message(org, query, session_id=None)
    assert r["type"] == "unsupported"
    assert "trends over time" in r["reply"]
    # a refusal nothing can resolve must not offer options
    assert not r.get("options")


def test_an_unsupported_answer_is_not_a_clarification(org):
    r = handle_chat_message(org, "Show the trend of revenue", session_id=None)
    assert r["type"] != "clarification"


def test_a_clarification_is_still_a_clarification(org):
    # portfolio %: CR % is a computable rate since working_days.py, so
    # it can no longer stand in for "a clarification".
    r = handle_chat_message(org, "What is Downtown's portfolio %?", session_id=None)
    assert r["type"] == "clarification"


def test_a_hierarchy_summary_is_unaffected(org):
    assert handle_chat_message(org, "How is Blue Area performing?",
                               session_id=None)["type"] == "team"


def test_a_profile_request_is_unaffected(org):
    assert handle_chat_message(org, "Tell me about Yasir Ali",
                               session_id=None)["type"] == "advisor"


def test_the_trace_records_the_mode_and_the_rejected_alternative(org):
    handle_chat_message(org, "What is Downtown revenue?", session_id=None)
    step = next(s for s in routing.current_trace().steps if s.stage == "Response")
    assert step.chose == "metric_value"
    assert "not leaderboard" in step.why


def test_the_trace_records_an_unsupported_refusal(org):
    handle_chat_message(org, "Show the trend of revenue", session_id=None)
    step = next(s for s in routing.current_trace().steps if s.stage == "Response")
    assert step.chose == "unsupported"


# =====================================================================
# The response-type / data-shape contract
# =====================================================================
#
# ONE WIRE TYPE, ONE DATA SHAPE. A client picks its renderer from
# `response["type"]`, so a type that carries two incompatible shapes is
# not consumable — it forces every client to sniff the payload before it
# can draw anything.
#
# This is not hypothetical. `filtered_list` and `population` both used to
# report as "breakdown", which already meant the hierarchy breakdown and
# carries a NESTED OBJECT (level_label, teams[], mtd_cleared). Those two
# carry a flat ARRAY of rows. The frontend's BreakdownCard read `b.teams`
# off an array, got undefined, rendered an empty shell — and because a
# card was assumed present the reply text was suppressed, so a fully
# correct backend answer arrived at the user as a blank message.
#
# These tests pin the separation so it cannot be undone silently.

class TestResponseTypeDataShapeContract:
    def test_filtered_list_does_not_report_as_a_hierarchy_breakdown(self):
        rows = [ROW, {"wid": 2, "name": "B", "value": 50.0}]
        plan = plan_response(_ir(intent="filtered_list"), rows)
        assert plan.mode == "filtered_list"
        assert plan.mode != "breakdown", (
            "filtered_list carries an array of rows; 'breakdown' carries the "
            "hierarchy object. Sharing the type makes the payload unrenderable."
        )

    def test_population_does_not_report_as_a_hierarchy_breakdown(self):
        rows = [{"wid": 1, "name": "A"}, {"wid": 2, "name": "B"}]
        ir = _ir(intent="filtered_list", operation="population", metric=None,
                 sort=Sort(metric=None, direction="desc"))
        plan = plan_response(ir, rows)
        assert plan.mode == "population"
        assert plan.mode != "breakdown"

    def test_population_stays_distinct_from_filtered_list(self):
        """A population has NO measure. Collapsing it into filtered_list
        would let a client render it as a ranking, printing "no data"
        beside every name — the regression plan_response's own comment
        warns about."""
        rows = [{"wid": 1, "name": "A"}, {"wid": 2, "name": "B"}]
        ranked = plan_response(_ir(intent="filtered_list"), rows)
        population = plan_response(
            _ir(intent="filtered_list", operation="population", metric=None,
                sort=Sort(metric=None, direction="desc")), rows)
        assert ranked.mode != population.mode

    def test_both_still_render_as_a_filtered_table(self):
        """The wire type changed; the RENDERING did not. `shape` is what
        the formatter dispatches on, and both remain a plain list."""
        rows = [ROW, {"wid": 2, "name": "B", "value": 50.0}]
        assert plan_response(_ir(intent="filtered_list"), rows).shape == "filtered_table"
        assert plan_response(
            _ir(intent="filtered_list", operation="population", metric=None,
                sort=Sort(metric=None, direction="desc")), rows).shape == "filtered_table"

    def test_every_operation_dispatch_mode_is_known_to_the_planner(self):
        """The registry cross-check, asserted here too: an operation may
        not invent a response type."""
        from app.llm.operations import OPERATIONS
        from app.llm.response_planner import DISPATCH_MODES

        unknown = {op.name: op.dispatch_mode for op in OPERATIONS.values()
                   if op.dispatch_mode not in DISPATCH_MODES}
        assert not unknown, f"operations dispatch as modes the planner does not know: {unknown}"

    def test_no_wire_type_is_shared_by_operations_with_different_payloads(self):
        """THE contract, stated structurally.

        `breakdown` is the type whose payload is an object; every other
        row-list operation must have a type of its own. Listed explicitly
        rather than derived, so adding an operation to the object-shaped
        family is a deliberate edit rather than an accident.
        """
        from app.llm.operations import OPERATIONS
        from app.llm.response_planner import DISPATCH_MODES

        OBJECT_PAYLOAD = {"breakdown", "lookup", "summary", "roster",
                          "reverse_hierarchy", "ancestry", "direct_reports",
                          "scoped_reports", "group_metric", "advisor_metric"}
        ROW_LIST_PAYLOAD = {"leaderboard", "filtered_list", "population", "comparison"}

        wire_of = {name: DISPATCH_MODES[op.dispatch_mode]
                   for name, op in OPERATIONS.items()}
        for row_op in ROW_LIST_PAYLOAD:
            clash = [o for o in OBJECT_PAYLOAD
                     if o in wire_of and wire_of[o] == wire_of[row_op]]
            assert not clash, (
                f"{row_op!r} (array payload) shares wire type "
                f"{wire_of[row_op]!r} with {clash} (object payload) — a client "
                "cannot pick a renderer from the type alone"
            )
