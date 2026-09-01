"""The four phases from the evaluation report, as regressions.

  1  two silent wrong answers are refused instead of half-answered
  2  a population can be asked for without a measure, and the metric
     join stops shrinking it
  3  a comparison follow-up keeps its subjects — and nothing ELSE starts
     leaking scope, which is the failure the old clear-everything rule
     was protecting against
  4  the LLM is offered the vocabulary it now needs

Phase 1's cases were found by the evaluation, not by a user report: both
answered CONFIDENTLY, one of them getting both halves of the question
wrong. That is the class these tests exist to keep closed.
"""

import pytest

from app.database.models import Advisor, Calls, Performance, PerformancePeriod
from app.llm import entity_extractor, nlu_pipeline, operations, semantic_parser
from app.llm.query_compiler import compile_and_run, count_ir
from app.llm.query_ir import Filter, FilterGroup, MetricRef, QueryIR, Sort


@pytest.fixture()
def org(db_session, monkeypatch):
    # Two advisors deliberately have NO calls row — the population must
    # still contain them, which is the whole of Phase 2.
    people = [(1, "Ayesha Khan", "Blue Area", True), (2, "Bilal Ahmed", "DownTown", True),
              (3, "Chand Bibi", "Blue Area", False), (4, "Danish Ali", "GCC", True),
              (5, "Erum Shah", "GCC", False)]
    for wid, name, team, has_calls in people:
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                               rm="UH", portfolio_lead="ZH", management_lead="BCM",
                               in_master_sheet=True))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   cleared=wid * 10, target=100))
        if has_calls:
            db_session.add(Calls(wid=wid, connects_mtd=wid * 100, answered_calls_mtd=wid))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    return db_session


def _resolve(db, text, session_id=None):
    return nlu_pipeline.resolve(text, db, session_id=session_id)


# ======================================= PHASE 1: no silent wrong answers
class TestPhase1Refusals:
    def test_a_period_comparison_is_refused(self, org):
        """Answered MTD revenue and said nothing about the YTD half."""
        resolution = _resolve(org, "revenue this month versus year to date")
        assert resolution.kind == "clarify"
        assert "two time periods" in resolution.clarify_message

    def test_a_two_part_question_is_refused(self, org):
        """Answered the SECOND part only — and at advisor level, where
        team_size is a per-row literal, so it reported a team size of 1."""
        resolution = _resolve(
            org, "who is the top advisor in Blue Area and what is their team size")
        assert resolution.kind == "clarify"
        assert "two separate questions" in resolution.clarify_message

    def test_the_refusal_names_no_figure(self, org):
        """The point is that no trustworthy number exists."""
        message = _resolve(org, "revenue this month versus year to date").clarify_message
        assert not any(ch.isdigit() for ch in message)

    @pytest.mark.parametrize("text", [
        "revenue this month",
        "revenue year to date",
        "top 5 advisors by connects",
        "connects and answered calls of all BCMs and their teams",
    ])
    def test_one_period_and_one_question_are_not_refused_for_these_reasons(self, text, org):
        """The detectors must not fire on ordinary phrasing. (The last
        one may still be refused for naming two measures — a different
        gap — so only these two reasons are checked.)"""
        from app.llm.query_planner import build_query_plan
        from app.llm.preprocessing import normalize

        cleaned = normalize(text)
        entities = entity_extractor.extract_entities(cleaned, org)
        gaps = nlu_pipeline._semantic_gaps(
            cleaned, entities, build_query_plan(cleaned, entities))
        assert "two time periods compared against each other" not in gaps, text
        assert "two separate questions in one message" not in gaps, text

    def test_a_plan_served_shape_is_also_gap_checked(self, org):
        """THE HOLE the required-query pass found. "list all advisors
        excluding Blue Area" resolves to a ROSTER, which is plan-served
        and therefore answered BEFORE the gap check ran — and a roster
        carries one entity filter with no negation, so it dropped the
        exclusion and returned Blue Area's members: the exact opposite of
        the question, stated confidently."""
        resolution = _resolve(org, "list all advisors excluding Blue Area")
        assert resolution.kind == "clarify", "an exclusion was silently dropped"

    def test_an_ordinary_roster_is_still_served(self, org):
        resolution = _resolve(org, "all advisors in Blue Area")
        assert resolution.kind == "plan"
        assert resolution.plan.action == "roster"

    def test_periods_are_counted_distinctly(self):
        """Two spellings of ONE period are not a comparison."""
        assert nlu_pipeline._periods_named("revenue this month") == {"MTD"}
        assert nlu_pipeline._periods_named("revenue current month this month") == {"MTD"}
        assert nlu_pipeline._periods_named("this month versus year to date") == {"MTD", "YTD"}

    def test_and_alone_is_not_a_second_question(self):
        """"and" is in most compound queries; an interrogative after it
        is what makes two questions."""
        assert not nlu_pipeline._SECOND_QUESTION_RE.search(
            "advisors with achievement below 50 and answered calls below 20")
        assert not nlu_pipeline._SECOND_QUESTION_RE.search("compare Blue Area and DownTown")
        assert nlu_pipeline._SECOND_QUESTION_RE.search(
            "who is top in Blue Area and what is their team size")


# ============================================ PHASE 2: population queries
def _population(**kw):
    fields = dict(intent="filtered_list", operation="population",
                  subject_level="advisor", metric=None, limit=None)
    fields.update(kw)
    return QueryIR(**fields)


class TestPhase2Population:
    def test_a_population_needs_no_metric(self, org):
        rows = compile_and_run(org, _population())
        assert len(rows) == 5
        assert all(r["value"] is None for r in rows)

    def test_the_metric_join_no_longer_shrinks_it(self, org):
        """THE Phase 2 defect. Two of five advisors have no calls row, so
        a connects-ranked "population" returns three."""
        population = compile_and_run(org, _population())
        ranked = compile_and_run(org, QueryIR(
            intent="leaderboard", operation="leaderboard", subject_level="advisor",
            metric=MetricRef(key="total_connects"),
            sort=Sort(metric="total_connects"), limit=None))

        assert len(population) == 5
        assert len(ranked) == 3
        assert {r["name"] for r in population} > {r["name"] for r in ranked}

    def test_not_is_preserved(self, org):
        rows = compile_and_run(org, _population(
            filter_tree=FilterGroup(op="not", children=[
                Filter(field="team", operator="=", value="Blue Area")])))
        assert sorted(r["name"] for r in rows) == ["Bilal Ahmed", "Danish Ali", "Erum Shah"]

    def test_or_is_preserved(self, org):
        rows = compile_and_run(org, _population(
            filter_tree=FilterGroup(op="or", children=[
                Filter(field="team", operator="=", value="Blue Area"),
                Filter(field="team", operator="=", value="DownTown")])))
        assert sorted(r["name"] for r in rows) == ["Ayesha Khan", "Bilal Ahmed", "Chand Bibi"]

    def test_no_artificial_metric_is_introduced(self, org):
        ir = _population()
        compile_and_run(org, ir)
        assert ir.metric is None
        assert ir.metric_keys() == []

    def test_count_matches_the_rows(self, org):
        """count_ir gates pagination, so a population it cannot see would
        page a different set than it counted."""
        ir = _population(filter_tree=FilterGroup(op="not", children=[
            Filter(field="team", operator="=", value="Blue Area")]))
        assert count_ir(org, ir) == len(compile_and_run(org, ir)) == 3

    def test_it_renders_as_names_without_values(self, org):
        from app.llm.response_formatter import format_ir_reply
        from app.llm.response_planner import plan_response

        ir = _population()
        rows = compile_and_run(org, ir)
        reply = format_ir_reply(ir, rows, total_count=len(rows),
                                plan=plan_response(ir, rows))
        assert "Ayesha Khan" in reply
        assert "no data" not in reply

    def test_a_group_level_population_works(self, org):
        rows = compile_and_run(org, _population(subject_level="team"))
        assert sorted(r["name"] for r in rows) == ["Blue Area", "DownTown", "GCC"]

    def test_a_missing_metric_without_the_operation_is_still_refused(self, org):
        """The absence of a metric only means "population" when the IR
        SAYS so. An IR that merely failed to resolve one must not be
        silently answered as an enumeration."""
        assert compile_and_run(org, QueryIR(
            intent="leaderboard", operation="leaderboard",
            subject_level="advisor", metric=None)) is None


# ========================================== PHASE 3: follow-up context
class TestPhase3Context:
    def test_a_comparison_follow_up_keeps_both_subjects(self, org):
        session = "ph3-comparison"
        _resolve(org, "compare Blue Area and DownTown", session)
        resolution = _resolve(org, "what about connects", session)

        assert resolution.ir is not None
        assert sorted(s.value for s in resolution.ir.subjects) == ["Blue Area", "DownTown"]
        assert resolution.ir.resolved_operation() == "comparison"
        assert resolution.ir.metric.key == "total_connects"

    @pytest.mark.parametrize("first", [
        "all advisors in Blue Area",
        "tell me about Ayesha Khan",
    ])
    def test_a_single_entity_turn_still_clears(self, first, org):
        """The narrowness is the point. A profile that left its scope
        behind sent the turn AFTER it back to the wrong team — a leak one
        turn late, and therefore invisible."""
        from app.llm import conversation_memory

        session = f"ph3-clear-{abs(hash(first))}"
        _resolve(org, first, session)
        assert conversation_memory.get(session) is None

    def test_only_comparisons_are_carried(self, org):
        from app.llm.query_planner import QueryPlan

        assert nlu_pipeline._carryable_context(
            QueryPlan(action="roster", level="team", entity_value="Blue Area")) is None
        assert nlu_pipeline._carryable_context(
            QueryPlan(action="lookup", level="advisor", entity_value="Ayesha Khan")) is None
        carried = nlu_pipeline._carryable_context(QueryPlan(
            action="comparison", level="team", entity_value="Blue Area",
            comparison_targets=[("team", "Blue Area"), ("team", "DownTown")]))
        assert carried is not None and len(carried.subjects) == 2

    def test_a_leaderboard_follow_up_still_works(self, org):
        """The IR path's own context handling is untouched."""
        session = "ph3-leaderboard"
        _resolve(org, "top 5 advisors by connects", session)
        resolution = _resolve(org, "what about revenue", session)
        assert resolution.ir is not None
        assert resolution.ir.metric.key in ("mtd_cleared", "ytd_cleared")


# ================================== PHASE 4: what the LLM is offered
class TestPhase4Routing:
    def test_the_llm_is_offered_population(self, org):
        from app.llm.llm_client import QUERY_IR_JSON_SCHEMA as schema

        offered = set(schema["properties"]["operation"]["enum"])
        assert "population" in offered
        # A subset now, not an equality: the enum is the EXECUTABLE half
        # of IR_EXPRESSIBLE. `trend` and `clarify_metric` are expressible
        # and dispatch to "unsupported"/"clarification", so the model may
        # no longer select them.
        assert offered <= operations.IR_EXPRESSIBLE

    def test_the_prompt_explains_population_and_multi_part(self):
        from app.llm.prompt_builder import _ir_schema

        text = _ir_schema()
        assert "POPULATION vs RANKING" in text
        assert "TWO QUESTIONS IN ONE MESSAGE" in text

    def test_the_prompt_warns_against_inventing_a_metric(self):
        """The Phase 2 defect, stated where the model can act on it."""
        from app.llm.prompt_builder import _ir_schema

        assert "Do NOT invent a measure to rank a population by" in _ir_schema()

    def test_analytical_queries_reach_the_llm(self, org, monkeypatch):
        calls = []
        monkeypatch.setattr(semantic_parser, "call_llm_structured",
                            lambda p, s, schema_name=None: calls.append(p) or None)
        _resolve(org, "top advisors by connects")
        assert calls

    def test_plan_only_shapes_do_not(self, org, monkeypatch):
        calls = []
        monkeypatch.setattr(semantic_parser, "call_llm_structured",
                            lambda p, s, schema_name=None: calls.append(p) or None)
        _resolve(org, "all advisors in Blue Area")
        assert not calls

    def test_the_prompt_never_asks_for_sql(self, org, monkeypatch):
        captured = []
        monkeypatch.setattr(semantic_parser, "call_llm_structured",
                            lambda p, s, schema_name=None: captured.append(p) or None)
        _resolve(org, "top advisors by connects")
        prompt = captured[0].lower()
        assert "select " not in prompt and " sql" not in prompt

    def test_grounding_runs_before_the_llm_and_is_handed_to_it(self, org, monkeypatch):
        """Entity resolution stays deterministic and upstream."""
        captured = []
        monkeypatch.setattr(semantic_parser, "call_llm_structured",
                            lambda p, s, schema_name=None: captured.append(p) or None)
        _resolve(org, "top advisors in Blue Area by connects")
        assert "Blue Area" in captured[0]
        assert "rule-based grounding" in captured[0]
