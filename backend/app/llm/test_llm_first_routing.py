"""P1: the LLM leads, the rule planner catches.

BEFORE, whether the model was consulted depended on whether the sentence
happened to contain one of `but|compare|vs|still|and also|except|
excluding` (semantic_parser.looks_compound). A question that was complex
without using one of those words was answered by the rule planner's
single-metric reading, and the model that could have understood it was
never asked. The keyword list was the architecture.

AFTER, the gate is a CAPABILITY question — does any QueryIR express this
answer shape? Three families do not, and for them the plan is not a fast
path but the only path:

    the profile / summary CARDS   many measures at once
    the hierarchy READS           roster, ancestry, manager, direct reports
    the CLARIFICATIONS            which person was meant — the LLM has no WIDs

Everything else goes to the LLM first, unconditionally.

THE FALLBACK IS THE OTHER HALF. Inverting the order is only safe because
a failed parse now serves the rule plan instead of answering "I'm not
tracking that one" — which is what it did before, and which would make
every outage worse than the behaviour being replaced. Most of this file
runs with the LLM unreachable, because that is the state the fallback
exists for.
"""

import pytest

from app.database.models import Advisor, Calls, Performance, PerformancePeriod
from app.llm import entity_extractor, nlu_pipeline, semantic_parser
from app.llm.query_ir import MetricRef, QueryIR, Sort


@pytest.fixture()
def org(db_session, monkeypatch):
    for wid, name, team in [(1, "Ayesha Khan", "Alpha"), (2, "Bilal Ahmed", "Bravo"),
                            (3, "Chand Bibi", "Alpha")]:
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                               rm="UH", portfolio_lead="ZH", management_lead="BCM",
                               in_master_sheet=True))
        db_session.add(Calls(wid=wid, connects_mtd=wid * 100, answered_calls_mtd=wid))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   cleared=wid * 10, target=100))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    # Default for this file: the provider is DOWN. Tests that need it up
    # install their own stub.
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    return db_session


def _llm_returns(monkeypatch, ir: QueryIR):
    """Stub the ONE structured call semantic_parser makes."""
    calls = []

    def fake(prompt, schema, schema_name=None):
        calls.append(prompt)
        return ir.model_dump()

    monkeypatch.setattr(semantic_parser, "call_llm_structured", fake)
    return calls


def _resolve(db, text, session_id=None):
    return nlu_pipeline.resolve(text, db, session_id=session_id)


# ===================================================== the gate itself
class TestCapabilityGate:
    def test_keyword_free_complex_queries_still_reach_the_llm(self, org, monkeypatch):
        """The defect the inversion removes: no keyword, so the old gate
        kept the query away from the model."""
        calls = _llm_returns(monkeypatch, QueryIR(
            intent="leaderboard", subject_level="advisor",
            metric=MetricRef(key="total_connects"),
            sort=Sort(metric="total_connects", direction="desc")))

        _resolve(org, "who are the strongest performers this month")
        assert calls, "an analytical query must reach the LLM"

    def test_the_keyword_list_no_longer_decides(self, org, monkeypatch):
        """"but" used to be the whole reason a query was escalated."""
        calls = _llm_returns(monkeypatch, QueryIR(
            intent="leaderboard", metric=MetricRef(key="total_connects"),
            sort=Sort(metric="total_connects")))

        _resolve(org, "top advisors by connects")
        with_keyword = len(calls)
        _resolve(org, "top advisors by connects but only good ones")
        assert len(calls) == with_keyword + 1, "both forms go to the LLM now"

    @pytest.mark.parametrize("text,action", [
        ("tell me about Ayesha Khan", "lookup"),
        ("all advisors in Alpha", "roster"),
        ("who is Ayesha Khan's unit head", "reverse_hierarchy"),
    ])
    def test_shapes_no_ir_expresses_stay_on_the_plan(self, text, action, org, monkeypatch):
        """A card or a hierarchy read has no IR to be parsed into, so the
        LLM is not consulted at all."""
        calls = _llm_returns(monkeypatch, QueryIR(intent="leaderboard"))

        resolution = _resolve(org, text)
        assert resolution.kind == "plan"
        assert resolution.plan.action == action
        assert not calls, f"{text!r} should never reach the LLM"

    def test_a_card_shape_escalates_when_several_entities_ground(self, org, monkeypatch):
        """A single-entity card cannot answer a two-entity question, so
        the plan is no longer authoritative — decided from what GROUNDED,
        not from whether the sentence contains a keyword."""
        assert nlu_pipeline._names_several_entities({"teams": ["Alpha", "Bravo"]})
        assert not nlu_pipeline._names_several_entities({"teams": ["Alpha"]})

    def test_plan_only_is_derived_not_restated(self):
        """An action added to _RULE_BASED_ACTIONS cannot fall out of the
        capability reasoning by being forgotten here."""
        assert nlu_pipeline._PLAN_ONLY_ACTIONS == frozenset(nlu_pipeline._RULE_BASED_ACTIONS)


# ================================================== the LLM leads
class TestLLMIsPrimary:
    def test_the_llm_ir_is_what_answers(self, org, monkeypatch):
        """Not merely consulted — its IR is the one that runs. The stub
        sorts ASCENDING, which the rule planner would never produce for
        this wording, so the order proves whose IR executed."""
        _llm_returns(monkeypatch, QueryIR(
            intent="leaderboard", subject_level="advisor",
            metric=MetricRef(key="total_connects"),
            sort=Sort(metric="total_connects", direction="asc"), limit=10))

        resolution = _resolve(org, "top advisors by connects")
        assert resolution.kind == "ir"
        assert resolution.ir.sort.direction == "asc"

    def test_the_llm_decides_the_operation(self, org, monkeypatch):
        """The IR's own intent survives — the rule planner's action does
        not overrule it."""
        _llm_returns(monkeypatch, QueryIR(
            intent="filtered_list", subject_level="advisor",
            metric=MetricRef(key="total_connects"),
            sort=Sort(metric="total_connects")))

        resolution = _resolve(org, "advisors with connects over 100")
        assert resolution.kind == "ir"
        assert resolution.ir.intent == "filtered_list"

    def test_the_prompt_carries_the_grounded_entities(self, org, monkeypatch):
        """Deterministic grounding still runs FIRST and is handed to the
        model — the LLM leads planning, not entity resolution."""
        calls = _llm_returns(monkeypatch, QueryIR(
            intent="leaderboard", metric=MetricRef(key="total_connects"),
            sort=Sort(metric="total_connects")))

        _resolve(org, "top advisors in Alpha by connects")
        assert calls and "Alpha" in calls[0]
        assert "rule-based grounding" in calls[0]


# ============================================ the fallback is the safety
class TestRulePlannerFallback:
    def test_an_analytical_query_still_answers_with_the_llm_down(self, org):
        """The whole file's default: no provider. This must not become
        "I'm not tracking that one"."""
        resolution = _resolve(org, "top advisors by connects")
        assert resolution.kind in ("plan", "ir")
        assert resolution.kind != "clarify"

    def test_a_failed_parse_serves_the_plan_rather_than_giving_up(self, org, monkeypatch):
        """The behaviour the inversion depends on. Before P1 a parse that
        produced no IR returned the give-up message, discarding a plan
        the rule planner had already built correctly — the same defect
        class the audit found for a missing action registry entry."""
        monkeypatch.setattr(semantic_parser, "parse",
                            lambda *a, **k: semantic_parser.ParseOutcome(
                                ir=None, missing=["intent"], used_llm=False))

        resolution = _resolve(org, "top advisors by connects")
        assert resolution.kind == "plan"
        assert resolution.plan.action == "leaderboard"

    def test_a_genuinely_unresolvable_query_still_says_so(self, org, monkeypatch):
        """The fallback must not turn "I don't understand" into a
        confident wrong answer — an unresolved plan is not an answer."""
        monkeypatch.setattr(semantic_parser, "parse",
                            lambda *a, **k: semantic_parser.ParseOutcome(
                                ir=None, missing=["intent"], used_llm=False))

        resolution = _resolve(org, "what is the airspeed of a swallow")
        assert resolution.kind == "clarify"


# ================================================== rollback + invariants
class TestRollbackAndInvariants:
    def test_rules_first_restores_the_previous_gate(self, org, monkeypatch):
        """NLU_MODE is the rollback. Under it the old keyword gate runs
        verbatim, so a rule-servable action is served without the LLM."""
        monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
        calls = _llm_returns(monkeypatch, QueryIR(intent="leaderboard"))

        resolution = _resolve(org, "tell me about Ayesha Khan")
        assert resolution.kind == "plan"
        assert not calls

    def test_greetings_never_reach_the_planner(self, org, monkeypatch):
        """Shortcuts stay outside the analytical path entirely."""
        calls = _llm_returns(monkeypatch, QueryIR(intent="leaderboard"))

        assert _resolve(org, "hello").kind == "shortcut"
        assert not calls

    def test_the_llm_never_produces_sql(self, org, monkeypatch):
        """The safety property the inversion must not weaken: the model
        populates a validated IR and the compiler builds every query."""
        calls = _llm_returns(monkeypatch, QueryIR(
            intent="leaderboard", metric=MetricRef(key="total_connects"),
            sort=Sort(metric="total_connects")))

        _resolve(org, "top advisors by connects")
        prompt = calls[0].lower()
        assert "select " not in prompt and "sql" not in prompt

    def test_validation_still_runs_on_the_llm_ir(self, org, monkeypatch):
        """Grounding is deterministic and downstream of the model: an
        ungroundable filter field is recorded, not executed."""
        from app.llm.query_ir import Filter

        _llm_returns(monkeypatch, QueryIR(
            intent="leaderboard", metric=MetricRef(key="total_connects"),
            sort=Sort(metric="total_connects"),
            filters=[Filter(field="not_a_field", operator="=", value="x")]))

        resolution = _resolve(org, "top advisors by connects")
        # The bad filter never reaches the compiler, whichever way the
        # turn was finally resolved.
        if resolution.ir is not None:
            assert all(f.field != "not_a_field" for f in resolution.ir.filters)
