"""P0 SAFETY: a failed parse must not answer a narrower question.

The audit found the worst available failure mode. When the LLM could not
be reached, a complex question was answered by the rule planner's
single-metric reading — and the reply was INDISTINGUISHABLE from a
correct one: a well-formed number, carrying the label of whichever
measure the plan happened to hold. Nothing downstream could tell, and
neither could the reader.

It was observed live: a quota error meant every query ran that path and
still produced confident answers.

The rule is not "never fall back". For most questions the plan IS the
whole question, and refusing them during an outage would make the system
worse, not safer. The rule is that the plan may answer only when it loses
nothing — see nlu_pipeline._semantic_gaps, which reads what deterministic
extraction already produced (how many measures were named, how many
subjects grounded) rather than scanning for keywords.

The second half is metric widening. fallback_reasoning matched a measure
against any window of a sentence, turning "I don't know which measure
this is" into a confident wrong one. The approximate tier is off; the
exact tiers, which guess nothing, remain.
"""

import pytest

from app.database.models import Advisor, Calls, Performance, PerformancePeriod
from app.llm import entity_extractor, nlu_pipeline, semantic_parser
from app.llm.query_ir import MetricRef, QueryIR, Sort


@pytest.fixture()
def org(db_session, monkeypatch):
    for wid, name, team in [(1, "Ayesha Khan", "Blue Area"), (2, "Bilal Ahmed", "Downtown"),
                            (3, "Chand Bibi", "Blue Area")]:
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                               rm="UH", portfolio_lead="ZH", management_lead="BCM",
                               in_master_sheet=True))
        db_session.add(Calls(wid=wid, connects_mtd=wid * 100, answered_calls_mtd=wid))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   cleared=wid * 10, target=100))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    # The provider is DOWN unless a test says otherwise — the state this
    # whole file is about.
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    return db_session


def _resolve(db, text):
    return nlu_pipeline.resolve(text, db, session_id=None)


# =========================== 1. no narrower answer on LLM failure
class TestNoSilentDowngrade:
    @pytest.mark.parametrize("text,lost", [
        ("connects and answered calls of all BCMs", "more than one measure"),
        ("revenue and connects for every team", "more than one measure"),
        ("advisors in Blue Area excluding Downtown", "an exclusion"),
        ("advisors in Blue Area or Downtown", "an either/or"),
    ])
    def test_a_query_the_plan_cannot_hold_is_refused(self, text, lost, org):
        resolution = _resolve(org, text)

        assert resolution.kind == "clarify", f"{text!r} was answered anyway"
        assert "couldn't fully work out" in resolution.clarify_message

    def test_the_refusal_says_what_it_could_not_handle(self, org):
        """A generic apology gives the user nothing to act on."""
        message = _resolve(org, "connects and answered calls of all BCMs").clarify_message
        assert "more than one measure" in message

    def test_the_refusal_states_no_number(self, org):
        """The whole point is that no trustworthy figure exists — the
        reply must not imply one."""
        message = _resolve(org, "advisors in Blue Area excluding Downtown").clarify_message
        assert not any(ch.isdigit() for ch in message)

    def test_the_gap_is_read_from_structure_not_keywords(self, org):
        """"but" and "vs" used to route a query; they say nothing about
        whether the plan can hold it."""
        from app.llm.query_planner import build_query_plan

        entities = entity_extractor.extract_entities("top advisors by connects but quickly", org)
        plan = build_query_plan("top advisors by connects but quickly", entities)
        assert nlu_pipeline._semantic_gaps(
            "top advisors by connects but quickly", entities, plan) == []

    def test_the_parser_declines_to_degrade_rather_than_narrowing(self, org):
        """The other downgrade site: semantic_parser's own degrade, which
        runs before nlu_pipeline sees anything."""
        from app.llm.query_planner import build_query_plan

        text = "connects and answered calls of all BCMs"
        entities = entity_extractor.extract_entities(text, org)
        outcome = semantic_parser.parse(text, entities, org, None,
                                        plan=build_query_plan(text, entities))
        assert outcome.ir is None
        assert outcome.missing == ["understanding"]


# ======================= 2. simple deterministic queries still work
class TestSimpleQueriesUnaffected:
    @pytest.mark.parametrize("text", [
        "top 5 advisors by connects",
        "all advisors in Blue Area",
        "tell me about Ayesha Khan",
        "advisors in Blue Area",
        "who has the most connects",
    ])
    def test_a_query_the_plan_fully_holds_still_answers(self, text, org):
        """With the LLM down. Refusing these would make an outage worse
        than the behaviour being replaced."""
        resolution = _resolve(org, text)
        assert resolution.kind in ("plan", "ir"), f"{text!r} was refused"

    def test_two_conditions_on_different_measures_still_answer(self, org):
        """NOT a downgrade: each threshold carries the measure it was
        written beside, so the plan loses neither. Refusing this would be
        its own kind of wrong."""
        text = "advisors with connects over 100 and answered calls over 1"
        resolution = _resolve(org, text)

        assert resolution.kind != "clarify"
        fields = {f.field for f in (resolution.ir.filter_leaves() if resolution.ir else [])}
        assert {"total_connects", "answered_calls"} <= fields or resolution.kind == "plan"


# ================= 3. unknown metrics/entities clarified, not guessed
class TestNoGuessing:
    def test_the_approximate_metric_tier_is_off(self):
        from app.llm.fallback_reasoning import (
            _APPROXIMATE_WIDENING_ENABLED, fuzzy_resolve_metric,
        )

        assert _APPROXIMATE_WIDENING_ENABLED is False
        # A measure that merely RESEMBLES part of the sentence.
        assert fuzzy_resolve_metric("atendance rate above 90") is None
        assert fuzzy_resolve_metric("revnue") is None

    def test_exact_resolution_is_untouched(self):
        """Only the guessing tier went. An exact registry or token hit is
        the same lookup metric_ontology performs."""
        from app.llm.fallback_reasoning import fuzzy_resolve_metric

        assert fuzzy_resolve_metric("revenue") == "mtd_cleared"
        assert fuzzy_resolve_metric("team size") == "team_size"
        assert fuzzy_resolve_metric("who has the highest sales") == "mtd_cleared"

    def test_a_plural_resolves_exactly_rather_than_by_resemblance(self):
        """Retiring the tier took ordinary plurals with it until they were
        DECLARED. A word form belongs in the alias table, not in an edit
        distance."""
        from app.llm.metric_ontology import resolve_metric

        assert resolve_metric("top performers") == "achievement_pct"
        assert resolve_metric("best performers") == "achievement_pct"

    def test_an_unknown_measure_is_asked_about(self, org):
        resolution = _resolve(org, "top advisors by widget velocity")
        assert resolution.kind == "clarify"

    def test_a_level_word_is_never_read_as_a_measure(self, org):
        """The audit's example of the tier guessing: a metric name
        resembling a hierarchy word."""
        from app.llm.fallback_reasoning import fuzzy_resolve_metric

        assert fuzzy_resolve_metric("advisors in Mars Region") is None
        assert fuzzy_resolve_metric("advisors in North Region") is None


# ==================== 4. valid LLM plans continue normally
class TestValidLLMPlansUnaffected:
    def _llm_returns(self, monkeypatch, ir):
        monkeypatch.setattr(semantic_parser, "call_llm_structured",
                            lambda *a, **k: ir.model_dump())

    def test_a_good_ir_answers_as_before(self, org, monkeypatch):
        self._llm_returns(monkeypatch, QueryIR(
            intent="leaderboard", subject_level="advisor",
            metric=MetricRef(key="total_connects"),
            sort=Sort(metric="total_connects", direction="desc"), limit=10))

        resolution = _resolve(org, "top advisors by connects")
        assert resolution.kind == "ir"
        assert resolution.ir.metric.key == "total_connects"

    def test_the_guard_never_fires_when_the_llm_succeeds(self, org, monkeypatch):
        """The refusal is a FALLBACK behaviour. A query the model handled
        must not be refused for being complex — that would remove the
        capability this whole direction exists to add."""
        self._llm_returns(monkeypatch, QueryIR(
            intent="leaderboard", subject_level="bcm",
            metric=MetricRef(key="total_connects"),
            metrics=[MetricRef(key="total_connects"), MetricRef(key="answered_calls")],
            sort=Sort(metric="total_connects", direction="desc")))

        resolution = _resolve(org, "connects and answered calls of all BCMs")
        assert resolution.kind == "ir", "a complex query the LLM understood was refused"
        assert resolution.ir.metric_keys() == ["total_connects", "answered_calls"]

    def test_deterministic_validation_still_runs_on_the_llm_ir(self, org, monkeypatch):
        """Grounding stays downstream of the model and stays deterministic."""
        from app.llm.query_ir import Filter

        self._llm_returns(monkeypatch, QueryIR(
            intent="leaderboard", metric=MetricRef(key="total_connects"),
            sort=Sort(metric="total_connects"),
            filters=[Filter(field="not_a_field", operator="=", value="x")]))

        resolution = _resolve(org, "top advisors by connects")
        if resolution.ir is not None:
            assert all(f.field != "not_a_field" for f in resolution.ir.filters)
