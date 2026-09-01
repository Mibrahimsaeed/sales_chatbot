"""ONE authoritative field decides what a query is.

THE DEFECT. `operation` and `intent` are two names for the same
decision. `resolved_operation()` prefers `operation`, and the compiler,
the response planner and chat_service._dispatch all read it — but the
model emitted both fields independently and nothing made them agree, so
a query could arrive with a coherent operation and an intent that meant
something else. Whichever field a given consumer happened to read then
decided the answer.

Three things made that reachable rather than theoretical:

  - The prompt told the model to leave `operation` NULL and "communicate
    the uncertainty" through `intent`. On exactly the queries it was
    least sure about, the model discarded the authoritative field.
  - Two operations that ANSWER — `population` and `group_metric` — have
    no `ir_intent`, so no intent value corresponds to them. A query whose
    correct operation was one of those had no consistent intent
    available, and the model picked a contradictory one.
  - The intent enum offered four values that could not lead to an answer:
    `lookup` and `trend` (rejected outright), `clarify` (asks a question)
    and `breakdown` (routed away from the compiler, discarding the
    metric).

So "what is X's <measure>?" — parsed with the right metric, the right
subject and the right level — was answered with a generic entity
summary, or refused, depending on which dead value the model reached for.

Both enums are now derived from the operation registry: the model may
select an operation only if the IR can express it AND the dispatcher can
execute it, and an intent only if some offered operation declares it.
The validator then re-derives `intent` from `operation`, so the two
cannot disagree downstream.

Entities here are synthetic. The point is the shape of the contract.
"""

import pytest

from app.llm import operations
from app.llm.ir_validator import _UNSUPPORTED_INTENTS, validate_ir
from app.llm.llm_client import (
    _NON_EXECUTABLE_DISPATCH,
    QUERY_IR_JSON_SCHEMA,
)
from app.llm.prompt_builder import _operation_union
from app.llm.query_ir import Filter, FilterGroup, Intent, MetricRef, QueryIR, Sort

OFFERED_OPS = QUERY_IR_JSON_SCHEMA["properties"]["operation"]["enum"]
OFFERED_INTENTS = QUERY_IR_JSON_SCHEMA["properties"]["intent"]["enum"]

TEAM_A, TEAM_B = "Team Alpha", "Team Beta"


def _ir(**overrides):
    base = dict(intent="filtered_list", operation="filtered_list",
                subject_level="advisor", sort=Sort(metric=None), limit=None)
    base.update(overrides)
    return QueryIR(**base)


def _missing(db, ir):
    return validate_ir(ir, db).missing


# =====================================================================
# A. Every operation offered to the model can actually execute
# =====================================================================
class TestTheGrammarOffersOnlyExecutableOperations:

    def test_operation_is_required_and_not_nullable(self):
        """The prompt used to invite a null here, and the model took the
        invitation on precisely the queries that then failed."""
        assert "operation" in QUERY_IR_JSON_SCHEMA["required"]
        assert QUERY_IR_JSON_SCHEMA["properties"]["operation"]["type"] == "string"
        assert "anyOf" not in QUERY_IR_JSON_SCHEMA["properties"]["operation"]

    @pytest.mark.parametrize("name", OFFERED_OPS)
    def test_every_offered_operation_is_expressible_and_executable(self, name):
        op = operations.OPERATIONS[name]
        assert op.expressible_in_ir, f"{name} cannot be expressed in the IR"
        assert op.dispatch_mode not in _NON_EXECUTABLE_DISPATCH, (
            f"{name} dispatches to {op.dispatch_mode!r}, which never produces "
            "an answer — the model could select it and the system could not "
            "serve it"
        )

    def test_no_dead_operation_is_offered(self):
        """`trend` (no historical snapshot to diff) and `clarify_metric`
        (it IS the question) are the two that were."""
        dead = {n for n, op in operations.OPERATIONS.items()
                if op.dispatch_mode in _NON_EXECUTABLE_DISPATCH}
        assert not (set(OFFERED_OPS) & dead), sorted(set(OFFERED_OPS) & dead)

    def test_the_prompt_offers_exactly_what_the_grammar_permits(self):
        """CONTRACT 3. The set the prompt ADVERTISES and the set
        grammar-constrained decoding ENFORCES must be one set.

        They were two. `prompt_builder._operation_union()` rendered
        `operations.IR_EXPRESSIBLE` (seven names) while the enum came from
        `llm_client._ir_operations()` (five), so the prompt told the model
        it could choose `trend` or `clarify_metric` and strict decoding
        then made both unrepresentable. An instruction the model can only
        disobey is worse than no instruction: it spends attention and
        yields a value the prompt never described.

        Compared as a SET of names rather than as the rendered string, so
        the union's formatting stays free to change.
        """
        import re

        advertised = set(re.findall(r'"([a-z_]+)"', _operation_union()))
        assert advertised == set(OFFERED_OPS), (
            "the prompt and the grammar disagree about which operations "
            f"exist — prompt-only: {sorted(advertised - set(OFFERED_OPS))}, "
            f"grammar-only: {sorted(set(OFFERED_OPS) - advertised)}"
        )

    def test_the_model_can_say_it_does_not_know(self):
        """An operation set whose every member ASSERTS something leaves an
        unsure model no representable way to be unsure, so it emits its
        best guess instead. A confident wrong answer displacing an honest
        question is the worst trade this pipeline can make, and it was the
        only one available."""
        assert "clarify_metric" in OFFERED_OPS
        assert operations.OPERATIONS["clarify_metric"].ir_intent in OFFERED_INTENTS

    def test_the_answering_operations_are_all_reachable(self):
        """A guard that deleted too much would also pass the checks
        above. These five are the shapes the system answers with — and
        `population` and `group_metric` are the two that were previously
        unreachable from the intent vocabulary."""
        assert {"leaderboard", "comparison", "filtered_list",
                "population", "group_metric"} <= set(OFFERED_OPS)


# =====================================================================
# B. intent is derived, and cannot contradict operation
# =====================================================================
class TestIntentIsDerivedFromOperation:

    @pytest.mark.parametrize("name", OFFERED_INTENTS)
    def test_every_offered_intent_belongs_to_an_offered_operation(self, name):
        owners = [n for n in OFFERED_OPS if operations.OPERATIONS[n].ir_intent == name]
        assert owners, f"{name} is selectable but no offered operation declares it"

    def test_no_offered_intent_is_one_the_validator_refuses(self):
        """THE NON-DRIFT GUARD. The enum and _UNSUPPORTED_INTENTS are
        maintained in different modules; deriving the enum from the
        registry cannot see the validator's refusals, so the invariant is
        asserted rather than assumed."""
        clash = set(OFFERED_INTENTS) & set(_UNSUPPORTED_INTENTS)
        assert not clash, sorted(clash)

    def test_every_offered_intent_is_one_the_ir_can_hold(self):
        assert set(OFFERED_INTENTS) <= set(Intent.__args__)

    @pytest.mark.parametrize("name", OFFERED_OPS)
    def test_validation_derives_the_intent_the_registry_declares(self, db_session, name):
        """operation=X  ->  intent == OPERATIONS[X].ir_intent, whatever
        the model said."""
        declared = operations.OPERATIONS[name].ir_intent
        if not declared:
            pytest.skip(f"{name} declares no ir_intent")
        ir = _ir(intent="leaderboard", operation=name,
                 metric=MetricRef(key="total_connects", confidence=0.9),
                 sort=Sort(metric="total_connects"),
                 subjects=[])
        result = validate_ir(ir, db_session)
        assert result.ir.intent == declared

    def test_a_contradictory_intent_is_normalized_not_obeyed(self, db_session):
        """THE CASE THAT BROKE. `breakdown` diverted the whole IR away
        from the compiler in chat_service, discarding a metric that had
        been resolved correctly. It can no longer survive validation
        alongside an operation that says otherwise."""
        ir = _ir(intent="breakdown", operation="group_metric",
                 subject_level="team",
                 metric=MetricRef(key="total_connects", confidence=0.9))
        result = validate_ir(ir, db_session)
        # `group_metric` declares no ir_intent, so there is nothing to
        # derive and nothing is invented. What matters is that the stale
        # value can no longer DECIDE anything: every consumer that acts on
        # this reads resolved_operation().
        assert result.ir.resolved_operation() == "group_metric"
        assert result.missing == []

    def test_the_model_cannot_emit_the_contradiction_in_the_first_place(self):
        """The case above is only reachable from code that builds an IR
        directly. `breakdown` is not in the grammar's intent vocabulary,
        because no operation the model may choose declares it."""
        assert "breakdown" not in OFFERED_INTENTS

    def test_a_measured_operation_without_a_measure_is_still_refused(self, db_session):
        """The hole this closed: keyed on `intent`, a group_metric whose
        intent was not normalised skipped the metric requirement
        entirely and reached the compiler with nothing to measure."""
        ir = _ir(intent="breakdown", operation="group_metric", subject_level="team")
        assert "metric" in _missing(db_session, ir)

    def test_an_operation_with_no_declared_intent_keeps_its_own(self, db_session):
        """`population` has no ir_intent — there is nothing to derive, and
        inventing one here would be a second mapping. It is left alone,
        and resolved_operation() still reports the operation."""
        ir = _ir(intent="filtered_list", operation="population")
        result = validate_ir(ir, db_session)
        assert result.ir.resolved_operation() == "population"


# =====================================================================
# C + D. population and metric queries both validate
# =====================================================================
class TestTheTwoShapesThatWereUnreachable:

    def test_a_metric_free_population_passes(self, db_session):
        """operation=population, metric=null — a whole question."""
        ir = _ir(operation="population", filter_tree=FilterGroup(op="or", children=[
            Filter(field="team", operator="=", value=TEAM_A),
            Filter(field="team", operator="=", value=TEAM_B),
        ]))
        result = validate_ir(ir, db_session)
        assert result.missing == []
        assert result.ir.metric is None

    def test_a_group_metric_query_passes_with_its_measure(self, db_session):
        ir = _ir(operation="group_metric", subject_level="team",
                 metric=MetricRef(key="total_connects", confidence=0.9))
        assert _missing(db_session, ir) == []

    # ---- the distinction, redrawn ------------------------------------
    def test_a_metric_free_list_that_constrains_NOTHING_is_still_refused(self, db_session):
        """POLICY CHANGED IN PHASE 3 — recorded here rather than deleted,
        because the reasoning it replaces was deliberate and is worth
        keeping legible.

        This test used to assert that ANY metric-free `filtered_list` was
        refused, on the grounds that `operation="population"` marks the
        absence of a measure as INTENDED, and that a parse which merely
        FAILED to resolve one must still be caught: "the fix is that the
        model can now SELECT population, not that metric-free means
        population."

        What the measurement showed is that the label was not evidence of
        intent. It is sampled output, and it tracked WORDING:

            "advisors in Blue Area or DownTown"  -> population    (answered)
            "all advisors excluding Blue Area"   -> filtered_list (refused)

        Identical structure, opposite outcomes. And the guard did not
        actually protect against the case it named: a model that drops a
        measure the user asked for is equally free to emit `population`,
        in which case the same unranked list comes back with no refusal.
        The label moved the decision into the model rather than making it.

        So the rule is now structural and NARROWER than "metric-free means
        population" — it additionally requires that the query constrained
        SOMETHING. What survives from the original reasoning is exactly
        this case: no measure, no filter, no subject is an empty parse
        wearing a label, nothing makes the absence deliberate, and the
        clarifying question is still the right answer to it.

        The three operations whose answer IS a measure keep the
        requirement unconditionally — see the tests below.
        """
        assert "metric" in _missing(db_session, _ir(operation="filtered_list"))

    def test_a_constrained_metric_free_list_is_answered_as_a_population(self, db_session):
        """The other half of the same decision, so the pair reads as one
        rule. Full coverage of the boundary lives in
        test_metric_presence.TestAMetricFreeListIsAPopulation."""
        ir = _ir(operation="filtered_list", filter_tree=FilterGroup(op="or", children=[
            Filter(field="team", operator="=", value=TEAM_A),
            Filter(field="team", operator="=", value=TEAM_B),
        ]))
        result = validate_ir(ir, db_session)
        assert result.is_valid, result.missing
        assert result.ir.resolved_operation() == "population"

    @pytest.mark.parametrize("op", ["leaderboard", "comparison"])
    def test_a_metric_free_ranking_is_still_refused(self, db_session, op):
        assert "metric" in _missing(db_session, _ir(intent=op, operation=op))


# =====================================================================
# E. the defensive guard survives
# =====================================================================
class TestTheDefensiveGuardIsRetained:

    @pytest.mark.parametrize("intent", ["lookup", "trend"])
    def test_an_ir_built_outside_the_schema_is_still_rejected(self, db_session, intent):
        """The schema is only one way an IR is built — plan_to_ir,
        conversation patches and tests construct them directly — so
        narrowing the grammar must not be mistaken for making the
        validator's check redundant."""
        ir = _ir(intent=intent, operation="leaderboard",
                 metric=MetricRef(key="total_connects", confidence=0.9))
        assert any(m.startswith(f"unsupported_intent:{intent}")
                   for m in _missing(db_session, ir))

    def test_the_unsupported_registry_still_holds_both(self):
        assert {"lookup", "trend"} <= set(_UNSUPPORTED_INTENTS)
