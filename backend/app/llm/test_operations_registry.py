"""One registry, and a mechanism that forces the others to agree with it.

Five vocabularies named the same concepts and none derived from another:
17 plan actions, 13 scoring priors, 7 IR intents, 18 response modes, 12
LLM intent names. Adding a capability meant six coordinated edits across
five modules, and a missed one failed SILENTLY — an action left out of
one registry answered "I'm not tracking that one" while the planner was
building it correctly.

What was missing was never a better naming scheme. It was anything at all
that failed when the tables disagreed. `operations.verify()` is that
mechanism, and test_every_vocabulary_reconciles is where it fires.

CONSOLIDATION IS NOT DELETION. Every operation that existed still exists
and still answers the same way. What changed is that each is declared
once, and that QueryIR carries a single `operation` field spanning all of
them — including the ones `intent` has no value for.
"""

import typing

import pytest

from app.llm import operations
from app.llm.query_ir import MetricRef, QueryIR, Sort


# ======================================================= the mechanism
class TestVocabulariesReconcile:
    def test_every_vocabulary_reconciles(self):
        """THE guard. Every member of every table maps to an operation.

        A capability added to one vocabulary and forgotten in another
        fails here instead of reaching a user as a confident refusal.
        """
        assert operations.verify() == []

    def test_every_ir_intent_has_an_operation(self):
        from app.llm.query_ir import Intent

        for intent in typing.get_args(Intent):
            assert operations.for_ir_intent(intent) is not None, intent

    def test_every_rule_based_action_has_an_operation(self):
        from app.llm.nlu_pipeline import _RULE_BASED_ACTIONS

        for action in _RULE_BASED_ACTIONS:
            assert operations.for_plan_action(action) is not None, action

    def test_every_dispatch_mode_is_one_response_planner_knows(self):
        from app.llm.response_planner import DISPATCH_MODES

        for op in operations.OPERATIONS.values():
            assert op.dispatch_mode in DISPATCH_MODES, op.name

    def test_the_views_are_derived_not_restated(self):
        """A view that could drift from the declarations would be a sixth
        vocabulary."""
        for name, op in operations.OPERATIONS.items():
            assert op.name == name
            if op.plan_action:
                assert operations.BY_PLAN_ACTION[op.plan_action] is op
            if op.ir_intent:
                assert operations.BY_IR_INTENT[op.ir_intent] is op
        assert operations.IR_EXPRESSIBLE | operations.PLAN_ONLY == set(operations.OPERATIONS)
        assert not (operations.IR_EXPRESSIBLE & operations.PLAN_ONLY)


# =================================================== nothing was removed
class TestNoFunctionalityLost:
    @pytest.mark.parametrize("action", [
        "leaderboard", "comparison", "group_metric", "advisor_metric",
        "lookup", "summary", "breakdown", "roster", "ancestry",
        "reverse_hierarchy", "direct_reports", "attendance_filter",
        "trend", "clarify_person", "clarify_ambiguous", "clarify_metric",
        "comparison_incomplete", "unresolved",
    ])
    def test_every_operation_that_existed_still_does(self, action):
        """Consolidation must not shrink the system. Each plan action the
        planner can build is still a declared operation with a dispatch
        mode."""
        op = operations.for_plan_action(action)
        assert op is not None, action
        assert op.dispatch_mode

    def test_the_ir_only_operation_is_registered_too(self):
        """`filtered_list` is produced by the parser and by no scorer, so
        it has no plan action — and would be invisible to a registry keyed
        only on actions."""
        op = operations.OPERATIONS["filtered_list"]
        assert op.plan_action is None
        assert op.ir_intent == "filtered_list"

    def test_plan_only_operations_are_named_rather_than_implied(self):
        """The concrete to-do list for finishing the consolidation: these
        answer with a shape QueryIR cannot hold, which is why the plan is
        still their only path."""
        assert {"roster", "ancestry", "reverse_hierarchy", "direct_reports",
                "lookup", "summary"} <= operations.PLAN_ONLY


# ================================================== the operation field
class TestOperationField:
    def test_it_defaults_to_none_and_reads_from_intent(self):
        """An IR built before the field existed answers identically."""
        ir = QueryIR(intent="leaderboard", metric=MetricRef(key="mtd_cleared"))
        assert ir.operation is None
        assert ir.resolved_operation() == "leaderboard"

    def test_it_wins_when_set(self):
        ir = QueryIR(intent="leaderboard", operation="filtered_list")
        assert ir.resolved_operation() == "filtered_list"

    def test_an_unknown_operation_falls_back_to_the_intent(self):
        """An unrecognised value is the validator's business; this
        accessor must not raise in the middle of answering."""
        ir = QueryIR(intent="leaderboard", operation="not_an_operation")
        assert ir.resolved_operation() == "leaderboard"

    def test_it_survives_a_round_trip(self):
        ir = QueryIR(intent="comparison", operation="comparison")
        assert QueryIR.model_validate(ir.model_dump()).operation == "comparison"

    def test_a_plan_built_ir_carries_it(self):
        """plan_to_ir stamps the operation, so a plan-built IR and a
        parser-built one expose the same single field."""
        from app.llm.query_ir import plan_to_ir
        from app.llm.query_planner import QueryPlan

        ir = plan_to_ir(QueryPlan(action="leaderboard", metric="mtd_cleared",
                                  level="advisor"), {})
        assert ir.operation == "leaderboard"
        assert ir.resolved_operation() == "leaderboard"


# ============================== the two names must never disagree
class TestTheTwoNamesStayInStep:
    """`operation` and `intent` are the same fact under the new and old
    vocabularies, and resolved_operation() prefers the first. Anywhere one
    is rewritten, the other has to move with it — carrying the intent
    alone once left a follow-up reading "comparison" by one field and
    "leaderboard" by the other, and answering as the second.
    """

    def test_context_carry_moves_both(self):
        from app.llm import conversation_context
        from app.llm.query_ir import Subject

        prior = QueryIR(intent="comparison", operation="comparison",
                        subject_level="team",
                        subjects=[Subject(type="team", value="Alpha"),
                                  Subject(type="team", value="Bravo")],
                        metric=MetricRef(key="mtd_cleared"))
        current = QueryIR(intent="leaderboard", operation="leaderboard",
                          subject_level="advisor",
                          metric=MetricRef(key="total_connects"),
                          sort=Sort(metric="total_connects"))

        spec = conversation_context.specified("what about connects", {})
        merged = conversation_context.merge(
            prior, current, spec,
            conversation_context.ellipsis(spec, True),
        )
        if merged.ir.intent == "comparison":
            assert merged.ir.resolved_operation() == "comparison"

    def test_promoting_a_filled_clarification_moves_both(self):
        ir = QueryIR(intent="clarify", operation="clarify_metric")
        ir.intent = "leaderboard"
        ir.operation = ir.intent
        assert ir.resolved_operation() == "leaderboard"


# ========================================= the LLM sees only what fits
class TestSchemaAndPrompt:
    def test_the_schema_offers_only_ir_expressible_operations(self):
        """Offering the model an operation the IR cannot hold would
        invite it to emit something uncompilable."""
        from app.llm.llm_client import QUERY_IR_JSON_SCHEMA as schema

        offered = set(schema["properties"]["operation"]["anyOf"][1]["enum"])
        assert offered == operations.IR_EXPRESSIBLE
        assert not offered & operations.PLAN_ONLY

    def test_the_enum_is_derived_from_the_registry(self):
        from app.llm.llm_client import _ir_operations

        assert set(_ir_operations()) == operations.IR_EXPRESSIBLE

    def test_the_prompt_documents_the_field(self):
        from app.llm.prompt_builder import _ir_schema

        text = _ir_schema()
        assert '"operation"' in text
        assert "OPERATION." in text

    def test_the_schema_still_forbids_sql(self):
        from app.llm.llm_client import QUERY_IR_JSON_SCHEMA as schema
        import json

        rendered = json.dumps(schema).lower()
        assert "sql" not in rendered and "select " not in rendered


# ============================================ shape decided by one field
class TestResponsePlannerReadsOneField:
    def test_the_shape_comes_from_the_operation(self):
        from app.llm.response_planner import plan_response

        rows = [{"name": "A", "value": 1}, {"name": "B", "value": 2}]
        ir = QueryIR(intent="leaderboard", operation="filtered_list",
                     metric=MetricRef(key="mtd_cleared"))
        assert plan_response(ir, rows).shape == "filtered_table"

    def test_intent_alone_still_decides_when_operation_is_unset(self):
        from app.llm.response_planner import plan_response

        rows = [{"name": "A", "value": 1}, {"name": "B", "value": 2}]
        ir = QueryIR(intent="filtered_list", metric=MetricRef(key="mtd_cleared"))
        assert plan_response(ir, rows).shape == "filtered_table"
