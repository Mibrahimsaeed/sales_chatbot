"""P0: QueryIR can hold what complex questions actually say.

The audit found the bottleneck was the IR's SHAPE, not the model's
understanding. Five things had no representation:

    "BCMs in Blue Area OR Downtown"   filters is AND-combined by
                                      construction — the disjunction
                                      either became a conjunction
                                      matching nobody, or a branch was
                                      dropped
    "advisors excluding Blue Area"    `excluding` routed the query to the
                                      LLM and was then unrepresentable,
                                      so the word changed the PATH
                                      without changing the ANSWER
    "connects AND answered calls"     one `metric` field, so the second
                                      measure was lost before compilation
    "Zainab's connects and Awais's
     answered calls"                  refused outright by
                                      _distributes_metrics, because no
                                      structure could carry the pairing
    group_by / compare_to             carried by the schema and read by
                                      NOTHING

Everything below is additive. The first section pins that: an IR built
the old way compiles byte-identically, because the new fields default to
"absent" and every accessor collapses to the old reading.
"""

import pytest

from app.database.models import Advisor, Calls, Performance, PerformancePeriod
from app.llm.query_compiler import (
    UncompilableFilterTree, compile_and_run, count_ir,
)
from app.llm.query_ir import (
    Filter, FilterGroup, MetricRef, QueryIR, Sort, Subject,
)

# Three teams, three BCMs, sizes that make every filter's answer distinct.
_PEOPLE = [
    # wid, name,   team,    bcm,     connects, cleared
    (1, "A1", "Alpha", "BCM-A", 100, 10),
    (2, "A2", "Alpha", "BCM-A", 200, 20),
    (3, "B1", "Bravo", "BCM-B", 300, 30),
    (4, "C1", "Charlie", "BCM-C", 400, 40),
    (5, "C2", "Charlie", "BCM-C", 500, 50),
    (6, "C3", "Charlie", "BCM-C", 600, 60),
]


@pytest.fixture()
def org(db_session):
    for wid, name, team, bcm, connects, cleared in _PEOPLE:
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                               rm="UH", portfolio_lead="ZH", management_lead=bcm,
                               in_master_sheet=True))
        db_session.add(Calls(wid=wid, connects_mtd=connects, answered_calls_mtd=connects // 10))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   cleared=cleared, target=100))
    db_session.commit()
    return db_session


def _ir(**kw):
    base = dict(intent="leaderboard", subject_level="team",
                metric=MetricRef(key="total_connects"),
                sort=Sort(metric="total_connects", direction="desc"), limit=None)
    base.update(kw)
    return QueryIR(**base)


def _names(db, ir):
    return sorted(r["name"] for r in (compile_and_run(db, ir) or []))


# ===================================================== backward compat
class TestUnchanged:
    """An IR built the old way must behave exactly as it did."""

    def test_defaults_collapse_to_the_old_reading(self):
        old = QueryIR(intent="leaderboard", metric=MetricRef(key="mtd_cleared"))
        assert old.filter_tree is None
        assert old.metrics == []
        assert old.filter_leaves() == []
        assert old.metric_keys() == ["mtd_cleared"]
        assert old.grouping_level() == old.subject_level == "advisor"
        assert old.compare_period() is None

    def test_a_flat_filter_query_is_untouched(self, org):
        assert _names(org, _ir(filters=[Filter(field="team", operator="=", value="Alpha")])) == ["Alpha"]

    def test_an_unfiltered_ranking_is_untouched(self, org):
        assert _names(org, _ir()) == ["Alpha", "Bravo", "Charlie"]

    def test_subject_metric_defaults_to_none(self):
        assert Subject(type="advisor", value="X").metric is None


# ============================================================ OR / NOT
class TestFilterTree:
    def test_or_returns_the_union(self, org):
        ir = _ir(filter_tree=FilterGroup(op="or", children=[
            Filter(field="team", operator="=", value="Alpha"),
            Filter(field="team", operator="=", value="Charlie")]))
        assert _names(org, ir) == ["Alpha", "Charlie"]

    def test_not_excludes(self, org):
        ir = _ir(filter_tree=FilterGroup(op="not", children=[
            Filter(field="team", operator="=", value="Alpha")]))
        assert _names(org, ir) == ["Bravo", "Charlie"]

    def test_not_over_several_negates_their_conjunction(self, org):
        """`not(A, B)` is NOT(A AND B) — "excluding X and Y" — so a row
        failing either test survives."""
        ir = _ir(filter_tree=FilterGroup(op="not", children=[
            Filter(field="team", operator="=", value="Alpha"),
            Filter(field="company", operator="=", value="Nobody")]))
        assert _names(org, ir) == ["Alpha", "Bravo", "Charlie"]

    def test_nested_groups_compose(self, org):
        ir = _ir(filter_tree=FilterGroup(op="and", children=[
            FilterGroup(op="or", children=[
                Filter(field="team", operator="=", value="Alpha"),
                Filter(field="team", operator="=", value="Charlie")]),
            FilterGroup(op="not", children=[
                Filter(field="team", operator="=", value="Charlie")])]))
        assert _names(org, ir) == ["Alpha"]

    def test_tree_is_anded_with_the_flat_list(self, org):
        """Both are conjuncts, so a query can carry ordinary filters and
        one disjunction at the same time."""
        ir = _ir(filters=[Filter(field="company", operator="=", value="Graana")],
                 filter_tree=FilterGroup(op="or", children=[
                     Filter(field="team", operator="=", value="Alpha"),
                     Filter(field="team", operator="=", value="Bravo")]))
        assert _names(org, ir) == ["Alpha", "Bravo"]

        ir_miss = ir.model_copy(update={
            "filters": [Filter(field="company", operator="=", value="Nobody")]})
        assert _names(org, ir_miss) == []

    def test_count_agrees_with_the_rows(self, org):
        """count_ir gates pagination, so a tree it cannot see would page
        a different set than it counted."""
        ir = _ir(filter_tree=FilterGroup(op="or", children=[
            Filter(field="team", operator="=", value="Alpha"),
            Filter(field="team", operator="=", value="Bravo")]))
        assert count_ir(org, ir) == len(_names(org, ir)) == 2

    def test_a_metric_leaf_works_inside_a_tree(self, org):
        """Charlie's three advisors have 1,500 connects; the others less."""
        ir = _ir(filter_tree=FilterGroup(op="and", children=[
            Filter(field="total_connects", operator=">", value=1000)]))
        assert _names(org, ir) == ["Charlie"]

    def test_an_empty_group_is_a_no_op(self, org):
        assert _names(org, _ir(filter_tree=FilterGroup(op="and", children=[]))) == \
            ["Alpha", "Bravo", "Charlie"]


# ======================================================= honest refusal
class TestRefusesRatherThanGuesses:
    def test_or_mixing_a_row_test_and_an_aggregate_is_refused(self, org):
        """`WHERE team=... OR HAVING count(*)>5` is not a clause SQL has.
        Pushing either side into the other returns a plausible wrong set."""
        ir = _ir(subject_level="bcm", filter_tree=FilterGroup(op="or", children=[
            Filter(field="team", operator="=", value="Alpha"),
            Filter(field="team_size", operator=">", value=2)]))
        with pytest.raises(UncompilableFilterTree):
            compile_and_run(org, ir)

    def test_the_same_mix_is_fine_under_and(self, org):
        """A conjunction splits between WHERE and HAVING, so it compiles."""
        ir = _ir(subject_level="bcm", filter_tree=FilterGroup(op="and", children=[
            Filter(field="team", operator="=", value="Charlie"),
            Filter(field="team_size", operator=">", value=2)]))
        assert _names(org, ir) == ["BCM-C"]

    def test_an_uncompilable_leaf_under_or_is_refused(self, org):
        """Dropping it would WIDEN the result — silently answering a
        broader question than was asked."""
        ir = _ir(filter_tree=FilterGroup(op="or", children=[
            Filter(field="team", operator="=", value="Alpha"),
            Filter(field="not_a_field", operator="=", value="x")]))
        with pytest.raises(UncompilableFilterTree):
            compile_and_run(org, ir)

    def test_the_same_leaf_under_and_is_dropped_as_before(self, org):
        """Under AND it only narrows less, which is what the flat list
        has always done with an ungroundable field."""
        ir = _ir(filter_tree=FilterGroup(op="and", children=[
            Filter(field="team", operator="=", value="Alpha"),
            Filter(field="not_a_field", operator="=", value="x")]))
        assert _names(org, ir) == ["Alpha"]


# ====================================================== multiple metrics
class TestMultipleMetrics:
    def test_metric_keys_is_primary_first_and_deduped(self):
        ir = _ir(metric=MetricRef(key="total_connects"),
                 metrics=[MetricRef(key="total_connects"), MetricRef(key="answered_calls")])
        assert ir.metric_keys() == ["total_connects", "answered_calls"]

    def test_the_primary_still_decides_the_ranking(self, org):
        """A second measure must not change the order."""
        one = _ir()
        two = _ir(metrics=[MetricRef(key="total_connects"), MetricRef(key="mtd_cleared")])
        assert [r["value"] for r in compile_and_run(org, one)] == \
               [r["value"] for r in compile_and_run(org, two)]

    def test_every_named_measure_gets_a_column(self, org):
        from app.services import chat_service

        ir = _ir(metrics=[MetricRef(key="total_connects"), MetricRef(key="mtd_cleared")])
        rows = compile_and_run(org, ir)
        keys = chat_service._attach_bundle_columns(org, ir, rows)

        assert keys[0] == "total_connects"
        assert "mtd_cleared" in keys
        assert rows[0]["columns"]["mtd_cleared"]["value"] is not None

    def test_one_measure_still_attaches_nothing(self, org):
        """An unbundled single-measure ranking is unchanged."""
        from app.services import chat_service

        ir = _ir(metric=MetricRef(key="mtd_cleared"), sort=Sort(metric="mtd_cleared"))
        rows = compile_and_run(org, ir)
        assert chat_service._attach_bundle_columns(org, ir, rows) == []


# ================================================ per-subject binding
class TestPerSubjectMetric:
    def test_a_subject_can_carry_its_own_measure(self):
        ir = _ir(intent="comparison", subject_level="advisor",
                 subjects=[Subject(type="advisor", value="A1", metric=MetricRef(key="total_connects")),
                           Subject(type="advisor", value="B1", metric=MetricRef(key="mtd_cleared"))])
        assert [s.metric.key for s in ir.subjects] == ["total_connects", "mtd_cleared"]

    def test_it_survives_a_round_trip(self):
        ir = _ir(subjects=[Subject(type="advisor", value="A1", metric=MetricRef(key="overdue"))])
        assert QueryIR.model_validate(ir.model_dump()).subjects[0].metric.key == "overdue"


# ============================================== group_by and compare_to
class TestFormerlyDeadFields:
    def test_group_by_changes_the_reported_level(self, org):
        """Six advisors, three teams. Grouped by team the answer has three
        rows — before this the field was read by nothing and the answer
        had six."""
        rows = compile_and_run(org, _ir(subject_level="advisor", group_by="team"))
        assert sorted(r["name"] for r in rows) == ["Alpha", "Bravo", "Charlie"]

    def test_group_by_none_is_the_subject_level(self, org):
        rows = compile_and_run(org, _ir(subject_level="advisor"))
        assert len(rows) == len(_PEOPLE)

    def test_compare_period_reads_the_field(self):
        ir = _ir()
        assert ir.compare_period() is None
        ir.time_range.compare_to = "YTD"
        assert ir.compare_period() == "YTD"

    def test_a_compare_to_equal_to_the_period_is_not_a_comparison(self):
        ir = _ir()
        ir.time_range.compare_to = ir.time_range.period
        assert ir.compare_period() is None


# ==================================================== schema + rendering
class TestSchemaAndRendering:
    def test_the_json_schema_carries_the_new_fields(self):
        from app.llm.llm_client import QUERY_IR_JSON_SCHEMA as schema

        props = schema["properties"]
        assert "metrics" in props and "filter_tree" in props
        assert "metric" in props["subjects"]["items"]["properties"]
        for field in ("metrics", "filter_tree"):
            assert field in schema["required"], field

    def test_the_tree_schema_is_bounded_not_recursive(self):
        """Strict decoding builds a grammar from this, and a $ref cycle
        has no grammar — so the depth is finite by construction."""
        from app.llm.llm_client import QUERY_IR_JSON_SCHEMA as schema
        import json

        node = schema["properties"]["filter_tree"]
        assert "$ref" not in json.dumps(node)

        # Walk the nesting: each level's `children` may be a leaf or the
        # next group down, and the chain has to terminate.
        depth = 0
        group = next(o for o in node["anyOf"] if o.get("type") == "object")
        while True:
            depth += 1
            options = group["properties"]["children"]["items"]["anyOf"]
            deeper = [o for o in options if "children" in o.get("properties", {})]
            if not deeper:
                break
            group = deeper[0]
        assert depth == 3, depth

    def test_the_prompt_documents_the_new_shape(self):
        from app.llm.prompt_builder import _ir_schema

        text = _ir_schema()
        for token in ("metrics", "filter_tree", "MULTIPLE MEASURES", "BOOLEAN FILTERS"):
            assert token in text, token

    def test_a_disjunction_is_not_rendered_as_a_comma_list(self):
        """Commas read as AND, so listing OR branches beside the conjuncts
        would describe a narrower query than the one that ran."""
        from app.llm.response_formatter import _filters_summary

        summary = _filters_summary(_ir(filter_tree=FilterGroup(op="or", children=[
            Filter(field="team", operator="=", value="Alpha"),
            Filter(field="team", operator="=", value="Bravo")])))
        assert "OR" in summary and "Alpha" in summary and "Bravo" in summary

    def test_an_exclusion_says_so(self):
        from app.llm.response_formatter import _filters_summary

        summary = _filters_summary(_ir(filter_tree=FilterGroup(op="not", children=[
            Filter(field="team", operator="=", value="Alpha")])))
        assert "NOT" in summary


# =========================================================== validation
class TestValidation:
    def test_a_bad_tree_leaf_is_reported_not_pruned(self, org):
        """Pruning a child of an `or` widens the result and of a `not`
        inverts it, so the validator records instead of dropping."""
        from app.llm.ir_validator import validate_ir

        ir = _ir(filter_tree=FilterGroup(op="or", children=[
            Filter(field="team", operator="=", value="Alpha"),
            Filter(field="not_a_field", operator="=", value="x")]))
        validate_ir(ir, org)

        assert any(m.startswith("filter:not_a_field") for m in ir.missing)
        assert ir.filter_tree is not None
        assert len(ir.filter_tree.children) == 2

    def test_a_valid_tree_passes_clean(self, org):
        from app.llm.ir_validator import validate_ir

        ir = _ir(filter_tree=FilterGroup(op="or", children=[
            Filter(field="team", operator="=", value="Alpha"),
            Filter(field="total_connects", operator=">", value=1)]))
        validate_ir(ir, org)
        assert not [m for m in ir.missing if m.startswith("filter")]

    def test_condition_columns_see_metrics_inside_a_tree(self, org):
        """A measure constrained inside an OR is still a measure the user
        named, so its value belongs in the table."""
        from app.services import chat_service

        ir = _ir(filter_tree=FilterGroup(op="or", children=[
            Filter(field="mtd_cleared", operator=">", value=0)]))
        assert "mtd_cleared" in chat_service._condition_metrics(ir)
