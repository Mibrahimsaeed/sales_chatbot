"""ONE definition of "which measure is this query about".

THE DEFECT. A metric can legitimately live in four places, and which one
the parser uses depends on how the question was phrased:

    sort.metric   "top advisors BY revenue"
    metric.key    "revenue of Blue Area"
    metrics[0]    "connects and answered calls of all BCMs"
    a filter      "advisors with connects above 1000"   <- the common one

`ir_validator` and `query_compiler` both read only the first two. So a
query whose measure was stated plainly as a CONDITION looked like a query
with no measure at all: the validator reported `missing=["metric"]` and
the user was asked "which metric you'd like" for a sentence that named
one. Four of seven realistic explicit-metric queries were affected.

Reading them through `QueryIR.primary_metric()` is what stops the two
layers disagreeing. These tests pin the four shapes, the two rules that
must NOT change (a population stays metric-free, a genuinely metric-free
ranking still clarifies), and the non-mutation invariant that keeps a
filtered list from being rendered as a ranking.
"""

import pytest

from app.llm.ir_validator import validate_ir
from app.llm.query_ir import (
    Filter, FilterGroup, MetricRef, QueryIR, Sort, Subject,
)


def _ir(**overrides):
    base = dict(intent="filtered_list", operation="filtered_list",
                subject_level="advisor", sort=Sort(metric=None), limit=None)
    base.update(overrides)
    return QueryIR(**base)


# =====================================================================
# The accessor — all four shapes, in documented precedence order
# =====================================================================
class TestPrimaryMetricAccessor:
    def test_1_metric_field(self):
        ir = _ir(metric=MetricRef(key="mtd_cleared"))
        assert ir.primary_metric() == "mtd_cleared"

    def test_2_sort_only(self):
        ir = _ir(sort=Sort(metric="total_connects"))
        assert ir.primary_metric() == "total_connects"

    def test_3_metrics_list_only(self):
        ir = _ir(metrics=[MetricRef(key="answered_calls")])
        assert ir.primary_metric() == "answered_calls"

    def test_4_filter_only(self):
        """The shape that produced the bug."""
        ir = _ir(filters=[Filter(field="total_connects", operator=">", value=1000)])
        assert ir.primary_metric() == "total_connects"

    def test_sort_outranks_metric_outranks_metrics_outranks_filter(self):
        """Precedence is the documented semantics, not a preference: an
        explicit ranking beats a named primary, which beats the first of
        several, which beats a condition."""
        ir = _ir(sort=Sort(metric="mtd_cleared"),
                 metric=MetricRef(key="total_connects"),
                 metrics=[MetricRef(key="answered_calls")],
                 filters=[Filter(field="overdue", operator=">", value=1)])
        assert ir.primary_metric() == "mtd_cleared"

        ir = _ir(metric=MetricRef(key="total_connects"),
                 metrics=[MetricRef(key="answered_calls")],
                 filters=[Filter(field="overdue", operator=">", value=1)])
        assert ir.primary_metric() == "total_connects"

        ir = _ir(metrics=[MetricRef(key="answered_calls")],
                 filters=[Filter(field="overdue", operator=">", value=1)])
        assert ir.primary_metric() == "answered_calls"

    def test_metrics_first_entry_is_the_primary(self):
        """The prompt specifies "primary first" for `metrics`, so the
        accessor must honour that order rather than picking arbitrarily."""
        ir = _ir(metrics=[MetricRef(key="total_connects"),
                          MetricRef(key="answered_calls")])
        assert ir.primary_metric() == "total_connects"

    def test_an_entity_filter_is_not_a_metric(self):
        """"advisors in Blue Area" names no measure; `team` is a level."""
        ir = _ir(filters=[Filter(field="team", operator="=", value="Blue Area")])
        assert ir.primary_metric() is None

    def test_a_population_has_no_primary_metric(self):
        """By definition: it asks WHO, with nothing to rank by. Deriving
        one from its filter would put a value column on a question that
        asked for names."""
        ir = _ir(operation="population",
                 filters=[Filter(field="team_size", operator=">", value=5)])
        assert ir.primary_metric() is None

    def test_the_accessor_does_not_mutate(self):
        """Deriving the column to compute must not decide the answer's
        SHAPE. If this wrote `sort.metric`, response_planner would start
        rendering filtered lists as rankings."""
        ir = _ir(metrics=[MetricRef(key="total_connects")],
                 filters=[Filter(field="total_connects", operator=">", value=1000)])
        assert ir.primary_metric() == "total_connects"
        assert ir.metric is None
        assert ir.sort.metric is None
        assert ir.resolved_operation() == "filtered_list"


# =====================================================================
# A metric-free list is a POPULATION, whatever the model called it
# =====================================================================
class TestAMetricFreeListIsAPopulation:
    """`population` and `filtered_list` are the same answer shape with and
    without a measure, and the model was the only thing choosing between
    them for a metric-free list. It chose on WORDING:

        "advisors in Blue Area or DownTown"  -> population    (answered)
        "all advisors excluding Blue Area"   -> filtered_list (refused)

    Identical structure, opposite outcomes, because `filtered_list` is in
    `_MEASURED_OPERATIONS`. The user was asked "which metric would you
    like?" about a question that correctly named none, and the boolean
    filter machinery the shape exists for never ran.

    The validator now re-labels on the IR's own content, so the answer no
    longer depends on which conjunction the user reached for.
    """

    def _op(self, db_session, ir):
        return validate_ir(ir, db_session).ir.resolved_operation()

    def _valid(self, db_session, ir):
        return validate_ir(ir, db_session).is_valid

    # ---- the shapes that must be re-labelled ------------------------
    @pytest.mark.parametrize("label,ir_kwargs", [
        ("an exclusion", dict(filter_tree=FilterGroup(
            op="not", children=[Filter(field="team", value="Blue Area")]))),
        ("a disjunction", dict(filter_tree=FilterGroup(
            op="or", children=[Filter(field="team", value="Blue Area"),
                               Filter(field="team", value="DownTown")]))),
        ("a flat entity filter", dict(filters=[Filter(field="team", value="Blue Area")])),
        ("an attendance filter",
         dict(filters=[Filter(field="attendance_status", value="Late")])),
        ("a named subject", dict(subjects=[Subject(type="team", value="Blue Area")])),
    ])
    def test_a_constrained_metric_free_list_becomes_a_population(
            self, db_session, label, ir_kwargs):
        ir = _ir(operation="filtered_list", **ir_kwargs)
        assert self._op(db_session, ir) == "population", label

    @pytest.mark.parametrize("label,ir_kwargs", [
        ("an exclusion", dict(filter_tree=FilterGroup(
            op="not", children=[Filter(field="team", value="Blue Area")]))),
        ("a disjunction", dict(filter_tree=FilterGroup(
            op="or", children=[Filter(field="team", value="Blue Area"),
                               Filter(field="team", value="DownTown")]))),
        ("a flat entity filter", dict(filters=[Filter(field="team", value="Blue Area")])),
    ])
    def test_no_clarifying_question_is_asked_for_a_missing_measure(
            self, db_session, label, ir_kwargs):
        """The exit condition stated as an assertion: absence of a measure
        is not, by itself, a reason to ask the user anything."""
        result = validate_ir(_ir(operation="filtered_list", **ir_kwargs), db_session)
        assert "metric" not in result.missing, label
        assert result.is_valid, f"{label}: {result.missing}"

    # ---- the shapes that must NOT be ---------------------------------
    @pytest.mark.parametrize("operation", ["leaderboard", "group_metric"])
    def test_an_operation_whose_answer_IS_a_measure_still_needs_one(
            self, db_session, operation):
        """A ranking with nothing to rank by, and a group's "figure" with
        no figure named, are genuinely incomplete parses. Re-labelling
        those would turn a clarifying question into a silently different
        answer."""
        ir = _ir(operation=operation, intent="leaderboard",
                 subjects=[Subject(type="team", value="Blue Area")])
        assert "metric" in validate_ir(ir, db_session).missing

    def test_a_comparison_still_needs_a_measure(self, db_session):
        ir = _ir(operation="comparison", intent="comparison",
                 subjects=[Subject(type="team", value="Blue Area"),
                           Subject(type="team", value="DownTown")])
        assert "metric" in validate_ir(ir, db_session).missing

    def test_a_list_that_constrains_NOTHING_is_still_refused(self, db_session):
        """"show me the advisors" with no filter, no subject and no
        measure is an empty parse wearing a label. There is nothing that
        makes the absence deliberate, so the clarifying question is the
        right answer to it."""
        result = validate_ir(_ir(operation="filtered_list"), db_session)
        assert "metric" in result.missing

    # ---- and a measure anywhere keeps the list a list ----------------
    @pytest.mark.parametrize("label,ir_kwargs", [
        ("as a condition",
         dict(filters=[Filter(field="total_connects", operator=">", value=1000)])),
        ("in metrics[]", dict(metrics=[MetricRef(key="total_connects")],
                              filters=[Filter(field="team", value="Blue Area")])),
        ("as the sort key", dict(sort=Sort(metric="total_connects"),
                                 filters=[Filter(field="team", value="Blue Area")])),
        ("as the primary", dict(metric=MetricRef(key="total_connects"),
                                filters=[Filter(field="team", value="Blue Area")])),
    ])
    def test_a_measure_anywhere_keeps_it_a_filtered_list(
            self, db_session, label, ir_kwargs):
        """All four places a measure can live, so the re-label cannot
        strip a value column off a query that asked for one."""
        ir = _ir(operation="filtered_list", **ir_kwargs)
        assert self._op(db_session, ir) == "filtered_list", label

    # ---- the ordering the re-label depends on ------------------------
    def test_the_named_group_stays_the_SCOPE_not_the_answers_level(self, db_session):
        """`_SUBJECT_IS_A_SCOPE` lists `population`, so the re-label has to
        happen BEFORE the subject_level normalisation — otherwise
        "advisors in Blue Area" reports Blue Area itself instead of
        listing the people in it."""
        ir = _ir(operation="filtered_list", subject_level="advisor",
                 subjects=[Subject(type="team", value="Blue Area")])
        result = validate_ir(ir, db_session)
        assert result.ir.resolved_operation() == "population"
        assert result.ir.subject_level == "advisor"


class TestTheRelabelledPopulationActuallyAnswers:
    """The re-label is only worth anything if the rows come back — and
    come back COMPLETE. A population joins no fact table, which is the
    whole reason it must not be given a measure it was never asked for.
    """

    @pytest.fixture()
    def org(self, db_session):
        from app.database.models import Advisor, Calls
        from app.llm import entity_extractor

        for wid, (name, team) in enumerate(
                [("A", "Blue Area"), ("B", "Blue Area"), ("C", "DownTown"),
                 ("D", "DownTown"), ("E", "GCC")], start=1):
            db_session.add(Advisor(wid=wid, name=name, team=team,
                                   company="Graana", in_master_sheet=True))
        # Only two of the five have a `calls` row. A metric-bearing query
        # inner-joins that table and drops the other three.
        db_session.add(Calls(wid=1, connects_mtd=10))
        db_session.add(Calls(wid=3, connects_mtd=20))
        db_session.commit()
        entity_extractor._cache["loaded_at"] = 0
        return db_session

    def _rows(self, db, ir):
        from app.llm.query_compiler import compile_and_run

        return compile_and_run(db, validate_ir(ir, db).ir) or []

    def test_an_exclusion_returns_everyone_it_did_not_exclude(self, org):
        rows = self._rows(org, _ir(operation="filtered_list", filter_tree=FilterGroup(
            op="not", children=[Filter(field="team", value="Blue Area")])))
        assert sorted(r["name"] for r in rows) == ["C", "D", "E"]

    def test_a_disjunction_returns_the_union(self, org):
        rows = self._rows(org, _ir(operation="filtered_list", filter_tree=FilterGroup(
            op="or", children=[Filter(field="team", value="Blue Area"),
                               Filter(field="team", value="GCC")])))
        assert sorted(r["name"] for r in rows) == ["A", "B", "E"]

    def test_nobody_is_dropped_for_having_no_fact_row(self, org):
        """The defect the metric-free rule exists to prevent: three of
        these five have no `calls` row, and a measure attached to a
        question that never asked for one would silently omit them."""
        rows = self._rows(org, _ir(operation="filtered_list", filter_tree=FilterGroup(
            op="not", children=[Filter(field="team", value="Nowhere")])))
        assert len(rows) == 5

    def test_the_count_agrees_with_the_rows(self, org):
        from app.llm.query_compiler import count_ir

        ir = _ir(operation="filtered_list", filter_tree=FilterGroup(
            op="not", children=[Filter(field="team", value="Blue Area")]))
        validated = validate_ir(ir, org).ir
        assert count_ir(org, validated) == 3

    def test_it_renders_as_a_population_not_a_ranking(self, org):
        """A list with no measure must not print "no data" beside every
        name, which is what the leaderboard renderer does with it."""
        from app.llm.response_planner import plan_response

        ir = _ir(operation="filtered_list", filter_tree=FilterGroup(
            op="not", children=[Filter(field="team", value="Blue Area")]))
        validated = validate_ir(ir, org).ir
        rows = self._rows(org, ir)
        assert plan_response(validated, rows).mode == "population"
        assert all(r["value"] is None for r in rows)


# =====================================================================
# The validator — presence, and the two rules that must not change
# =====================================================================
class TestValidatorMetricPresence:
    def _missing(self, db_session, ir):
        return validate_ir(ir, db_session).missing

    def test_metric_in_metric_field_is_present(self, db_session):
        assert "metric" not in self._missing(
            db_session, _ir(intent="leaderboard", operation="leaderboard",
                            metric=MetricRef(key="mtd_cleared")))

    def test_metric_in_sort_only_is_present(self, db_session):
        assert "metric" not in self._missing(
            db_session, _ir(intent="leaderboard", operation="leaderboard",
                            sort=Sort(metric="mtd_cleared")))

    def test_metric_in_metrics_list_only_is_present(self, db_session):
        assert "metric" not in self._missing(
            db_session, _ir(metrics=[MetricRef(key="total_connects")]))

    def test_metric_only_in_a_filter_is_present(self, db_session):
        """THE regression. "advisors with connects above 1000"."""
        assert "metric" not in self._missing(
            db_session,
            _ir(filters=[Filter(field="total_connects", operator=">", value=1000)]))

    def test_two_different_metric_filters_is_ONE_query(self, db_session):
        """"achievement below 50% and answered calls % below 50%" is one
        query with two conditions, not an ambiguous one."""
        ir = _ir(metrics=[MetricRef(key="achievement_pct"),
                          MetricRef(key="answered_calls_rate")],
                 filters=[Filter(field="achievement_pct", operator="<", value=50),
                          Filter(field="answered_calls_rate", operator="<", value=50)])
        assert self._missing(db_session, ir) == []
        # Both conditions survive validation — neither is pruned.
        assert len(ir.filter_leaves()) == 2

    def test_several_requested_metrics_is_ONE_query(self, db_session):
        """"show connects and answered calls for all BCMs"."""
        ir = _ir(intent="leaderboard", operation="leaderboard", subject_level="bcm",
                 metric=MetricRef(key="total_connects"),
                 metrics=[MetricRef(key="total_connects"),
                          MetricRef(key="answered_calls")])
        assert self._missing(db_session, ir) == []
        assert ir.metric_keys() == ["total_connects", "answered_calls"]

    def test_a_population_with_a_metric_filter_needs_no_metric(self, db_session):
        """The prompt tells the model to emit `metric: null` here, so
        requiring one rejected the output the prompt asked for."""
        ir = _ir(operation="population", subject_level="bcm",
                 filters=[Filter(field="team_size", operator=">", value=5)])
        assert self._missing(db_session, ir) == []
        assert ir.primary_metric() is None

    def test_a_genuinely_metric_free_ranking_STILL_clarifies(self, db_session):
        """The guard must not be removed wholesale — a ranking with
        nothing to rank by is still an unanswerable question."""
        ir = _ir(intent="leaderboard", operation="leaderboard")
        assert "metric" in self._missing(db_session, ir)

    def test_an_entity_only_filter_still_clarifies(self, db_session):
        """"advisors in Blue Area" ranked by nothing — the level is not a
        measure, so this is genuinely missing one."""
        ir = _ir(intent="leaderboard", operation="leaderboard",
                 filters=[Filter(field="team", operator="=", value="Blue Area")])
        assert "metric" in self._missing(db_session, ir)

    def test_a_filter_metric_different_from_the_sort_metric(self, db_session):
        """Sorting by one measure while filtering on another is a valid
        query; the sort must remain the primary."""
        ir = _ir(intent="leaderboard", operation="leaderboard",
                 sort=Sort(metric="mtd_cleared"),
                 filters=[Filter(field="total_connects", operator=">", value=1000)])
        assert self._missing(db_session, ir) == []
        assert ir.primary_metric() == "mtd_cleared"

    def test_an_or_tree_metric_counts_as_present(self, db_session):
        ir = _ir(filter_tree=FilterGroup(op="or", children=[
            Filter(field="total_connects", operator=">", value=1000),
            Filter(field="answered_calls", operator=">", value=500)]))
        assert "metric" not in self._missing(db_session, ir)
        assert ir.primary_metric() == "total_connects"

    def test_a_not_tree_metric_counts_as_present(self, db_session):
        ir = _ir(filter_tree=FilterGroup(op="not", children=[
            Filter(field="overdue", operator=">", value=0)]))
        assert "metric" not in self._missing(db_session, ir)
        assert ir.primary_metric() == "overdue"

    def test_validation_does_not_turn_a_filtered_list_into_a_ranking(self, db_session):
        """The non-mutation invariant, asserted after validation rather
        than only on the accessor: response_planner decides the answer's
        shape from the operation, and every other consumer reads `sort`."""
        ir = _ir(filters=[Filter(field="total_connects", operator=">", value=1000)])
        validate_ir(ir, db_session)
        assert ir.metric is None
        assert ir.sort.metric is None
        assert ir.resolved_operation() == "filtered_list"


# =====================================================================
# The compiler must agree with the validator
# =====================================================================
class TestCompilerAgreesWithValidator:
    """Making validation pass is not enough. The compiler read the same
    two fields, so relaxing only the validator would have produced a
    silent empty result — worse than the clarification it replaced."""

    @pytest.mark.parametrize("ir", [
        _ir(filters=[Filter(field="total_connects", operator=">", value=1000)]),
        _ir(metrics=[MetricRef(key="achievement_pct"),
                     MetricRef(key="answered_calls_rate")],
            filters=[Filter(field="achievement_pct", operator="<", value=50),
                     Filter(field="answered_calls_rate", operator="<", value=50)]),
        _ir(metrics=[MetricRef(key="attendance_rate")], subject_level="team",
            filters=[Filter(field="attendance_rate", operator="<", value=80)]),
    ])
    def test_every_shape_the_validator_accepts_has_a_metric_to_compile(self, ir):
        from app.llm.query_compiler import _effective_metric

        assert ir.primary_metric() is not None
        assert _effective_metric(ir) is not None, (
            "the compiler must resolve a metric for every IR the validator "
            "accepts, or the answer is a silent empty result")

    def test_a_population_compiles_without_one_and_that_is_correct(self):
        from app.llm.query_compiler import _effective_metric

        ir = _ir(operation="population", subject_level="bcm",
                 filters=[Filter(field="team_size", operator=">", value=5)])
        assert _effective_metric(ir) is None


# =====================================================================
# The route gate must see metrics wherever they live
# =====================================================================
class TestRouteValidationSeesEveryMetric:
    """The mirror-image gap: the same narrow read let an INVENTED key
    reach the compiler unchallenged when it sat in `metrics[]` or a
    filter. The compiler then found no binding and returned nothing,
    which reads as "no data" rather than "that isn't a measure I have"."""

    def test_a_bad_key_in_metrics_is_rejected(self):
        from app.llm.routing import validate_route

        ir = _ir(metrics=[MetricRef(key="not_a_real_metric")])
        assert validate_route(ir) is not None

    def test_a_bad_key_in_a_filter_is_rejected(self):
        from app.llm.routing import validate_route

        ir = _ir(filters=[Filter(field="not_a_real_metric", operator=">", value=1)])
        assert validate_route(ir) is not None

    def test_an_entity_filter_is_not_mistaken_for_a_bad_metric(self):
        from app.llm.routing import validate_route

        ir = _ir(filters=[Filter(field="team", operator="=", value="Blue Area")],
                 metric=MetricRef(key="mtd_cleared"))
        assert validate_route(ir) is None

    def test_a_valid_filter_only_query_passes(self):
        from app.llm.routing import validate_route

        ir = _ir(filters=[Filter(field="total_connects", operator=">", value=1000)])
        assert validate_route(ir) is None
