"""Phase 2 — the metric foundation.

Three properties, each of which was previously spread across modules
that could disagree:

1. PERIOD. Which key answers a measure at a period is derived from the
   declarations, not from a hardcoded swap table. A measure with no data
   at a period says so instead of returning MTD numbers under a YTD
   label.
2. SORTING. Direction comes from the metric's own polarity when the user
   names none, so a lower-is-better metric cannot be ranked worst-first.
3. ONE VOCABULARY EACH. Comparators and period phrases have a single
   definition, and adding to either is a one-place change.
"""

import pytest

from app.database.models import PerformancePeriod
from app.llm import comparators, query_compiler, temporal_parser
from app.llm.metric_ontology import (
    DEFAULT_THRESHOLDS, METRICS, Thresholds, daily_target_rate, lower_is_better,
    metric_for_period, status_for, supported_periods, thresholds_for,
)
from app.llm.query_ir import plan_to_ir
from app.llm.query_planner import QueryPlan, build_query_plan


# ---------------------------------------------------------------------
# 1. Period resolution
# ---------------------------------------------------------------------

def test_supported_periods_is_derived_from_the_family():
    assert set(supported_periods("mtd_cleared")) == {
        PerformancePeriod.MTD, PerformancePeriod.YTD, PerformancePeriod.THREE_M
    }
    # every member of a family reports the same set
    assert set(supported_periods("ytd_cleared")) == set(supported_periods("mtd_cleared"))


def test_a_metric_with_no_family_supports_only_its_own_period():
    """RETIRED EXAMPLES. This used total_connects and bookings, on the
    premise that "SalesFunnel columns are MTD-only" — true until the ETL
    imported the "YTD CCMC" tab and gave them ytd_* siblings.

    The PROPERTY is unchanged and still the point: a measure with no
    period family answers only its own period, and claiming otherwise is
    what produced YTD-labelled MTD numbers. Re-pinned on measures that
    genuinely have one period."""
    assert supported_periods("attendance_rate") == (PerformancePeriod.MTD,)
    assert supported_periods("late_count") == (PerformancePeriod.MTD,)
    # `answered_calls` was a third example here until Phase 12 gave it a
    # real DAILY sibling from calls.answered_calls_daily. Replaced rather
    # than removed — the property needs an example that still holds, and
    # login_rate has no period siblings of any kind.
    assert supported_periods("login_rate") == (PerformancePeriod.MTD,)


@pytest.mark.parametrize("key,period,expected", [
    ("mtd_cleared", "YTD", "ytd_cleared"),
    ("mtd_cleared", "3M", "three_month_cleared"),
    ("ytd_cleared", "MTD", "mtd_cleared"),
    ("three_month_cleared", "YTD", "ytd_cleared"),
    ("mtd_cleared", "MTD", "mtd_cleared"),      # already correct
    ("mtd_cleared", None, "mtd_cleared"),       # no period asked for
])
def test_family_members_resolve_to_each_other(key, period, expected):
    assert metric_for_period(key, period) == expected


@pytest.mark.parametrize("key", ["attendance_rate", "late_count", "answered_calls",
                                 "achievement_pct", "mtd_target"])
def test_an_unsupported_period_returns_none_rather_than_mtd(key):
    """The core correctness change. None means "cannot answer"; the old
    behaviour silently kept the MTD binding.

    The example list changed when the YTD tabs were imported — connects,
    meetings, bookings, pipeline and overdue now have real YTD siblings.
    These five still do not."""
    assert metric_for_period(key, "YTD") is None
    assert metric_for_period(key, "3M") is None
    assert metric_for_period(key, "MTD") == key


@pytest.mark.parametrize("key,ytd_key", [
    ("total_connects", "ytd_connects"),
    ("total_meetings", "ytd_meetings"),
    ("bookings", "ytd_bookings"),
    ("pipeline_value", "ytd_pipeline_value"),
    ("overdue", "ytd_overdue"),
    ("client_registrations", "ytd_client_registrations"),
    ("conversion", "ytd_conversion"),
])
def test_a_measure_with_an_imported_ytd_source_resolves_to_it(key, ytd_key):
    """The other half of the same rule: where the data DOES exist, the
    period must resolve rather than refuse."""
    assert metric_for_period(key, "YTD") == ytd_key
    assert metric_for_period(key, "MTD") == key
    # 3M was never imported for these families.
    assert metric_for_period(key, "3M") is None


def test_the_compiler_is_the_authority():
    assert query_compiler.resolve_metric_for_period("mtd_cleared", "YTD") == "ytd_cleared"
    assert query_compiler.resolve_metric_for_period("total_connects", "YTD") == "ytd_connects"
    # Still no YTD source for attendance.
    assert query_compiler.resolve_metric_for_period("attendance_rate", "YTD") is None


def test_the_hardcoded_swap_table_is_gone():
    from app.llm import ir_patcher

    assert not hasattr(ir_patcher, "_PERIOD_METRIC_SWAP")
    assert not hasattr(ir_patcher, "_PERIOD_PHRASES")


def test_period_follow_up_swaps_a_family_metric():
    from app.llm.ir_patcher import _patch_period
    from app.llm.query_ir import MetricRef, QueryIR, Sort

    ir = QueryIR(intent="leaderboard", metric=MetricRef(key="mtd_cleared"),
                 sort=Sort(metric="mtd_cleared"))
    assert _patch_period(ir, "what about ytd") is True
    assert ir.metric.key == "ytd_cleared"
    assert ir.sort.metric == "ytd_cleared"
    assert ir.time_range.period == "YTD"


def test_period_follow_up_declines_when_the_measure_has_no_such_period():
    """Previously this set time_range.period=YTD and left the metric on
    MTD — the IR claimed a period the data did not have."""
    from app.llm.ir_patcher import _patch_period
    from app.llm.query_ir import MetricRef, QueryIR, Sort

    # attendance_rate, not total_connects: connects gained a real YTD
    # sibling when the "YTD CCMC" tab was imported, so it is no longer an
    # example of a measure that cannot answer YTD.
    ir = QueryIR(intent="leaderboard", metric=MetricRef(key="attendance_rate"),
                 sort=Sort(metric="attendance_rate"))
    assert _patch_period(ir, "what about ytd") is False
    assert ir.metric.key == "attendance_rate"
    assert ir.time_range.period == "MTD"        # untouched, not relabelled


def test_period_follow_up_uses_the_shared_vocabulary():
    """"last quarter" is an unsupported window; the patcher must not
    resolve it to 3M as its own substring table used to."""
    from app.llm.ir_patcher import _patch_period
    from app.llm.query_ir import MetricRef, QueryIR, Sort

    ir = QueryIR(intent="leaderboard", metric=MetricRef(key="mtd_cleared"),
                 sort=Sort(metric="mtd_cleared"))
    assert _patch_period(ir, "what about last quarter") is False
    assert ir.time_range.period == "MTD"


def test_the_compiler_resolves_metric_and_period_together(db_session):
    """The authority is USED, not merely available. `compile_and_run`
    and `count_ir` both go through `_effective_metric`, so an IR naming a
    period its measure cannot answer is declined rather than served with
    MTD numbers."""
    from app.database.models import Advisor, Attendance, Performance, SalesFunnel
    from app.llm.query_ir import MetricRef, QueryIR, Sort, TimeRange

    db_session.add(Advisor(wid=1, name="A", team="T", company="C"))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=5, mtd_followup_connect=0,
                               ytd_new_connect=60, ytd_followup_connect=0))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, target=100, cleared=50))
    db_session.add(Performance(wid=1, period=PerformancePeriod.YTD, target=900, cleared=600))
    db_session.commit()

    def run(metric, period):
        ir = QueryIR(intent="leaderboard", metric=MetricRef(key=metric),
                     sort=Sort(metric=metric), time_range=TimeRange(period=period))
        return query_compiler.compile_and_run(db_session, ir)

    # a family metric follows the period to its sibling's row
    assert run("mtd_cleared", "MTD")[0]["value"] == 50
    assert run("mtd_cleared", "YTD")[0]["value"] == 600
    # connects gained a real YTD source when the "YTD CCMC" tab was
    # imported, so it now follows the period to its own sibling COLUMN
    # (not a sibling row — see the SalesFunnel docstring).
    assert run("total_connects", "MTD")[0]["value"] == 5
    assert run("total_connects", "YTD")[0]["value"] == 60
    # A measure with no YTD source anywhere still declines rather than
    # serving MTD numbers under a YTD label — the property this test
    # exists for.
    db_session.add(Attendance(wid=1, biometric_mtd_ontime=9, biometric_mtd_late=1,
                              biometric_mtd_not_marked=0))
    db_session.commit()
    assert run("attendance_rate", "YTD") is None
    assert run("attendance_rate", "MTD")[0]["value"] == 90


def test_count_ir_uses_the_same_resolution(db_session):
    from app.database.models import Advisor, SalesFunnel
    from app.llm.query_ir import MetricRef, QueryIR, Sort, TimeRange

    db_session.add(Advisor(wid=1, name="A", team="T", company="C"))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=5))
    db_session.commit()
    # See the note in test_period_follow_up_declines...: connects now has
    # a YTD source, attendance does not.
    ir = QueryIR(intent="leaderboard", metric=MetricRef(key="attendance_rate"),
                 sort=Sort(metric="attendance_rate"), time_range=TimeRange(period="YTD"))
    assert query_compiler.count_ir(db_session, ir) is None


def test_plan_to_ir_keeps_the_period_consistent_with_the_metric():
    """Otherwise a default MTD time_range beside a YTD metric would read
    as a conflict and resolve the metric back to MTD."""
    for metric, expected in (("mtd_cleared", "MTD"), ("ytd_cleared", "YTD"),
                             ("three_month_cleared", "3M"), ("total_connects", "MTD")):
        ir = plan_to_ir(QueryPlan(action="leaderboard", metric=metric), {})
        assert ir.time_range.period == expected, metric


def test_the_superseded_direction_vocabulary_is_gone():
    """Dead configuration is worse than none — it reads as authoritative."""
    from app.llm import intent_catalog, query_planner

    assert not hasattr(intent_catalog, "DESCENDING_FLIPPED_BY")
    assert not hasattr(query_planner, "DESCENDING_DEFAULT_FLIPPED_BY")


def test_temporal_parser_is_the_single_period_vocabulary():
    assert temporal_parser.match_period("what about ytd") == "YTD"
    assert temporal_parser.match_period("this month") == "MTD"
    assert temporal_parser.match_period("last quarter") is None      # unsupported
    assert temporal_parser.match_period("top 5 by revenue") is None  # no period named


# ---------------------------------------------------------------------
# 2. Sorting polarity
# ---------------------------------------------------------------------

def test_lower_is_better_is_declared_where_it_applies():
    assert lower_is_better("overdue")
    # "overdue_amount" was removed — it bound the same column as
    # `overdue` under a label claiming a different unit.
    assert lower_is_better("ytd_overdue")
    assert lower_is_better("late_count")
    assert not lower_is_better("mtd_cleared")
    assert not lower_is_better("total_connects")
    assert not lower_is_better(None)


def test_default_direction_follows_polarity():
    assert query_compiler.default_direction("overdue") == "asc"
    assert query_compiler.default_direction("mtd_cleared") == "desc"


def test_a_ranking_with_no_direction_word_shows_the_good_end():
    """"top 5 by overdue" used to rank the WORST advisors first and call
    them the top."""
    ir = plan_to_ir(QueryPlan(action="leaderboard", metric="overdue"), {})
    assert ir.sort.direction == "asc"

    ir = plan_to_ir(QueryPlan(action="leaderboard", metric="mtd_cleared"), {})
    assert ir.sort.direction == "desc"


def test_an_explicit_direction_always_wins():
    ir = plan_to_ir(QueryPlan(action="leaderboard", metric="overdue", ascending=False), {})
    assert ir.sort.direction == "desc"
    ir = plan_to_ir(QueryPlan(action="leaderboard", metric="mtd_cleared", ascending=True), {})
    assert ir.sort.direction == "asc"


@pytest.mark.parametrize("query,metric,expected_asc", [
    ("highest overdue teams", "overdue", False),   # absolute: largest number
    ("lowest overdue teams", "overdue", True),     # absolute: smallest number
    ("worst overdue teams", "overdue", False),     # relative: most overdue
    ("worst revenue teams", "mtd_cleared", True),  # relative: least revenue
    ("top teams by revenue", "mtd_cleared", None), # unstated -> metric decides
])
def test_direction_words_are_absolute_or_relative(query, metric, expected_asc):
    plan = build_query_plan(query, {})
    assert plan.metric == metric
    assert plan.ascending is expected_asc


def test_normal_metrics_are_unaffected_by_the_polarity_change():
    """Everything that is not lower_is_better must sort exactly as before."""
    for key in METRICS:
        if lower_is_better(key):
            continue
        assert query_compiler.default_direction(key) == "desc", key


# ---------------------------------------------------------------------
# 3. Thresholds and target rates
# ---------------------------------------------------------------------

def test_banded_thresholds_use_the_documented_defaults():
    """RETIRED ASSERTIONS on 84.9 and 59.9.

    Both used to sit just under a band edge and be scored on the RAW
    value: 84.9 -> yellow, 59.9 -> red. FIX 2 rounds first, matching the
    dashboard's `pct = round(...)` before it bands, so both now land a
    band higher. The band EDGES themselves (85, 60) are unchanged."""
    assert DEFAULT_THRESHOLDS.green == 85.0
    assert DEFAULT_THRESHOLDS.yellow == 60.0
    assert status_for("achievement_pct", 85) == "green"
    assert status_for("achievement_pct", 60) == "yellow"
    # Rounded UP across the edge.
    assert status_for("achievement_pct", 84.9) == "green"
    assert status_for("achievement_pct", 59.9) == "yellow"
    # Still below once rounded.
    assert status_for("achievement_pct", 84.4) == "yellow"
    assert status_for("achievement_pct", 59.4) == "red"


def test_a_pass_fail_metric_is_green_only_at_zero():
    assert status_for("overdue", 0) == "green"
    assert status_for("overdue", 1) == "red"


def test_an_override_is_a_value_on_the_metric_not_a_branch():
    """The 1-Unit board's 45/30 bands are expressible today, even though
    that board does not exist yet."""
    bands = Thresholds(green=45.0, yellow=30.0)
    assert bands.status(45) == "green"
    assert bands.status(30) == "yellow"
    assert bands.status(29) == "red"


def test_a_metric_without_bands_has_no_status():
    assert thresholds_for("total_connects") is None
    assert status_for("total_connects", 40) is None


def test_unknown_value_has_no_status():
    assert status_for("achievement_pct", None) is None


def test_daily_target_rates_are_declared_on_the_metric():
    assert daily_target_rate("total_connects") == 10.0
    assert daily_target_rate("total_meetings") == 0.6
    assert daily_target_rate("mtd_cleared") is None


# ---------------------------------------------------------------------
# 4. Comparator vocabulary
# ---------------------------------------------------------------------

@pytest.mark.parametrize("phrase,operator", [
    ("above", ">"), ("over", ">"), ("more than", ">"), ("greater than", ">"),
    ("below", "<"), ("under", "<"), ("less than", "<"),
    ("at least", ">="), ("at most", "<="),
])
def test_every_required_comparator_parses_deterministically(phrase, operator):
    """"above" used to exist only as an embedding exemplar, so it dropped
    silently whenever the LLM provider was unavailable."""
    from app.llm.entity_extractor import _extract_thresholds

    found = _extract_thresholds(f"advisors with revenue {phrase} 80")
    assert found == [{"operator": operator, "value": 80.0}], phrase


@pytest.mark.parametrize("symbol,operator", [(">=", ">="), ("<=", "<="), (">", ">"), ("<", "<")])
def test_symbols_parse_and_do_not_shadow_each_other(symbol, operator):
    from app.llm.entity_extractor import _extract_thresholds

    found = _extract_thresholds(f"revenue {symbol} 80")
    assert {"operator": operator, "value": 80.0} in found


def test_one_declaration_carries_both_deterministic_and_semantic_forms():
    deterministic = {phrase for c in comparators.COMPARATORS for phrase in c.phrases}
    semantic = {exemplar for c in comparators.COMPARATORS for exemplar in c.exemplars}
    assert not (deterministic & semantic), "a phrase should not be in both lists"
    assert "above" in deterministic
    assert "north of" in semantic


def test_entity_extractor_reads_the_shared_vocabulary():
    from app.llm import entity_extractor

    assert entity_extractor._THRESHOLD_PATTERNS == comparators.threshold_patterns()
    assert entity_extractor._COMPARATOR_EXEMPLARS == comparators.semantic_exemplars()


# ---------------------------------------------------------------------
# 5. Labels
# ---------------------------------------------------------------------

@pytest.mark.parametrize("key", ["conversion", "bookings", "pipeline_value", "overdue"])
def test_mtd_only_metrics_say_so_in_their_label(key):
    """A period-neutral label on an MTD-only metric reads as a total for
    whatever period the user had in mind."""
    assert "MTD" in METRICS[key].label, key


def test_every_metric_declares_a_period():
    for key, metric in METRICS.items():
        assert isinstance(metric.period, PerformancePeriod), key
