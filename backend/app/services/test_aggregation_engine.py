"""Phase 4 — ONE aggregation pipeline.

The characterisation file next door records what the five old paths each
produced. This file asserts the property that replaces them: for a given
entity and metric there is now exactly ONE number, and it is the right
one.

The fixture is deliberately UNBALANCED. Average-of-percentages and
ratio-of-sums agree whenever advisors carry equal weight, so a balanced
fixture cannot tell which rule is in force and would pass against the
defect. Every advisor here has a different denominator.
"""

import ast
import pathlib

import pytest

from app.database.models import (
    Advisor, Attendance, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import aggregation, hierarchy
from app.llm.metric_ontology import METRICS, Rollup
from app.llm.query_compiler import compile_and_run
from app.llm.query_ir import MetricRef, QueryIR, Sort, Subject
from app.services import (
    comparison_service, company_service, hierarchy_service, team_service,
)


@pytest.fixture()
def org(db_session):
    """Two advisors under one chain, on every level at once.

    Advisor A: cleared 900 / target 1000   -> 90%   attendance 9/10  -> 90%
    Advisor B: cleared  10 / target  100   -> 10%   attendance 1/2   -> 50%

    achievement   average 50.0   ratio of sums 910/1100 = 82.73
    attendance    average 70.0   ratio of sums  10/12   = 83.33

    Both metrics separate the two rules, and they separate them to
    DIFFERENT numbers — so a path cannot accidentally look correct on one
    by using the wrong rule on the other.
    """
    for wid, name in ((1, "Adv A"), (2, "Adv B")):
        db_session.add(Advisor(wid=wid, name=name, team="Blue Area", company="Graana",
                               rm="UH Ali", portfolio_lead="ZH Sara",
                               management_lead="BCM Omar", office="Center One"))

    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, target=1000, cleared=900, pct=90))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, target=100, cleared=10, pct=10))
    db_session.add(Performance(wid=1, period=PerformancePeriod.YTD, target=2000, cleared=1800))
    db_session.add(Performance(wid=2, period=PerformancePeriod.YTD, target=200, cleared=20))

    db_session.add(SalesFunnel(wid=1, mtd_new_connect=30, mtd_followup_connect=0, mtd_conversion=3))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=10, mtd_followup_connect=0, mtd_conversion=1))

    db_session.add(Pipeline(wid=1, pipeline=5000, overdue=2))
    db_session.add(Pipeline(wid=2, pipeline=1000, overdue=0))

    db_session.add(Attendance(wid=1, biometric_mtd_ontime=9, biometric_mtd_late=1,
                              biometric_mtd_not_marked=0))
    db_session.add(Attendance(wid=2, biometric_mtd_ontime=1, biometric_mtd_late=1,
                              biometric_mtd_not_marked=0))
    db_session.commit()
    return db_session


# Every level that contains exactly these two advisors.
CHAIN_GROUPS = [
    ("team", "Blue Area"),
    ("unit_head", "UH Ali"),
    ("zonal_head", "ZH Sara"),
    ("bcm", "BCM Omar"),
]
ALL_GROUPS = CHAIN_GROUPS + [("company", "Graana"), ("office", "Center One")]

RATIO_ACHIEVEMENT = 910 / 1100 * 100   # 82.73
RATIO_ATTENDANCE = 10 / 12 * 100       # 83.33
AVERAGED_ACHIEVEMENT = 50.0            # what the old paths returned
AVERAGED_ATTENDANCE = 70.0


def _leaderboard(db, metric, level):
    ir = QueryIR(intent="leaderboard", subject_level=level,
                 metric=MetricRef(key=metric), sort=Sort(metric=metric))
    rows = compile_and_run(db, ir)
    return rows[0]["value"] if rows else None


# ---------------------------------------------------------------------
# Percentages: the ratio of sums, at every level
# ---------------------------------------------------------------------

@pytest.mark.parametrize("level,value", ALL_GROUPS)
def test_achievement_is_the_ratio_of_sums_at_every_level(org, level, value):
    got = aggregation.metric_value(org, level, value, "achievement_pct")
    assert got == pytest.approx(RATIO_ACHIEVEMENT, abs=0.1)
    # The specific wrong answer this replaces, named so a regression is
    # recognisable rather than just "not 82.7".
    assert got != pytest.approx(AVERAGED_ACHIEVEMENT, abs=0.1)


@pytest.mark.parametrize("level,value", ALL_GROUPS)
def test_attendance_rate_is_the_ratio_of_sums_at_every_level(org, level, value):
    got = aggregation.metric_value(org, level, value, "attendance_rate")
    assert got == pytest.approx(RATIO_ATTENDANCE, abs=0.1)
    assert got != pytest.approx(AVERAGED_ATTENDANCE, abs=0.1)


def test_a_percentage_of_an_empty_denominator_is_no_data_not_zero(org):
    """NULLIF, not division by zero — and None, not 0.0. A group with no
    targets has NO achievement; reporting 0% would rank it as the worst
    performer rather than as unreported."""
    org.add(Advisor(wid=9, name="Adv C", team="Empty Team", company="Graana"))
    org.commit()
    assert aggregation.metric_value(org, "team", "Empty Team", "achievement_pct") is None


# ---------------------------------------------------------------------
# Additive metrics and counts
# ---------------------------------------------------------------------

@pytest.mark.parametrize("metric,expected", [
    ("total_connects", 40),    # 30 + 10
    ("mtd_cleared", 910),
    ("mtd_target", 1100),
    ("ytd_cleared", 1820),
    ("pipeline_value", 6000),
    ("overdue", 2),
    ("conversion", 4),         # 3 + 1 — a count, so it sums
])
@pytest.mark.parametrize("level,value", CHAIN_GROUPS)
def test_additive_metrics_sum(org, metric, expected, level, value):
    assert aggregation.metric_value(org, level, value, metric) == pytest.approx(expected)


def test_a_count_is_never_averaged(org):
    """conversion counts conversions. Averaging reported 2, which is not
    a number of anything that happened."""
    assert aggregation.rollup_for("conversion") is Rollup.SUM
    assert aggregation.metric_value(org, "team", "Blue Area", "conversion") == 4


# ---------------------------------------------------------------------
# The leaf is never rolled up
# ---------------------------------------------------------------------

def test_an_advisors_own_value_is_not_aggregated(org):
    assert aggregation.metric_value(org, "advisor", "Adv A", "achievement_pct") == pytest.approx(90.0)
    assert aggregation.metric_value(org, "advisor", "Adv B", "achievement_pct") == pytest.approx(10.0)
    assert aggregation.metric_value(org, "advisor", "Adv A", "total_connects") == 30


def test_the_leaf_is_the_only_level_that_skips_rollup(org):
    binding = aggregation.binding_for("achievement_pct", "advisor")
    assert not aggregation.needs_rollup("advisor", binding)
    for level, _ in CHAIN_GROUPS:
        assert aggregation.needs_rollup(level, binding), level


# ---------------------------------------------------------------------
# ONE pipeline: every consumer agrees
# ---------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["achievement_pct", "total_connects", "mtd_cleared", "conversion"])
def test_comparison_leaderboard_and_engine_agree(org, metric):
    """The defect Phase 4 exists to remove: these three used to be three
    independent implementations, and for percentages they disagreed."""
    engine = aggregation.metric_value(org, "team", "Blue Area", metric)
    comparison = comparison_service._metric_value(org, "team", "Blue Area", metric)
    leaderboard = _leaderboard(org, metric, "team")

    assert comparison == pytest.approx(engine)
    assert leaderboard == pytest.approx(engine)


@pytest.mark.parametrize("level,value", CHAIN_GROUPS)
def test_every_level_reports_the_same_group_identically(org, level, value):
    """team / unit_head / zonal_head / bcm all contain exactly these two
    advisors here, so every metric must match across all four. A per-level
    implementation is what made them differ."""
    for metric in ("achievement_pct", "attendance_rate", "total_connects", "mtd_cleared"):
        assert aggregation.metric_value(org, level, value, metric) == pytest.approx(
            aggregation.metric_value(org, "team", "Blue Area", metric)
        ), metric


def test_the_summary_services_agree_with_the_engine(org):
    team = team_service.get_team_summary(org, "Blue Area")
    company = company_service.get_company_summary(org, "Graana")
    level = hierarchy_service.get_level_summary(org, "unit_head", "UH Ali")
    engine = aggregation.summary(org, "team", "Blue Area")

    for field in ("advisors", "connects", "overdue", "pipeline",
                  "mtd_target", "mtd_cleared", "ytd_target", "ytd_cleared"):
        assert team[field] == engine[field], field
        assert company[field] == engine[field], field
        assert level[field] == engine[field], field


def test_the_team_sheet_figure_stays_separate_from_the_rollup(org):
    """TeamTarget is a published team-level source, not a roll-up of the
    team's advisors, and the two can legitimately disagree. The API keeps
    both — the sheet's under target/achieved/achievement_pct, the roll-up
    under mtd_*. Collapsing them would silently pick a winner for an open
    business question (Phase 1, Q3)."""
    team = team_service.get_team_summary(org, "Blue Area")

    assert team["mtd_target"] == 1100          # rolled up from advisors
    assert team["target"] is None              # no TeamTarget row in this fixture
    assert "achievement_pct" in team


# ---------------------------------------------------------------------
# Scope: one definition of membership
# ---------------------------------------------------------------------

@pytest.mark.parametrize("level,value", ALL_GROUPS)
def test_headcount_matches_the_hierarchy_scope(org, level, value):
    direct = org.query(Advisor).filter(
        hierarchy.scope_filter(level, value),
        Advisor.in_master_sheet.is_(True),
    ).count()
    assert aggregation.headcount(org, level, value) == direct == 2


@pytest.mark.parametrize("level,value", CHAIN_GROUPS)
def test_breakdown_summary_and_engine_scope_identically(org, level, value):
    breakdown = hierarchy_service.get_level_breakdown(org, level, value)
    assert breakdown["advisors"] == aggregation.headcount(org, level, value)
    assert breakdown["connects"] == aggregation.metric_value(org, level, value, "total_connects")


def test_aggregation_scope_is_the_hierarchys_scope():
    """Not equivalent-looking — literally delegated. An independent
    reimplementation is how membership came to differ between a
    breakdown and a comparison."""
    for level, _ in ALL_GROUPS:
        assert str(aggregation.scope(level, "x")) == str(hierarchy.scope_filter(level, "x"))


# ---------------------------------------------------------------------
# The engine is the only implementation
# ---------------------------------------------------------------------

SERVICE_MODULES = [
    "app/services/team_service.py",
    "app/services/company_service.py",
    "app/services/comparison_service.py",
    "app/services/hierarchy_service.py",
]


@pytest.mark.parametrize("module", SERVICE_MODULES)
def test_no_service_rolls_metrics_up_by_hand(module):
    """Structural guard, parsed rather than grepped so prose in a
    docstring doesn't trip it. Each of these modules used to contain its
    own `func.sum(...)` roll-up; a new one reintroduces the drift this
    phase removed.

    hierarchy_service still SELECTS per-advisor columns for its breakdown
    detail — that is row fetching, not aggregation, and uses no
    aggregate function.
    """
    tree = ast.parse(pathlib.Path(module).read_text())
    aggregates = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "func"
        and node.func.attr in {"sum", "avg", "count"}
    }
    assert aggregates == set(), f"{module} aggregates directly: {aggregates}"


def test_the_rollup_rule_is_declared_on_the_metric_not_in_code():
    """Requirement 4: a new metric is added by editing MetricDef alone.
    The engine reads the rule; no branch anywhere names a metric key."""
    for key, metric in METRICS.items():
        assert aggregation.rollup_for(key) is metric.rollup, key


def test_every_ratio_metric_declares_its_components():
    """A metric declaring RATIO without a numerator/denominator would
    silently fall back to summing — the wrong answer, quietly. Catch the
    declaration error instead."""
    for key, metric in METRICS.items():
        if metric.rollup is not Rollup.RATIO:
            continue
        binding = metric.bindings.get("advisor")
        assert binding is not None, key
        assert binding.ratio_numerator is not None, key
        assert binding.ratio_denominator is not None, key


# ---------------------------------------------------------------------
# Comparisons ride on the same engine
# ---------------------------------------------------------------------

def test_a_cross_level_comparison_uses_one_rule(org):
    """Comparing a team against a BCM must not compare a ratio against an
    average. Both sides here cover the same advisors, so a correct
    comparison finds them equal."""
    result = comparison_service.get_comparison(
        org, [("team", "Blue Area"), ("bcm", "BCM Omar")], metric="achievement_pct",
    )
    values = [entity["metrics"]["achievement_pct"] for entity in result["entities"]]

    assert len(values) == 2
    assert all(v == pytest.approx(RATIO_ACHIEVEMENT, abs=0.1) for v in values), values
    # Neither side may be the old averaged reading — that is exactly the
    # shape of a comparison between two different rules.
    assert result["winners"]["achievement_pct"] is None or len(set(values)) == 1


def test_a_leaderboard_ranks_by_the_rolled_up_value(org):
    """Ranking and aggregating must use the same expression, or the top
    row is not the row with the highest value."""
    org.add(Advisor(wid=3, name="Adv C", team="Red Area", company="Graana"))
    org.add(Performance(wid=3, period=PerformancePeriod.MTD, target=100, cleared=99))
    org.commit()

    ir = QueryIR(intent="leaderboard", subject_level="team",
                 metric=MetricRef(key="achievement_pct"),
                 sort=Sort(metric="achievement_pct", direction="desc"))
    rows = compile_and_run(org, ir)

    assert [r["name"] for r in rows] == ["Red Area", "Blue Area"]
    assert rows[0]["value"] == pytest.approx(99.0)
    assert rows[1]["value"] == pytest.approx(RATIO_ACHIEVEMENT, abs=0.1)
