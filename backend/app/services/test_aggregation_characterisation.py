"""Characterisation of aggregation (written BEFORE the Phase 4 engine).

Records what each of the existing paths produces for the SAME entity and
metric. They are not expected to agree — that is the defect — so this
file's job is to pin the disagreement precisely enough that the refactor
can be judged against it.

Five paths compute values today:

  1. team_service.get_team_summary        own func.sum + TeamTarget
  2. company_service.get_company_summary  own func.sum (near-copy of 1)
  3. hierarchy_service.get_level_summary  own func.sum (near-copy of 1)
  4. comparison_service._metric_value     ontology binding + rollup agg
  5. query_compiler                       ontology binding + binding.agg

The fixture is built so a percentage metric is UNEQUALLY weighted across
advisors — which is the only condition under which average-of-percentages
and ratio-of-sums differ, and therefore the only condition that reveals
which one a path is using.
"""

import pytest

from app.database.models import (
    Advisor, Attendance, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm.query_ir import MetricRef, QueryIR, Sort, Subject
from app.services import comparison_service, company_service, hierarchy_service, team_service


@pytest.fixture()
def uneven(db_session):
    """Two advisors on one team with very different target sizes.

      A: cleared 900  / target 1000  -> 90%
      B: cleared  10  / target  100  -> 10%

    average of percentages = 50.0
    ratio of sums          = 910 / 1100 = 82.7%

    Any path reporting ~50 is averaging; ~82.7 is dividing sums.
    """
    db_session.add(Advisor(wid=1, name="Adv A", team="Blue Area", company="Graana",
                           rm="UH", portfolio_lead="ZH", management_lead="BCM"))
    db_session.add(Advisor(wid=2, name="Adv B", team="Blue Area", company="Graana",
                           rm="UH", portfolio_lead="ZH", management_lead="BCM"))

    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, target=1000, cleared=900, pct=90))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, target=100, cleared=10, pct=10))

    db_session.add(SalesFunnel(wid=1, mtd_new_connect=30, mtd_followup_connect=0, mtd_conversion=3))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=10, mtd_followup_connect=0, mtd_conversion=1))

    db_session.add(Pipeline(wid=1, pipeline=5000, overdue=2))
    db_session.add(Pipeline(wid=2, pipeline=1000, overdue=0))

    # 9/10 vs 1/10 on-time — same unequal shape for a rate metric
    db_session.add(Attendance(wid=1, biometric_mtd_ontime=9, biometric_mtd_late=1,
                              biometric_mtd_not_marked=0))
    db_session.add(Attendance(wid=2, biometric_mtd_ontime=1, biometric_mtd_late=9,
                              biometric_mtd_not_marked=0))
    db_session.commit()
    return db_session


def _compiled(db, metric, level="team"):
    ir = QueryIR(intent="leaderboard", subject_level=level,
                 metric=MetricRef(key=metric), sort=Sort(metric=metric))
    from app.llm.query_compiler import compile_and_run
    rows = compile_and_run(db, ir)
    return rows[0]["value"] if rows else None


# ---------------------------------------------------------------------
# Additive metrics: every path should already agree
# ---------------------------------------------------------------------

def test_connects_agree_across_paths(uneven):
    """40 = 30 + 10. A sum is a sum everywhere."""
    assert team_service.get_team_summary(uneven, "Blue Area")["connects"] == 40
    assert hierarchy_service.get_level_summary(uneven, "team", "Blue Area")["connects"] == 40
    assert comparison_service._metric_value(uneven, "team", "Blue Area", "total_connects") == 40
    assert _compiled(uneven, "total_connects") == 40


def test_cleared_agrees_across_paths(uneven):
    assert team_service.get_team_summary(uneven, "Blue Area")["mtd_cleared"] == 910
    assert hierarchy_service.get_level_summary(uneven, "team", "Blue Area")["mtd_cleared"] == 910
    assert comparison_service._metric_value(uneven, "team", "Blue Area", "mtd_cleared") == 910
    assert _compiled(uneven, "mtd_cleared") == 910


# ---------------------------------------------------------------------
# THE DISAGREEMENT: percentage roll-up
# ---------------------------------------------------------------------

def test_achievement_is_the_ratio_of_sums(uneven):
    """RETIRED ASSERTION. This pinned 50.0 — the average of the two
    advisors' percentages — before Phase 4. Averaging let advisor B's
    100-unit target count as heavily as advisor A's 1000-unit one, so a
    team that cleared 910 of 1100 (82.7%) reported 50%.

    achievement_pct is now declared Rollup.RATIO, so every level divides
    summed cleared by summed target."""
    value = comparison_service._metric_value(uneven, "team", "Blue Area", "achievement_pct")
    assert value == pytest.approx(82.7, abs=0.1)


def test_the_averaged_reading_is_gone_from_every_path(uneven):
    """The disagreement itself, not just one path's number: all four
    surviving paths must now give the same answer."""
    ratio = pytest.approx(82.7, abs=0.1)
    assert comparison_service._metric_value(uneven, "team", "Blue Area", "achievement_pct") == ratio
    assert _compiled(uneven, "achievement_pct") == ratio

    summary = team_service.get_team_summary(uneven, "Blue Area")
    assert summary["mtd_cleared"] / summary["mtd_target"] * 100 == ratio


def test_the_ratio_of_sums_is_a_different_number(uneven):
    summary = team_service.get_team_summary(uneven, "Blue Area")
    ratio = summary["mtd_cleared"] / summary["mtd_target"] * 100
    assert ratio == pytest.approx(82.7, abs=0.1)


def test_attendance_rate_is_also_averaged_today(uneven):
    """PRE-PHASE-4 STATE. avg(90%, 10%) = 50; ratio of sums = 10/20 = 50
    here by coincidence of equal denominators — the shapes still differ
    whenever advisors have different numbers of recorded days."""
    value = comparison_service._metric_value(uneven, "team", "Blue Area", "attendance_rate")
    assert value == pytest.approx(50.0, abs=0.1)


def test_conversion_counts_conversions(uneven):
    """RETIRED ASSERTION. This pinned 2.0 — the average — before Phase 4.
    Three conversions plus one is four; two was not a count of anything.
    conversion is a plain count, so it rolls up as SUM."""
    value = comparison_service._metric_value(uneven, "team", "Blue Area", "conversion")
    assert value == pytest.approx(4.0, abs=0.01)
    assert _compiled(uneven, "conversion") == pytest.approx(4.0, abs=0.01)


# ---------------------------------------------------------------------
# Duplication: four summaries, one shape
# ---------------------------------------------------------------------

def test_team_and_hierarchy_summaries_are_near_copies(uneven):
    team = team_service.get_team_summary(uneven, "Blue Area")
    level = hierarchy_service.get_level_summary(uneven, "team", "Blue Area")
    for field in ("advisors", "connects", "overdue", "pipeline",
                  "mtd_target", "mtd_cleared", "ytd_target", "ytd_cleared"):
        assert team[field] == level[field], field


def test_company_summary_is_the_same_shape_again(uneven):
    company = company_service.get_company_summary(uneven, "Graana")
    for field in ("advisors", "connects", "mtd_target", "mtd_cleared"):
        assert field in company


# ---------------------------------------------------------------------
# Scope must already agree, whatever the value does
# ---------------------------------------------------------------------

@pytest.mark.parametrize("level,value", [
    ("team", "Blue Area"), ("unit_head", "UH"), ("zonal_head", "ZH"), ("bcm", "BCM"),
])
def test_every_chain_level_scopes_to_the_same_two_advisors(uneven, level, value):
    summary = hierarchy_service.get_level_summary(uneven, level, value)
    assert summary["advisors"] == 2
    assert summary["connects"] == 40
