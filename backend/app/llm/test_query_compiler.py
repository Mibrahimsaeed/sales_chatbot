"""Regression test for the documented join-aliasing bug: sorting by one
Performance-backed metric while filtering by a DIFFERENT Performance-backed
metric with a different `period` used to silently apply the filter against
the sort metric's period instead of its own."""

from app.database.models import Advisor, Attendance, Performance, PerformancePeriod, SalesFunnel
from app.llm.query_compiler import compile_and_run
from app.llm.query_ir import Filter, MetricRef, QueryIR, Sort


def _seed_advisor(db, wid, name, mtd_cleared, ytd_cleared):
    db.add(Advisor(wid=wid, name=name, team="Alpha", company="IMARAT"))
    db.add(Performance(wid=wid, period=PerformancePeriod.MTD, cleared=mtd_cleared))
    db.add(Performance(wid=wid, period=PerformancePeriod.YTD, cleared=ytd_cleared))


def test_filter_on_different_period_than_sort_metric_binds_to_its_own_period(db_session):
    # Advisor 1 has the higher MTD figure (so it would sort first) but a
    # low YTD figure that should fail the filter.
    _seed_advisor(db_session, wid=1, name="Advisor One", mtd_cleared=1000, ytd_cleared=50)
    # Advisor 2 has a lower MTD figure but a YTD figure that passes the filter.
    _seed_advisor(db_session, wid=2, name="Advisor Two", mtd_cleared=200, ytd_cleared=2000)
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="mtd_cleared"),
        filters=[Filter(field="ytd_cleared", operator=">", value=1500)],
        sort=Sort(metric="mtd_cleared", direction="desc"),
        limit=10,
    )

    rows = compile_and_run(db_session, ir)

    assert [r["wid"] for r in rows] == [2]
    assert rows[0]["value"] == 200


def test_compound_filter_across_different_models_still_works(db_session):
    """The compiler's primary use case ("high sales but poor attendance"):
    filtering by a metric on a DIFFERENT model than the sort metric never
    needed aliasing and must keep working unchanged by the (model, period)
    join-keying refactor."""
    db_session.add(Advisor(wid=1, name="On Time Advisor", team="Alpha", company="IMARAT"))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, cleared=1000))
    db_session.add(Attendance(wid=1, biometric_status="On Time"))

    db_session.add(Advisor(wid=2, name="Late Advisor", team="Alpha", company="IMARAT"))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, cleared=2000))
    db_session.add(Attendance(wid=2, biometric_status="Late"))
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="mtd_cleared"),
        filters=[Filter(field="attendance_status", operator="=", value="On Time")],
        sort=Sort(metric="mtd_cleared", direction="desc"),
        limit=10,
    )

    rows = compile_and_run(db_session, ir)

    assert [r["wid"] for r in rows] == [1]
    assert rows[0]["value"] == 1000


def test_filter_on_computed_metric_expression_does_not_crash(db_session):
    """Regression: filtering on a metric whose binding is a computed
    expression (total_connects = new + followup) crashed with TypeError
    when _rebind_to_entity treated a BinaryExpression (key=None) as a
    rebindeable column."""
    db_session.add(Advisor(wid=1, name="Busy Advisor", team="Alpha", company="IMARAT"))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, cleared=1000))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=10, mtd_followup_connect=5))

    db_session.add(Advisor(wid=2, name="Idle Advisor", team="Alpha", company="IMARAT"))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, cleared=2000))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=1, mtd_followup_connect=0))
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="mtd_cleared"),
        filters=[Filter(field="total_connects", operator=">", value=5)],
        sort=Sort(metric="mtd_cleared", direction="desc"),
        limit=10,
    )

    rows = compile_and_run(db_session, ir)

    assert [r["wid"] for r in rows] == [1]


def test_attendance_rate_metric_handles_zero_denominator(db_session):
    db_session.add(Advisor(wid=1, name="Regular", team="Alpha", company="IMARAT"))
    db_session.add(Attendance(wid=1, biometric_mtd_ontime=18, biometric_mtd_late=2, biometric_mtd_not_marked=0))
    # zero rows of any attendance kind — rate must compile to NULL, not crash
    db_session.add(Advisor(wid=2, name="Ghost", team="Alpha", company="IMARAT"))
    db_session.add(Attendance(wid=2, biometric_mtd_ontime=0, biometric_mtd_late=0, biometric_mtd_not_marked=0))
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="attendance_rate"),
        sort=Sort(metric="attendance_rate", direction="desc"),
        limit=10,
    )

    rows = compile_and_run(db_session, ir)

    by_wid = {r["wid"]: r["value"] for r in rows}
    assert by_wid[1] == 90.0
    assert by_wid[2] is None


def test_total_meetings_metric_sums_new_and_followup(db_session):
    db_session.add(Advisor(wid=1, name="A", team="Alpha", company="IMARAT"))
    db_session.add(SalesFunnel(wid=1, mtd_new_meeting=3, mtd_followup_meeting=2))
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="total_meetings"),
        sort=Sort(metric="total_meetings", direction="desc"),
        limit=10,
    )

    rows = compile_and_run(db_session, ir)

    assert rows[0]["value"] == 5


def test_mtd_cleared_team_rollup(db_session):
    db_session.add(Advisor(wid=1, name="A", team="Alpha", company="IMARAT"))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, cleared=300))
    db_session.add(Performance(wid=1, period=PerformancePeriod.YTD, cleared=9000))
    db_session.add(Advisor(wid=2, name="B", team="Alpha", company="IMARAT"))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, cleared=700))
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="team",
        metric=MetricRef(key="mtd_cleared"),
        sort=Sort(metric="mtd_cleared", direction="desc"),
        limit=10,
    )

    rows = compile_and_run(db_session, ir)

    # sums only the MTD rows (300+700), never the YTD 9000
    assert rows == [{"wid": None, "name": "Alpha", "team": "Alpha", "company": None, "value": 1000}]


def test_team_level_rollup_still_sums_correctly(db_session):
    """Team/company rollups (sum/avg over binding.expr, unaliased since the
    sort join always claims the unaliased slot) must be unaffected."""
    db_session.add(Advisor(wid=1, name="A", team="Alpha", company="IMARAT"))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=100, mtd_followup_connect=0))
    db_session.add(Advisor(wid=2, name="B", team="Alpha", company="IMARAT"))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=200, mtd_followup_connect=0))
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="team",
        metric=MetricRef(key="total_connects"),
        sort=Sort(metric="total_connects", direction="desc"),
        limit=10,
    )

    rows = compile_and_run(db_session, ir)

    assert rows == [{"wid": None, "name": "Alpha", "team": "Alpha", "company": None, "value": 300}]
