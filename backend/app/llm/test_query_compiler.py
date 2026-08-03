"""Regression test for the documented join-aliasing bug: sorting by one
Performance-backed metric while filtering by a DIFFERENT Performance-backed
metric with a different `period` used to silently apply the filter against
the sort metric's period instead of its own."""

from app.database.models import Advisor, Attendance, Performance, PerformancePeriod, SalesFunnel, TeamTarget
from app.llm.query_compiler import compile_and_run, count_ir
from app.llm.query_ir import Filter, MetricRef, QueryIR, Sort, Subject


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


def test_non_master_sheet_advisor_excluded_from_advisor_level_leaderboard(db_session):
    """A WID that only ever appeared in a raw activity sheet (never the
    MasterSheet) must not show up in leaderboards alongside real advisors
    — this used to mix "who has most connects" results with junk data."""
    db_session.add(Advisor(wid=1, name="Real Advisor", team="Alpha", company="IMARAT", in_master_sheet=True))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=100, mtd_followup_connect=0))
    db_session.add(Advisor(wid=2, name="Raw Data Ghost", team="Alpha", company="IMARAT", in_master_sheet=False))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=900, mtd_followup_connect=0))
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="total_connects"),
        sort=Sort(metric="total_connects", direction="desc"),
        limit=10,
    )

    rows = compile_and_run(db_session, ir)

    assert [r["name"] for r in rows] == ["Real Advisor"]


def test_non_master_sheet_advisor_excluded_from_team_rollup(db_session):
    db_session.add(Advisor(wid=1, name="A", team="Alpha", company="IMARAT", in_master_sheet=True))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=100, mtd_followup_connect=0))
    db_session.add(Advisor(wid=2, name="Ghost", team="Alpha", company="IMARAT", in_master_sheet=False))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=900, mtd_followup_connect=0))
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="team",
        metric=MetricRef(key="total_connects"),
        sort=Sort(metric="total_connects", direction="desc"),
        limit=10,
    )

    rows = compile_and_run(db_session, ir)

    # only the real advisor's 100 connects should count, not the ghost's 900
    assert rows == [{"wid": None, "name": "Alpha", "team": "Alpha", "company": None, "value": 100}]


# ---- Part 8: pagination — count_ir + compile_and_run(offset=) ----

def _seed_n_advisors(db, n, team="Alpha", company="IMARAT", start_wid=1):
    for i in range(n):
        wid = start_wid + i
        db.add(Advisor(wid=wid, name=f"Advisor {i + 1}", team=team, company=company))
        db.add(SalesFunnel(wid=wid, mtd_new_connect=100 - i, mtd_followup_connect=0))


def _ir(**overrides):
    base = dict(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="total_connects"),
        sort=Sort(metric="total_connects", direction="desc"),
        limit=10,
    )
    base.update(overrides)
    return QueryIR(**base)


def test_count_ir_matches_total_rows_at_advisor_level(db_session):
    _seed_n_advisors(db_session, 20)
    db_session.commit()

    assert count_ir(db_session, _ir(limit=None)) == 20


def test_count_ir_ignores_limit_returns_true_total(db_session):
    """count_ir reports the TRUE total regardless of ir.limit — capping
    against ir.limit is the caller's (chat_service's) job, not the
    compiler's, so the same count_ir call works whether or not the
    query has an explicit top-N."""
    _seed_n_advisors(db_session, 20)
    db_session.commit()

    assert count_ir(db_session, _ir(limit=5)) == 20


def test_count_ir_counts_groups_not_rows_for_team_rollup(db_session):
    _seed_n_advisors(db_session, 5, team="Alpha", start_wid=1)
    _seed_n_advisors(db_session, 3, team="Beta", start_wid=101)
    db_session.commit()

    total = count_ir(db_session, _ir(subject_level="team", limit=None))
    assert total == 2  # two teams, not eight advisor rows


def test_compile_and_run_offset_returns_next_slice(db_session):
    _seed_n_advisors(db_session, 20)
    db_session.commit()

    page1 = compile_and_run(db_session, _ir(limit=10), offset=0)
    page2 = compile_and_run(db_session, _ir(limit=10), offset=10)

    assert [r["name"] for r in page1] == [f"Advisor {i}" for i in range(1, 11)]
    assert [r["name"] for r in page2] == [f"Advisor {i}" for i in range(11, 21)]
    assert not set(r["wid"] for r in page1) & set(r["wid"] for r in page2)


# ---- Hierarchy rework: unit_head / zonal_head / business_center ----

def test_unit_head_rollup_uses_generic_advisor_binding_fallback(db_session):
    """total_connects has no explicit "unit_head" binding in metric_
    ontology.py — the compiler must fall back to the advisor-level binding
    and roll it up via Advisor.bm, the same way an explicit team binding
    would, without a new ontology entry per level."""
    db_session.add(Advisor(wid=1, name="A", team="Alpha", bm="Zeeshan Tariq", rm="Zeeshan Tariq"))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=10, mtd_followup_connect=0))
    db_session.add(Advisor(wid=2, name="B", team="Beta", bm="Zeeshan Tariq", rm="Zeeshan Tariq"))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=20, mtd_followup_connect=0))
    db_session.add(Advisor(wid=3, name="C", team="Gamma", bm="Someone Else", rm="Someone Else"))
    db_session.add(SalesFunnel(wid=3, mtd_new_connect=900, mtd_followup_connect=0))
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="unit_head",
        metric=MetricRef(key="total_connects"),
        sort=Sort(metric="total_connects", direction="desc"),
        limit=10,
    )
    rows = compile_and_run(db_session, ir)

    by_name = {r["name"]: r["value"] for r in rows}
    assert by_name["Zeeshan Tariq"] == 30
    assert by_name["Someone Else"] == 900


def test_zonal_head_metric_filter_uses_hierarchy_column(db_session):
    db_session.add(Advisor(wid=1, name="A", team="Alpha", zm="Ahmed Ali", portfolio_lead="Ahmed Ali"))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=10, mtd_followup_connect=0))
    db_session.add(Advisor(wid=2, name="B", team="Beta", zm="Bilal Khan", portfolio_lead="Bilal Khan"))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=999, mtd_followup_connect=0))
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="total_connects"),
        filters=[Filter(field="zonal_head", operator="=", value="Ahmed Ali")],
        sort=Sort(metric="total_connects", direction="desc"),
        limit=10,
    )
    rows = compile_and_run(db_session, ir)

    assert [r["name"] for r in rows] == ["A"]


def test_business_center_comparison_subject_filter(db_session):
    db_session.add(Advisor(wid=1, name="A", team="Alpha", office="Center One"))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, cleared=100))
    db_session.add(Advisor(wid=2, name="B", team="Beta", office="Center Two"))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, cleared=200))
    db_session.add(Advisor(wid=3, name="C", team="Gamma", office="Center Three"))
    db_session.add(Performance(wid=3, period=PerformancePeriod.MTD, cleared=999))
    db_session.commit()

    ir = QueryIR(
        intent="comparison",
        subject_level="office",
        subjects=[
            Subject(type="office", value="Center One", resolved_id="Center One"),
            Subject(type="office", value="Center Two", resolved_id="Center Two"),
        ],
        metric=MetricRef(key="mtd_cleared"),
        sort=Sort(metric="mtd_cleared", direction="desc"),
        limit=10,
    )
    rows = compile_and_run(db_session, ir)

    assert {r["name"] for r in rows} == {"Center One", "Center Two"}
    assert sum(r["value"] for r in rows) == 300


def test_leaderboard_with_single_subject_returns_only_that_subject(db_session):
    """Bug fix (live-reported): 'show me unit head X's performance' has the
    LLM ground X into ir.subjects while keeping intent="leaderboard" (it
    isn't comparing two things) — this used to be silently dropped
    (subject filtering only ran for intent=="comparison"), so the query
    ran as an unfiltered top-N ranking of EVERY unit head instead of the
    one asked about."""
    # PHASE 4: achievement at a group level is now the ratio of sums, so
    # the fixture carries the components it is a ratio OF. It used to set
    # only `pct`, which worked while the roll-up was avg(pct).
    db_session.add(Advisor(wid=1, name="A", team="Alpha", bm="Zeeshan Tariq", rm="Zeeshan Tariq"))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, pct=50, cleared=50, target=100))
    db_session.add(Advisor(wid=2, name="B", team="Beta", bm="Someone Else", rm="Someone Else"))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, pct=99, cleared=99, target=100))
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="unit_head",
        subjects=[Subject(type="unit_head", value="Zeeshan Tariq", match_confidence=1.0)],
        metric=MetricRef(key="achievement_pct"),
        sort=Sort(metric="achievement_pct", direction="desc"),
        limit=None,
    )
    rows = compile_and_run(db_session, ir)

    assert [r["name"] for r in rows] == ["Zeeshan Tariq"]
    assert rows[0]["value"] == 50


def test_comparison_still_requires_exact_subject_type_match(db_session):
    """Regression guard for the fix above: broadening subject filtering to
    every intent must not also start matching subjects of a DIFFERENT type
    than subject_level."""
    db_session.add(Advisor(wid=1, name="A", team="Alpha", company="Graana"))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, cleared=100))
    db_session.add(Advisor(wid=2, name="B", team="Beta", company="IMARAT"))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, cleared=200))
    db_session.commit()

    ir = QueryIR(
        intent="comparison",
        subject_level="team",
        subjects=[
            Subject(type="team", value="Alpha", match_confidence=1.0),
            Subject(type="company", value="IMARAT", match_confidence=1.0),  # wrong type — must be ignored
        ],
        metric=MetricRef(key="mtd_cleared"),
        sort=Sort(metric="mtd_cleared", direction="desc"),
        limit=10,
    )
    rows = compile_and_run(db_session, ir)

    assert [r["name"] for r in rows] == ["Alpha"]


def test_company_rolls_up_advisors_like_every_other_group_level(db_session):
    """RETIRED ASSERTION. This asserted None — the advisor-binding
    fallback was withheld from `company`, so "how many connects did
    IMARAT make" answered nothing while the identical question about a
    unit head answered fine.

    PHASE 4 makes binding selection uniform: a company is a set of
    advisors, `hierarchy.scope_filter` already knows which ones, and the
    engine rolls their connects up like any other group."""
    db_session.add(Advisor(wid=1, name="A", team="Alpha", company="IMARAT"))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=10, mtd_followup_connect=0))
    db_session.add(Advisor(wid=2, name="B", team="Beta", company="IMARAT"))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=5, mtd_followup_connect=0))
    db_session.add(Advisor(wid=3, name="C", team="Gamma", company="Graana"))
    db_session.add(SalesFunnel(wid=3, mtd_new_connect=7, mtd_followup_connect=0))
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="company",
        metric=MetricRef(key="total_connects"),
        sort=Sort(metric="total_connects", direction="desc"),
        limit=10,
    )
    rows = compile_and_run(db_session, ir)

    assert [(r["name"], r["value"]) for r in rows] == [("IMARAT", 15), ("Graana", 7)]


def test_count_ir_counts_teams_from_their_advisors(db_session):
    """RETIRED ASSERTION. This counted TeamTarget rows, because team
    achievement resolved to the team-named TeamTarget binding while every
    other level rolled advisors up — the same metric with two sources.

    PHASE 4 routes team achievement through the engine like any other
    group, so teams are counted from the advisors in scope. The sheet's
    own team figure is still available; team_service reads it explicitly
    (see `sheet_target` there) rather than it arriving here disguised as
    an aggregate."""
    for wid, team in ((1, "Alpha"), (2, "Alpha"), (3, "Beta")):
        db_session.add(Advisor(wid=wid, name=f"A{wid}", team=team))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=1000, cleared=500))
    db_session.commit()

    ir = QueryIR(
        intent="leaderboard",
        subject_level="team",
        metric=MetricRef(key="achievement_pct"),
        sort=Sort(metric="achievement_pct", direction="desc"),
        limit=None,
    )
    assert count_ir(db_session, ir) == 2
