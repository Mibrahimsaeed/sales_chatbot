"""The binding layer for the fields the sheet audit imported.

The ETL landed six sources; nothing read them. Each metric below turns
stored-but-inert data into an answerable measure:

    Advisor.unit                      -> one_unit_ratio
    login_mtd_ontime/late/not_marked  -> login_rate
    mtd_meetings_planned/conducted    -> meetings_planned / _conducted
                                         / meeting_conduction_rate
    SalesFunnel.ytd_* , Pipeline.ytd_* -> 9 YTD siblings

Every test asserts a VALUE, not just that a binding exists. A binding
that resolves and computes the wrong number is worse than none.
"""

import pytest

from app.database.models import (
    Advisor, Attendance, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import aggregation
from app.llm.metric_ontology import (
    METRICS, ONE_UNIT_THRESHOLDS, metric_for_period, resolve_metric,
    status_for, supported_periods,
)
from app.llm.query_compiler import compile_and_run, is_answerable
from app.llm.query_ir import MetricRef, QueryIR, Sort


@pytest.fixture()
def org(db_session):
    """Four advisors on one team.

      units:   2, 1, 0, 0     -> 2 of 4 hold a unit  = 50%
      login:   ontime 18/16/12/6, late 2/4/8/14, not_marked 0
               -> 52 on time of 80 recorded          = 65%
      IBD:     planned 10/10/10/10 = 40, conducted 8/6/4/2 = 20
               -> conduction                          = 50%
      YTD:     each MTD figure x10
    """
    rows = [(1, "2", 18, 2, 8), (2, "1", 16, 4, 6), (3, "0", 12, 8, 4), (4, None, 6, 14, 2)]
    for wid, unit, ontime, late, conducted in rows:
        db_session.add(Advisor(wid=wid, name=f"Adv {wid}", team="Blue Area",
                               company="Graana", unit=unit, in_master_sheet=True))
        db_session.add(Attendance(
            wid=wid, biometric_mtd_ontime=ontime, biometric_mtd_late=late,
            biometric_mtd_not_marked=0,
            login_mtd_ontime=ontime, login_mtd_late=late, login_mtd_not_marked=0,
        ))
        db_session.add(SalesFunnel(
            wid=wid, mtd_new_connect=10, mtd_followup_connect=0, mtd_cr=5,
            mtd_new_meeting=4, mtd_followup_meeting=0, mtd_conversion=2,
            mtd_booking_stored=3,
            mtd_meetings_planned=10, mtd_meetings_conducted=conducted,
            ytd_new_connect=100, ytd_followup_connect=0, ytd_cr=50,
            ytd_new_meeting=40, ytd_followup_meeting=0, ytd_conversion=20,
            ytd_booking_stored=30,
        ))
        db_session.add(Pipeline(wid=wid, pipeline=1000, overdue=2,
                                ytd_pipeline=10000, ytd_overdue=20))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=50, pct=50))
    db_session.commit()
    return db_session


def _team(db, metric):
    return aggregation.metric_value(db, "team", "Blue Area", metric)


# =====================================================================
# 1 Unit Ratio
# =====================================================================

def test_one_unit_ratio_is_advisors_with_units_over_team_size(org):
    """SPEC: `advisorsWithUnits / teamSize x 100`. Two of four advisors
    hold a unit."""
    assert _team(org, "one_unit_ratio") == pytest.approx(50.0)


def test_one_unit_counts_any_non_zero_tally(org):
    """`Unit` is a tally, not a flag — an advisor with 2 units counts
    once, not twice. The spec counts ADVISORS, not units."""
    assert _team(org, "one_unit_ratio") == pytest.approx(50.0)
    assert _team(org, "one_unit_ratio") != pytest.approx(75.0)   # 3 units / 4


def test_a_null_unit_counts_as_no_unit(org):
    """Advisor 4 has unit=None — an advisor the "1 Unit" tab never
    listed. Absent must read as "no unit", not be excluded from team
    size, or the denominator shrinks and the ratio inflates."""
    assert aggregation.headcount(org, "team", "Blue Area") == 4
    assert _team(org, "one_unit_ratio") == pytest.approx(50.0)


def test_one_unit_uses_its_own_thresholds():
    """SPEC grounding prompt #3: "1-Unit uses 45/30", not 85/60."""
    assert ONE_UNIT_THRESHOLDS.green == 45.0
    assert ONE_UNIT_THRESHOLDS.yellow == 30.0
    assert status_for("one_unit_ratio", 50) == "green"
    assert status_for("one_unit_ratio", 35) == "yellow"
    assert status_for("one_unit_ratio", 20) == "red"
    # The same values under the default banding would read differently.
    assert status_for("achievement_pct", 50) == "red"


def test_one_unit_defaults_to_a_group_level(org):
    """One advisor's ratio is 0 or 100, which answers nothing. An
    unqualified question ranks teams."""
    assert METRICS["one_unit_ratio"].primary_level == "team"


def test_one_unit_is_answerable_at_every_group_level():
    for level in ("team", "unit_head", "zonal_head", "bcm", "company", "office"):
        assert is_answerable("one_unit_ratio", level), level


def test_one_unit_binds_to_advisor_without_a_self_join(org):
    """The only metric whose column lives on the advisor row. Advisor is
    already the query root, so joining it would emit
    `FROM advisors JOIN advisors ON advisors.wid = advisors.wid`."""
    ir = QueryIR(intent="leaderboard", subject_level="team",
                 metric=MetricRef(key="one_unit_ratio"),
                 sort=Sort(metric="one_unit_ratio"))
    rows = compile_and_run(org, ir)
    assert rows[0]["value"] == pytest.approx(50.0)


def test_one_unit_resolves_from_business_language(org):
    for phrase in ("1 unit ratio", "one unit ratio", "1-unit ratio", "unit ownership"):
        assert resolve_metric(phrase) == "one_unit_ratio", phrase


# =====================================================================
# WorksApp Login
# =====================================================================

def test_login_rate_is_the_ratio_of_sums(org):
    """52 on time of 80 recorded = 65%. Ratio-of-sums, so an advisor with
    more recorded days weighs more — the Phase 4 rule."""
    assert _team(org, "login_rate") == pytest.approx(65.0)


def test_login_rate_includes_not_marked_in_the_denominator(org):
    """The column the ETL had to import. Without it the denominator is
    ontime+late only, and every not-marked day inflates the rate."""
    org.add(Advisor(wid=9, name="Adv 9", team="Solo", company="Graana",
                    in_master_sheet=True))
    org.add(Attendance(wid=9, login_mtd_ontime=5, login_mtd_late=0,
                       login_mtd_not_marked=5))
    org.commit()
    # 5 / (5 + 0 + 5) = 50%, not 5/5 = 100%.
    assert aggregation.metric_value(org, "team", "Solo", "login_rate") == pytest.approx(50.0)


def test_login_rate_is_separate_from_biometric(org):
    """Two different sources. They coincide in this fixture by
    construction, so the test asserts they are distinct METRICS rather
    than distinct numbers."""
    assert METRICS["login_rate"].bindings["advisor"].model is Attendance
    assert resolve_metric("worksapp login") == "login_rate"
    assert resolve_metric("attendance rate") == "attendance_rate"


def test_login_rate_uses_the_default_bands():
    assert status_for("login_rate", 90) == "green"
    assert status_for("login_rate", 65) == "yellow"
    assert status_for("login_rate", 40) == "red"


# =====================================================================
# IBD meetings
# =====================================================================

def test_meetings_planned_and_conducted_are_counts(org):
    assert _team(org, "meetings_planned") == 40
    assert _team(org, "meetings_conducted") == 20


def test_meeting_conduction_rate_is_conducted_over_planned(org):
    """SPEC: `round(Conducted / Planned x 100)`. 20 of 40 = 50%."""
    assert _team(org, "meeting_conduction_rate") == pytest.approx(50.0)


def test_conduction_rate_needs_no_working_days(org):
    """Unlike the other IBD board, this one is a ratio of two stored
    counts — so it ships while "Meetings Planned" (a working-day target
    rate) stays blocked."""
    assert _team(org, "meeting_conduction_rate") is not None
    # RETIRED REFUSAL: working_days.py made the meeting RATE computable.
    # The point of this test is that the CONDUCTION rate (planned vs
    # conducted, both stored) is a different measure that never needed a
    # working-day calendar — so the two must stay distinct keys.
    assert resolve_metric("meeting rate") == "meeting_rate"
    assert resolve_metric("meeting conduction rate") == "meeting_conduction_rate"


def test_conduction_rate_is_no_data_when_nothing_was_planned(org):
    """NULLIF, not a division error, and not 0%. A team that planned
    nothing has no conduction rate."""
    org.add(Advisor(wid=8, name="Adv 8", team="Idle", company="Graana",
                    in_master_sheet=True))
    org.add(SalesFunnel(wid=8, mtd_meetings_planned=0, mtd_meetings_conducted=0))
    org.commit()
    assert aggregation.metric_value(org, "team", "Idle", "meeting_conduction_rate") is None


def test_ibd_metrics_resolve_from_business_language():
    assert resolve_metric("meetings planned") == "meetings_planned"
    assert resolve_metric("meetings conducted") == "meetings_conducted"
    assert resolve_metric("conduction rate") == "meeting_conduction_rate"
    # The raw meeting COUNT is a different measure and keeps its phrase.
    assert resolve_metric("meetings") == "total_meetings"


# =====================================================================
# YTD siblings
# =====================================================================

YTD_PAIRS = [
    ("total_connects", "ytd_connects", 40, 400),
    ("new_connects", "ytd_new_connects", 40, 400),
    ("followup_connects", "ytd_followup_connects", 0, 0),
    ("client_registrations", "ytd_client_registrations", 20, 200),
    ("total_meetings", "ytd_meetings", 16, 160),
    ("conversion", "ytd_conversion", 8, 80),
    ("bookings", "ytd_bookings", 12, 120),
    ("pipeline_value", "ytd_pipeline_value", 4000, 40000),
    ("overdue", "ytd_overdue", 8, 80),
]


@pytest.mark.parametrize("mtd_key,ytd_key,mtd_value,ytd_value", YTD_PAIRS)
def test_each_ytd_sibling_reads_its_own_column(org, mtd_key, ytd_key, mtd_value, ytd_value):
    assert _team(org, mtd_key) == pytest.approx(mtd_value), mtd_key
    assert _team(org, ytd_key) == pytest.approx(ytd_value), ytd_key


@pytest.mark.parametrize("mtd_key,ytd_key,_m,_y", YTD_PAIRS)
def test_the_period_resolver_swaps_the_pair(org, mtd_key, ytd_key, _m, _y):
    """The point of `period_family`: asking for the MTD measure at YTD
    resolves to its sibling instead of refusing."""
    assert metric_for_period(mtd_key, "YTD") == ytd_key
    assert metric_for_period(mtd_key, "MTD") == mtd_key


@pytest.mark.parametrize("mtd_key,_y,_m,_v", YTD_PAIRS)
def test_three_month_is_still_refused(mtd_key, _y, _m, _v):
    """Only the YTD tabs were imported. 3M has no source for these, and
    must refuse rather than fall back to MTD."""
    assert metric_for_period(mtd_key, "3M") is None


@pytest.mark.parametrize("mtd_key,ytd_key,_m,_y", YTD_PAIRS)
def test_both_periods_are_reported_as_supported(mtd_key, ytd_key, _m, _y):
    """The YTD import's guarantee, unchanged: every one of these measures
    reports BOTH of its periods.

    Asserted as a subset since Phase 12, because connects also gained a
    real DAILY sibling (calls.connects_daily) and an equality check would
    now fail for having MORE data than it did. The guarantee this test
    exists for is that neither MTD nor YTD goes missing; DAILY is pinned
    separately, per measure, in test_daily_metrics.py — a blanket
    equality here would silently accept a daily binding appearing on a
    measure that has no daily source.
    """
    periods = {p.value for p in supported_periods(mtd_key)}
    assert {"MTD", "YTD"} <= periods, mtd_key
    assert periods <= {"MTD", "YTD", "DAILY"}, mtd_key


def test_a_ytd_query_returns_ytd_numbers_end_to_end(org):
    """The whole chain: the user says "year to date", the resolver swaps
    the metric, and the compiler reads the ytd_ column."""
    ir = QueryIR(intent="leaderboard", subject_level="team",
                 metric=MetricRef(key="total_connects"),
                 sort=Sort(metric="total_connects"))
    ir.time_range.period = "YTD"
    assert compile_and_run(org, ir)[0]["value"] == pytest.approx(400)

    ir.time_range.period = "MTD"
    assert compile_and_run(org, ir)[0]["value"] == pytest.approx(40)


def test_ytd_overdue_keeps_its_polarity_and_rule():
    """Lower is better in January as much as in December, and green only
    at zero."""
    assert METRICS["ytd_overdue"].lower_is_better is True
    assert status_for("ytd_overdue", 0) == "green"
    assert status_for("ytd_overdue", 1) == "red"


def test_ytd_pass_fail_metrics_match_their_mtd_siblings():
    for mtd_key, ytd_key in (("conversion", "ytd_conversion"),
                             ("pipeline_value", "ytd_pipeline_value")):
        assert status_for(ytd_key, 0) == status_for(mtd_key, 0) == "red"
        assert status_for(ytd_key, 5) == status_for(mtd_key, 5) == "green"


# =====================================================================
# Backward compatibility and structure
# =====================================================================

def test_no_mtd_value_changed(org):
    """The whole binding layer is additive. Every pre-existing metric
    must read exactly what it read before."""
    assert _team(org, "total_connects") == 40
    assert _team(org, "total_meetings") == 16
    assert _team(org, "conversion") == 8
    assert _team(org, "pipeline_value") == 4000
    assert _team(org, "overdue") == 8
    assert _team(org, "attendance_rate") == pytest.approx(65.0)
    assert _team(org, "achievement_pct") == pytest.approx(50.0)


def test_every_new_metric_is_nameable():
    """The import-time guard covers this, but naming the metrics here
    makes a silent alias removal fail loudly."""
    for key in ("one_unit_ratio", "login_rate", "meetings_planned",
                "meetings_conducted", "meeting_conduction_rate",
                *(ytd for _m, ytd, _a, _b in YTD_PAIRS)):
        assert METRICS[key].synonyms, key


def test_the_ytd_metrics_are_generated_not_hand_written():
    """Nine near-identical MetricDefs written out would be nine chances
    to pair a column with the wrong family."""
    from app.llm.metric_ontology import _YTD_SPECS

    assert len(_YTD_SPECS) == len(YTD_PAIRS)
    for key, _label, _family, _model, _expr, _thr, _lower in _YTD_SPECS:
        assert key in METRICS


def test_no_metric_produces_a_cartesian_product(org):
    """The guard from the parity phase, re-run over the enlarged
    ontology — a new binding referencing a second table without joining
    it returns a plausible number rather than failing."""
    import warnings

    offenders = []
    for key in METRICS:
        for level in ("advisor", "team"):
            ir = QueryIR(intent="leaderboard", subject_level=level,
                         metric=MetricRef(key=key), sort=Sort(metric=key))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                compile_and_run(org, ir)
            if any("cartesian product" in str(w.message).lower() for w in caught):
                offenders.append(f"{key}@{level}")
    assert not offenders, offenders
