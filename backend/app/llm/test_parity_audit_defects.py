"""Defects found by the end-to-end parity audit (212 realistic queries).

Three were CRASHES on ordinary questions, two returned wrong numbers, one
described a business concept that does not exist. All six were reachable
from plain user phrasing; none required an unusual query.

  D1  A leaderboard containing an advisor with a zero denominator raised
      TypeError and the whole reply failed.
  D2  advisor_service ignored ColumnBinding.join_models, cross-joined the
      second table, and .scalar() raised MultipleResultsFound.
  D3  The same function self-joined Advisor for a metric bound to the
      advisor row -> "ambiguous column name".
  D4  achievement_pct read the sheet's `pct` at advisor level and
      computed cleared/target at group level: 99% or 84.7% for the same
      person, depending which way you asked.
  D5  `overdue` and `overdue_amount` bound ONE column under two labels
      claiming different units.
  D6  Every percentage metric was narrated as target attainment, so a
      1-Unit ratio "achieved 66.7% of the assigned target, remaining
      33.3% short of the monthly goal" — a target that does not exist.
"""

import warnings

import pytest

from app.database.models import (
    Advisor, Attendance, Calls, Performance, PerformancePeriod, Pipeline,
    Portfolio, SalesFunnel,
)
from app.llm import aggregation, entity_extractor
from app.llm.metric_ontology import (
    METRICS, is_percentage_metric, measures_target_attainment, resolve_metric,
)
from app.llm.query_compiler import compile_and_run
from app.llm.query_ir import MetricRef, QueryIR, Sort
from app.llm.response_formatter import format_metric_value
from app.services import advisor_service


@pytest.fixture()
def org(db_session):
    """Three advisors, one of whom has NOTHING — no attendance days, no
    meetings planned, no target. That advisor is the audit's most
    productive fixture: every zero-denominator path runs through them.
    """
    rows = [
        # wid, name,   unit, cleared, target, ontime, late, planned, conducted
        (1, "Adv Full", "2", 900, 1000, 18, 2, 10, 8),
        (2, "Adv Half", "0", 500, 1000, 10, 10, 10, 4),
        (3, "Adv Empty", None,  0,    0,  0,  0,  0, 0),
    ]
    for wid, name, unit, cleared, target, ontime, late, planned, conducted in rows:
        db_session.add(Advisor(wid=wid, name=name, team="Blue Area", company="Graana",
                               unit=unit, in_master_sheet=True))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=target, cleared=cleared,
                                   pct=(cleared / target * 100) if target else 0))
        db_session.add(Attendance(wid=wid, biometric_mtd_ontime=ontime,
                                  biometric_mtd_late=late, biometric_mtd_not_marked=0,
                                  login_mtd_ontime=ontime, login_mtd_late=late,
                                  login_mtd_not_marked=0))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=10, mtd_followup_connect=0,
                                   mtd_cr=5, mtd_new_meeting=4, mtd_followup_meeting=0,
                                   mtd_conversion=2,
                                   mtd_meetings_planned=planned,
                                   mtd_meetings_conducted=conducted))
        db_session.add(Pipeline(wid=wid, pipeline=100, overdue=1))
        db_session.add(Portfolio(wid=wid, value=1000))
        db_session.add(Calls(wid=wid, answered_calls_mtd=20, connects_mtd=10))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    return db_session


def _leaderboard(db, metric, level="advisor"):
    ir = QueryIR(intent="leaderboard", subject_level=level,
                 metric=MetricRef(key=metric), sort=Sort(metric=metric))
    return compile_and_run(db, ir) or []


# =====================================================================
# D1 — a NULL value must not crash the reply
# =====================================================================

@pytest.mark.parametrize("metric", [
    "attendance_rate", "login_rate", "meeting_conduction_rate", "achievement_pct",
])
def test_a_null_row_does_not_crash_the_leaderboard(org, metric):
    """Adv Empty divides by zero on every one of these, so the compiler
    returns NULL by design. The formatter did `f"{value:,.0f}"` on it and
    raised TypeError — one row with nothing to divide by failed the
    ENTIRE response."""
    from app.llm.response_formatter import format_ir_reply

    ir = QueryIR(intent="leaderboard", subject_level="advisor",
                 metric=MetricRef(key=metric), sort=Sort(metric=metric))
    rows = compile_and_run(org, ir)
    assert any(r["value"] is None for r in rows), f"{metric}: fixture no longer exercises NULL"

    reply = format_ir_reply(ir, rows)          # must not raise
    assert "no data" in reply


def test_a_null_value_renders_as_no_data_not_zero():
    """0% and "no data" are different claims. An advisor with no recorded
    days has no attendance rate; calling it 0% ranks them as the worst
    performer rather than as unreported."""
    assert format_metric_value("attendance_rate", None) == "no data"
    assert format_metric_value("attendance_rate", 0) == "0%"


def test_no_metric_crashes_a_leaderboard_at_any_level(org):
    """The property, over the whole ontology. A new RATIO metric gets
    this coverage the day it is declared."""
    from app.llm.response_formatter import format_ir_reply

    for key in METRICS:
        for level in ("advisor", "team"):
            ir = QueryIR(intent="leaderboard", subject_level=level,
                         metric=MetricRef(key=key), sort=Sort(metric=key))
            rows = compile_and_run(org, ir)
            if rows:
                format_ir_reply(ir, rows)      # must not raise


# =====================================================================
# D1b — units
# =====================================================================

@pytest.mark.parametrize("metric,value,expected", [
    ("achievement_pct", 90.0, "90%"),
    ("attendance_rate", 66.666, "66.7%"),
    ("one_unit_ratio", 50.0, "50%"),
    ("mtd_cleared", 1234.0, "1,234"),
    ("total_connects", 100.0, "100"),
    ("overdue", 3.0, "3"),
])
def test_a_value_renders_with_its_unit(metric, value, expected):
    """The leaderboard printed a bare number, so 90% read as "90" —
    indistinguishable from a count of 90."""
    assert format_metric_value(metric, value) == expected


def test_every_percentage_metric_renders_a_percent_sign():
    for key in METRICS:
        if is_percentage_metric(key):
            assert format_metric_value(key, 50.0).endswith("%"), key


# =====================================================================
# D2 / D3 — advisor_service was a third query builder
# =====================================================================

def test_a_cross_table_ratio_works_for_one_advisor(org):
    """connect_to_cr_rate divides SalesFunnel.mtd_cr by
    Calls.answered_calls_mtd. Without the declared join the second table
    was cross-joined, .scalar() saw several rows and raised
    MultipleResultsFound."""
    assert advisor_service.get_advisor_metric(org, 1, "connect_to_cr_rate") == pytest.approx(25.0)


def test_an_advisor_rooted_metric_works_for_one_advisor(org):
    """one_unit_ratio binds to Advisor, which is already the query root.
    Joining it emitted `FROM advisors, advisors` -> ambiguous column."""
    assert advisor_service.get_advisor_metric(org, 1, "one_unit_ratio") == pytest.approx(100.0)
    assert advisor_service.get_advisor_metric(org, 2, "one_unit_ratio") == pytest.approx(0.0)


def test_every_metric_is_answerable_for_one_advisor(org):
    """The property. Three query builders read bindings — the compiler,
    the aggregation engine and this one — and only two honoured
    join_models. Any metric that raises here is a crash on a plain
    "what is X's <metric>" question."""
    for key in METRICS:
        try:
            advisor_service.get_advisor_metric(org, 1, key)
        except Exception as exc:                        # pragma: no cover
            pytest.fail(f"{key}: {type(exc).__name__}: {exc}")


def test_the_advisor_path_produces_no_cartesian_product(org):
    offenders = []
    for key in METRICS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            advisor_service.get_advisor_metric(org, 1, key)
        if any("cartesian product" in str(w.message).lower() for w in caught):
            offenders.append(key)
    assert not offenders, offenders


def test_the_advisor_path_agrees_with_the_aggregation_engine(org):
    """Three builders, one answer. They disagreed for exactly the metrics
    whose bindings the third one mishandled."""
    for key in METRICS:
        one = advisor_service.get_advisor_metric(org, 1, key)
        engine = aggregation.metric_value(org, "advisor", "Adv Full", key)
        if one is None and engine is None:
            continue
        assert one == pytest.approx(engine), key


# =====================================================================
# D4 — achievement had two answers
# =====================================================================

def test_achievement_agrees_at_advisor_and_group_level(db_session):
    """The sheet's `pct` column said 99; its own components say 84.7. The
    advisor level read the column and the group level computed, so the
    same person was 99% or 84.7% depending which way you asked."""
    db_session.add(Advisor(wid=1, name="Split", team="T", company="C", in_master_sheet=True))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD,
                               target=1000, cleared=847, pct=99))
    db_session.commit()

    advisor = aggregation.metric_value(db_session, "advisor", "Split", "achievement_pct")
    team = aggregation.metric_value(db_session, "team", "T", "achievement_pct")

    assert advisor == pytest.approx(84.7)
    assert team == pytest.approx(84.7)
    assert advisor == pytest.approx(team)


def test_achievement_is_computed_not_read(db_session):
    """SPEC: `round(Cleared / Target x 100)` — the dashboard computes it.
    Reading a precomputed column also means a stale or hand-edited cell
    silently becomes the answer."""
    db_session.add(Advisor(wid=1, name="A", team="T", company="C", in_master_sheet=True))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD,
                               target=200, cleared=100, pct=0))     # pct deliberately wrong
    db_session.commit()
    assert aggregation.metric_value(db_session, "advisor", "A", "achievement_pct") == pytest.approx(50.0)


def test_a_zero_target_is_no_data_not_zero(org):
    """Adv Empty has target=0. NULLIF makes that "no achievement", which
    is honest — an advisor with no target has no attainment."""
    assert aggregation.metric_value(org, "advisor", "Adv Empty", "achievement_pct") is None


# =====================================================================
# D5 — one column, two labels
# =====================================================================

def test_overdue_has_exactly_one_metric():
    """`overdue` and `overdue_amount` both bound Pipeline.overdue — the
    single "Total Overdue" sheet column — under labels claiming a count
    and an amount. One of the two was always wrong."""
    assert "overdue_amount" not in METRICS
    overdue_bound = [k for k, m in METRICS.items()
                     if any(b.model is Pipeline and "overdue" in str(b.expr).lower()
                            for b in m.bindings.values())]
    assert sorted(overdue_bound) == ["overdue", "ytd_overdue"]


@pytest.mark.parametrize("phrase", [
    "overdue", "overdue count", "overdue amount", "overdue value", "amount overdue",
])
def test_every_overdue_phrase_resolves_to_the_one_metric(phrase):
    """The phrasings are kept — only the duplicate metric is gone, so no
    question stopped being understood."""
    assert resolve_metric(phrase) == "overdue", phrase


def test_no_two_metrics_share_a_binding_expression():
    """The general property. Two metrics over one column mean at least
    one label is a claim the data does not support."""
    seen: dict[str, str] = {}
    for key, metric in METRICS.items():
        for binding in metric.bindings.values():
            # The period is part of the identity: mtd_cleared and
            # ytd_cleared both read Performance.cleared, from different
            # period ROWS, and are genuinely different measures.
            signature = f"{binding.model.__name__}:{binding.expr}:{binding.period}"
            other = seen.get(signature)
            # One metric declaring the same binding at several LEVELS is
            # the intended pattern (see one_unit_ratio); two DIFFERENT
            # metrics sharing one is the defect.
            assert other in (None, key), f"{key} and {other} share {signature}"
            seen[signature] = key


# =====================================================================
# D6 — an invented target
# =====================================================================

def test_only_achievement_claims_a_target():
    """"the assigned target" and "the monthly goal" are real for
    achievement_pct and for nothing else. A 1-Unit ratio's denominator is
    team size; an attendance rate's is recorded days."""
    assert measures_target_attainment("achievement_pct")
    for key in METRICS:
        if key == "achievement_pct":
            continue
        assert not measures_target_attainment(key), key


@pytest.mark.parametrize("metric", [
    "one_unit_ratio", "attendance_rate", "login_rate",
    "meeting_conduction_rate", "connect_to_cr_rate",
])
def test_a_non_target_percentage_is_not_narrated_as_attainment(org, metric):
    from app.llm.narrative import explain_subject

    ir = QueryIR(intent="leaderboard", subject_level="team",
                 metric=MetricRef(key=metric), sort=Sort(metric=metric))
    sentence = explain_subject({"name": "Blue Area", "value": 60.0},
                               metric, "team", ir, rank=1, total=2)

    assert "assigned target" not in sentence, metric
    assert "monthly goal" not in sentence, metric
    assert "60%" in sentence


def test_achievement_keeps_its_target_wording(org):
    from app.llm.narrative import explain_subject

    ir = QueryIR(intent="leaderboard", subject_level="team",
                 metric=MetricRef(key="achievement_pct"),
                 sort=Sort(metric="achievement_pct"))
    sentence = explain_subject({"name": "Blue Area", "value": 80.0},
                               "achievement_pct", "team", ir, rank=1, total=2)

    assert "assigned target" in sentence
    assert "short of the monthly goal" in sentence


def test_a_count_metric_is_unchanged(org):
    from app.llm.narrative import explain_subject

    ir = QueryIR(intent="leaderboard", subject_level="advisor",
                 metric=MetricRef(key="total_connects"), sort=Sort(metric="total_connects"))
    sentence = explain_subject({"name": "Adv Full", "value": 100.0},
                               "total_connects", "advisor", ir, rank=1, total=3)
    assert "100" in sentence
    assert "target" not in sentence


# =====================================================================
# Ranking polarity — verified correct by the audit, pinned here
# =====================================================================

@pytest.mark.parametrize("query_metric,direction,expect_first", [
    ("mtd_cleared", "desc", 900.0),
    ("mtd_cleared", "asc", 0.0),
])
def test_ranking_order_is_respected(org, query_metric, direction, expect_first):
    ir = QueryIR(intent="leaderboard", subject_level="advisor",
                 metric=MetricRef(key=query_metric),
                 sort=Sort(metric=query_metric, direction=direction))
    assert compile_and_run(org, ir)[0]["value"] == pytest.approx(expect_first)
