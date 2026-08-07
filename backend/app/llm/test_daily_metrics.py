"""Phase 12 — DAILY answered from real daily columns.

Until now DAILY was recognised VOCABULARY with no data behind it: every
measure returned None from metric_for_period("DAILY"), so a daily
question could only ever be refused. Two columns turned out to hold
genuine daily figures — `calls.answered_calls_daily` and
`calls.connects_daily`, both from the biometric sheet's "Answered Calls"
tab — and neither was bound to a metric.

WHAT MAKES THEM SAFE TO BIND, checked against the live source before any
of this was written:

  * containment — no row of 667 has daily > MTD, which a fabricated or
    mislabelled daily column would violate
  * magnitude   — daily/MTD ≈ 0.27, tracking working days elapsed
  * identity    — `calls.connects_mtd` and the SalesFunnel connects pair
    agree (36,823 vs 36,796; identical per advisor for 651 of 667), so
    the daily column is the same MEASURE's daily slice and the family
    compares like with like

The CCMC "Competition" tab was rejected by the same checks: its CR
exceeds MTD for 584 advisors while staying under YTD, making it a
contest-period cumulative rather than a day. Daily CR therefore has no
source and must stay unavailable — the last section here pins that,
because a daily question answered with the month is the exact failure
this whole line of work exists to prevent.

EVERY FIXTURE BELOW USES DIFFERENT DAILY AND MTD VALUES. A test that
asserted only the period label would pass while reading the wrong column;
these assert the NUMBER.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, SalesFunnel,
)
from app.database.session import Base
from app.llm import entity_extractor, nlu_pipeline, temporal_parser, working_days
from app.llm.metric_ontology import METRICS, metric_for_period, supported_periods
from app.llm.query_compiler import compile_and_run, effective_metric
from app.llm.query_ir import Filter, MetricRef, QueryIR, Sort, TimeRange

# wid: (answered_daily, answered_mtd, connects_daily, connects_mtd)
PEOPLE = {
    1: ("Zainab Riaz", 8, 200, 17, 500),
    2: ("Areeba Khan", 12, 300, 23, 700),
}


@pytest.fixture()
def org(db_session, monkeypatch):
    from app.core.config import settings
    from app.llm import llm_client, narrative, semantic_parser

    monkeypatch.setattr(settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(settings, "use_llm_planner", False, raising=False)
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False, raising=False)
    monkeypatch.setattr(llm_client, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)

    for wid, (name, ad, am, cd, cm) in PEOPLE.items():
        db_session.add(Advisor(wid=wid, name=name, team="Blue Area",
                               company="Graana", in_master_sheet=True))
        db_session.add(Calls(wid=wid, answered_calls_daily=ad, answered_calls_mtd=am,
                             connects_daily=cd, connects_mtd=cm))
        # CR has MTD and YTD only — no daily column exists anywhere.
        db_session.add(SalesFunnel(wid=wid, mtd_cr=7 * wid, ytd_cr=88 * wid,
                                   mtd_new_connect=cm, mtd_followup_connect=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=1000, cleared=400))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _ir(metric, period, level="advisor", team=None):
    return QueryIR(
        intent="leaderboard", subject_level=level,
        metric=MetricRef(key=metric), sort=Sort(metric=metric),
        time_range=TimeRange(period=period),
        filters=[Filter(field="team", operator="=", value=team)] if team else [],
    )


def _values(db, metric, period, level="advisor", team=None):
    return [r["value"] for r in compile_and_run(db, _ir(metric, period, level, team))]


# ---------------------------------------------------------------------
# Period resolution
# ---------------------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "answered calls today", "daily answered calls", "today's connects",
    "daily connects", "connects today",
])
def test_daily_phrasings_resolve_to_the_daily_window(phrase):
    match = temporal_parser.parse_period(phrase)
    assert match is not None and match.kind == "equivalent", phrase
    assert match.period == "DAILY", phrase


@pytest.mark.parametrize("phrase,expected", [
    ("answered calls this month", "MTD"), ("MTD answered calls", "MTD"),
    ("answered calls this year", "YTD"), ("YTD answered calls", "YTD"),
])
def test_the_other_windows_are_untouched(phrase, expected):
    assert temporal_parser.parse_period(phrase).period == expected


def test_no_stated_window_still_names_no_period():
    """The MTD default must stay a default, inferred from the metric."""
    assert temporal_parser.parse_period("Zainab's answered calls") is None


# ---------------------------------------------------------------------
# The ontology: DAILY is a real member of these families
# ---------------------------------------------------------------------


@pytest.mark.parametrize("base,daily_key,column", [
    ("answered_calls", "daily_answered_calls", "answered_calls_daily"),
    ("total_connects", "daily_connects", "connects_daily"),
    ("answered_calls_rate", "daily_answered_calls_rate", "answered_calls_daily"),
])
def test_the_daily_sibling_is_reachable_and_reads_the_daily_column(
        base, daily_key, column):
    assert metric_for_period(base, "DAILY") == daily_key
    assert PerformancePeriod.DAILY in supported_periods(base)
    binding = METRICS[daily_key].bindings["advisor"]
    assert column in str(binding.expr), f"{daily_key} does not read {column}"


def test_the_daily_key_resolves_back_to_its_monthly_sibling():
    """The swap works both ways, or a follow-up could not leave DAILY."""
    assert metric_for_period("daily_answered_calls", "MTD") == "answered_calls"
    assert metric_for_period("daily_connects", "MTD") == "total_connects"
    assert metric_for_period("daily_connects", "YTD") == "ytd_connects"


# ---------------------------------------------------------------------
# The values — different daily and MTD fixtures throughout
# ---------------------------------------------------------------------


def test_daily_answered_calls_come_from_the_daily_column(org):
    assert sorted(_values(org, "answered_calls", "DAILY")) == [8, 12]


def test_monthly_answered_calls_still_come_from_the_monthly_column(org):
    assert sorted(_values(org, "answered_calls", "MTD")) == [200, 300]


def test_daily_connects_come_from_the_daily_column(org):
    assert sorted(_values(org, "total_connects", "DAILY")) == [17, 23]


def test_monthly_connects_still_come_from_the_monthly_column(org):
    """Connects MTD reads SalesFunnel, daily reads Calls — verified
    against the live source to be the same measure. This pins that the
    monthly side did not move when the daily side was added."""
    assert sorted(_values(org, "total_connects", "MTD")) == [500, 700]


@pytest.mark.parametrize("metric", ["answered_calls", "total_connects"])
def test_a_daily_request_never_returns_the_monthly_number(org, metric):
    """THE invariant. Asserted on the VALUES, so it cannot pass by
    reading the right label off the wrong column."""
    daily = sorted(_values(org, metric, "DAILY"))
    monthly = sorted(_values(org, metric, "MTD"))
    assert daily != monthly
    assert not set(daily) & set(monthly)


# ---------------------------------------------------------------------
# The percentage — the existing working-day machinery, nothing new
# ---------------------------------------------------------------------


def test_a_daily_window_is_one_working_day():
    assert working_days.for_period("DAILY") == 1


def test_the_daily_rate_matches_the_hand_calculation(org):
    """answered_calls_daily / (teamSize x 10 x workingDays) x 100,
    with workingDays = 1:

        (8 + 12) / (2 x 10 x 1) x 100 = 100.0
    """
    values = _values(org, "answered_calls_rate", "DAILY", level="team", team="Blue Area")
    assert values == [pytest.approx((8 + 12) / (2 * 10 * 1) * 100)]
    assert values == [pytest.approx(100.0)]


def test_the_monthly_rate_uses_the_monthly_numerator_and_more_working_days(org):
    """The same formula with the month's numerator and its own working-day
    count — so the daily rate cannot be the monthly one relabelled."""
    monthly = _values(org, "answered_calls_rate", "MTD", level="team", team="Blue Area")
    expected = (200 + 300) / (2 * 10 * working_days.for_period("MTD")) * 100
    assert monthly == [pytest.approx(expected)]
    assert monthly != [pytest.approx(100.0)]


def test_the_daily_rate_is_computed_by_the_shared_engine_not_a_new_path(org):
    """`daily_answered_calls_rate` declares the same working_day_scaled
    binding and the same per-day target as its monthly sibling — the only
    differences are the column and the declared period."""
    daily, monthly = METRICS["daily_answered_calls_rate"], METRICS["answered_calls_rate"]
    assert daily.daily_target_rate == monthly.daily_target_rate == 10.0
    assert daily.rollup == monthly.rollup
    assert daily.bindings["advisor"].working_day_scaled


# ---------------------------------------------------------------------
# End to end, through the pipeline
# ---------------------------------------------------------------------


@pytest.mark.parametrize("text,expected_metric", [
    ("top advisors by answered calls today", "daily_answered_calls"),
    ("top advisors by daily answered calls", "daily_answered_calls"),
    ("top advisors by connects today", "daily_connects"),
    ("top advisors by daily connects", "daily_connects"),
])
def test_a_daily_question_compiles_against_the_daily_binding(org, text, expected_metric):
    resolution = nlu_pipeline.resolve(text, org, session_id=f"d-{text}")
    assert resolution.kind == "ir"
    assert resolution.ir.time_range.period == "DAILY"
    assert effective_metric(resolution.ir) == expected_metric


@pytest.mark.parametrize("text,expected_metric,period", [
    ("top advisors by answered calls this month", "answered_calls", "MTD"),
    ("top advisors by connects this month", "total_connects", "MTD"),
    ("top advisors by connects year to date", "ytd_connects", "YTD"),
])
def test_the_monthly_and_yearly_questions_are_unchanged(org, text, expected_metric, period):
    resolution = nlu_pipeline.resolve(text, org, session_id=f"m-{text}")
    assert resolution.ir.time_range.period == period
    assert effective_metric(resolution.ir) == expected_metric


# ---------------------------------------------------------------------
# Daily CR stays unavailable — no source exists
# ---------------------------------------------------------------------


@pytest.mark.parametrize("key", [
    "client_registrations", "ytd_client_registrations", "cr_rate", "ytd_cr_rate",
])
def test_no_cr_metric_gained_a_daily_sibling(key):
    """The Competition tab looked like a daily funnel and is not one: its
    CR exceeds MTD for 584 advisors. Binding it would have answered
    "today" with roughly seven times the month."""
    assert metric_for_period(key, "DAILY") is None
    assert PerformancePeriod.DAILY not in supported_periods(key)


def test_no_metric_reads_a_daily_cr_column():
    """There is no such column. If one is ever added, this fails and the
    daily CR metric can be written deliberately rather than by accident."""
    assert not hasattr(SalesFunnel, "daily_cr")


@pytest.mark.parametrize("metric", ["client_registrations", "conversion", "total_meetings"])
def test_a_daily_request_for_a_measure_without_daily_data_refuses(org, metric):
    ir = _ir(metric, "DAILY")
    assert effective_metric(ir) is None
    # None, not [] — the compiler's "cannot answer" signal, which
    # chat_service._dispatch_ir turns into the unavailable-period reply.
    # An empty list would mean "ran, found nobody", a different answer.
    assert compile_and_run(org, ir) is None


def test_a_daily_cr_question_never_returns_the_monthly_cr(org):
    """Values, not wording: the MTD CR figures are 7 and 14 in this
    fixture and neither may appear under a daily question."""
    from app.services.chat_service import handle_chat_message

    reply = handle_chat_message(org, "What is Zainab Riaz's CR today?",
                                session_id="cr-daily")["reply"]
    assert "7" not in reply and "14" not in reply
    assert "daily" in reply.lower()


# ---------------------------------------------------------------------
# Context: the period is part of the semantic state
# ---------------------------------------------------------------------


def test_daily_survives_a_subject_change_and_a_later_period_change(org):
    """Zainab today -> what about Areeba? -> what about MTD?

    Owned by conversation_context, which is unchanged — asserted here so
    the daily work cannot quietly regress it."""
    session = "ctx-daily"
    first = nlu_pipeline.resolve("top advisors by answered calls today",
                                 org, session_id=session)
    assert effective_metric(first.ir) == "daily_answered_calls"

    narrowed = nlu_pipeline.resolve("only Graana", org, session_id=session)
    assert narrowed.ir.time_range.period == "DAILY", "the window did not survive"
    assert effective_metric(narrowed.ir) == "daily_answered_calls"

    monthly = nlu_pipeline.resolve("what about this month?", org, session_id=session)
    assert monthly.ir.time_range.period == "MTD"
    assert effective_metric(monthly.ir) == "answered_calls"

    back = nlu_pipeline.resolve("and today?", org, session_id=session)
    assert back.ir.time_range.period == "DAILY"
    assert effective_metric(back.ir) == "daily_answered_calls"
