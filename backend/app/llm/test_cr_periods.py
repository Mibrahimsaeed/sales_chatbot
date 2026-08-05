"""CR count and CR % across DAILY / MTD / YTD.

Two things are pinned here, and the second is the one that produced a
wrong answer rather than a missing one:

  1. The period the user asked for is the period BOTH readings use. A
     YTD count reported beside an MTD rate is worse than either alone —
     it is two different questions answered in one sentence, both
     labelled correctly.

  2. CR rate is `CR / (teamSize x 2 x workingDays)`. Connect->CR is
     `CR / AnsweredCalls`. They share a name in conversation and nothing
     else.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, SalesFunnel,
)
from app.database.session import Base
from app.llm import aggregation, entity_extractor, nlu_pipeline, working_days
from app.llm.metric_ontology import METRICS, metric_for_period
from app.llm.query_compiler import effective_metric
from app.services.chat_service import handle_chat_message

# Deliberately different per period, so a period mix-up shows up as a
# wrong NUMBER rather than only as a wrong label.
CR_PER_ADVISOR_MTD = 2
CR_PER_ADVISOR_YTD = 20
TEAM_SIZE = 10
MTD_CR = CR_PER_ADVISOR_MTD * TEAM_SIZE     # 20
YTD_CR = CR_PER_ADVISOR_YTD * TEAM_SIZE     # 200


@pytest.fixture()
def org(monkeypatch):
    from conftest import _ADVISOR_PROFILE_VIEW
    from app.core.config import settings
    from app.llm import llm_client, semantic_parser

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    for wid in range(1, TEAM_SIZE + 1):
        s.add(Advisor(wid=wid, name=f"Advisor {wid}", team="Blue Area",
                      company="Graana", in_master_sheet=True))
        s.add(SalesFunnel(wid=wid, mtd_cr=CR_PER_ADVISOR_MTD, ytd_cr=CR_PER_ADVISOR_YTD,
                          mtd_new_meeting=1, mtd_followup_meeting=0))
        s.add(Calls(wid=wid, answered_calls_mtd=10))
        s.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                          target=100, cleared=50, pct=50))
    s.commit()
    with engine.begin() as conn:
        conn.exec_driver_sql(_ADVISOR_PROFILE_VIEW)

    monkeypatch.setattr(settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(settings, "use_llm_planner", False, raising=False)
    monkeypatch.setattr(llm_client, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    # Pinned so every assertion is about the FORMULA, not about today.
    # Reads the enum the way the real for_period does — `str()` on a
    # PerformancePeriod gives "PerformancePeriod.YTD", not "YTD".
    def _fixed_working_days(period, today=None):
        name = str(getattr(period, "value", period) or "").upper()
        return {"YTD": 200, "DAILY": 1}.get(name, 5)

    monkeypatch.setattr(working_days, "for_period", _fixed_working_days)
    entity_extractor._cache["loaded_at"] = 0
    yield s
    s.close()


def _period_of(query, org):
    resolution = nlu_pipeline.resolve(query, org, session_id=None)
    if resolution.kind == "ir" and resolution.ir.time_range:
        return resolution.ir.time_range.period
    return None


def _metric_of(query, org):
    resolution = nlu_pipeline.resolve(query, org, session_id=None)
    if resolution.kind == "ir":
        return effective_metric(resolution.ir)
    plan = getattr(resolution, "plan", None)
    return plan.metric if plan else None


# ---------------------------------------------------------------------
# The period reaches the IR unchanged
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query,period", [
    ("Blue Area CR today", "DAILY"),
    ("Blue Area CR this month", "MTD"),
    ("Blue Area CR year to date", "YTD"),
    ("What is Blue Area's CR today?", "DAILY"),
    ("What is Blue Area's CR this month?", "MTD"),
    ("What is Blue Area's CR year to date?", "YTD"),
    ("How many client registrations does Blue Area have today?", "DAILY"),
    ("How many client registrations does Blue Area have this month?", "MTD"),
    ("How many client registrations does Blue Area have YTD?", "YTD"),
])
def test_the_requested_period_survives_to_the_ir(org, query, period):
    """DAILY must never silently become MTD."""
    assert _period_of(query, org) == period


# ---------------------------------------------------------------------
# MTD and YTD: count and rate, both correct, both the same period
# ---------------------------------------------------------------------


def test_mtd_count_and_rate(org):
    assert aggregation.metric_value(org, "team", "Blue Area",
                                    "client_registrations") == MTD_CR
    # 20 / (10 x 2 x 5) x 100 = 20
    assert round(aggregation.metric_value(org, "team", "Blue Area", "cr_rate")) == 20


def test_ytd_count_and_rate(org):
    assert aggregation.metric_value(org, "team", "Blue Area",
                                    "ytd_client_registrations") == YTD_CR
    # 200 / (10 x 2 x 200) x 100 = 5
    assert round(aggregation.metric_value(org, "team", "Blue Area", "ytd_cr_rate")) == 5


def test_ytd_cr_rate_exists_and_is_reachable(org):
    """It refused for want of a period FAMILY: cr_rate had none, so
    _effective_metric could find no YTD member while ytd_cr sat in the
    table holding the data."""
    assert metric_for_period("cr_rate", "YTD") == "ytd_cr_rate"
    assert _metric_of("Blue Area CR% year to date", org) == "ytd_cr_rate"


def test_the_ytd_rate_uses_the_years_working_days(org):
    """Not the month's. Same numerator, a denominator 40x larger."""
    mtd = aggregation.metric_value(org, "team", "Blue Area", "cr_rate")
    ytd = aggregation.metric_value(org, "team", "Blue Area", "ytd_cr_rate")
    # MTD: 20/(10*2*5)=20% ; YTD: 200/(10*2*200)=5%
    assert round(mtd) == 20 and round(ytd) == 5


# ---------------------------------------------------------------------
# DAILY: no data, and an honest refusal rather than a substitution
# ---------------------------------------------------------------------


def test_daily_cr_is_refused_not_silently_answered_as_mtd(org):
    """There is no daily CR column — only mtd_cr and ytd_cr. The refusal
    is the correct answer; substituting MTD would be a wrong one wearing
    the right label."""
    response = handle_chat_message(org, "Blue Area CR today", session_id=None)

    assert "daily" in response["reply"].lower()
    assert str(MTD_CR) not in response["reply"]


def test_daily_has_no_cr_sibling(org):
    assert metric_for_period("client_registrations", "DAILY") is None
    assert metric_for_period("cr_rate", "DAILY") is None


# ---------------------------------------------------------------------
# Count and rate together, always in the same period
# ---------------------------------------------------------------------


def test_a_count_question_also_reports_the_rate(org):
    response = handle_chat_message(
        org, "How many client registrations does Blue Area have this month?",
        session_id=None)
    assert str(MTD_CR) in response["reply"]
    assert "20%" in response["reply"] or "20.0%" in response["reply"]


def test_a_rate_question_also_reports_the_count(org):
    response = handle_chat_message(org, "Blue Area CR% this month", session_id=None)
    assert "20%" in response["reply"] or "20.0%" in response["reply"]
    assert str(MTD_CR) in response["reply"]


def test_a_ytd_count_never_reports_an_mtd_rate(org):
    """THE failure this pairing could introduce: two periods in one
    sentence, both labelled correctly, describing different questions."""
    response = handle_chat_message(
        org, "How many client registrations does Blue Area have YTD?",
        session_id=None)

    assert str(YTD_CR) in response["reply"]
    assert "5%" in response["reply"] or "5.0%" in response["reply"]
    assert "(MTD)" not in response["reply"]


def test_a_ytd_rate_never_reports_an_mtd_count(org):
    response = handle_chat_message(org, "Blue Area CR% year to date", session_id=None)

    assert str(YTD_CR) in response["reply"]
    assert "(MTD)" not in response["reply"]
    assert str(MTD_CR) not in response["reply"].replace(str(YTD_CR), "")


def test_the_companion_is_resolved_to_the_executed_period(org):
    """The ontology names the MTD member of each family; taking it
    verbatim is what produced the mismatch."""
    from app.llm.response_planner import plan_response

    resolution = nlu_pipeline.resolve("Blue Area CR% year to date", org,
                                      session_id=None)
    from app.llm.query_compiler import compile_and_run

    rows = compile_and_run(org, resolution.ir, offset=0)
    plan = plan_response(resolution.ir, rows)
    assert plan.companion_metric == "ytd_client_registrations"


def test_a_metric_with_no_companion_reports_one_value(org):
    response = handle_chat_message(org, "Blue Area revenue", session_id=None)
    assert "with" not in response["reply"]


def test_companion_pairings_are_symmetric():
    """A one-way pairing renders on one phrasing of a question and not on
    its mirror. Enforced at import too; asserted here so the reason is
    written down."""
    for key, metric in METRICS.items():
        if metric.companion:
            assert METRICS[metric.companion].companion == key


# ---------------------------------------------------------------------
# CR rate is not the Connect->CR funnel conversion
# ---------------------------------------------------------------------


def test_cr_rate_and_connect_to_cr_are_different_metrics(org):
    """Different denominators: a working-day target vs answered calls.
    CR=20, calls=100 -> Connect->CR = 20%. CR rate = 20/(10*2*5) = 20%.
    Deliberately equal here, so the test cannot pass by coincidence of
    value — it asserts the KEYS and the denominators differ."""
    assert _metric_of("Blue Area CR%", org) == "cr_rate"
    assert _metric_of("Blue Area connect to CR ratio", org) == "connect_to_cr_rate"

    assert METRICS["cr_rate"].daily_target_rate == 2.0
    assert METRICS["connect_to_cr_rate"].daily_target_rate is None
    assert METRICS["connect_to_cr_rate"].bindings["advisor"].ratio_denominator is not None


def test_the_funnel_conversion_is_untouched_by_working_days(org):
    """CR / AnsweredCalls = 20/100 = 20%, whatever the calendar says."""
    assert round(aggregation.metric_value(org, "team", "Blue Area",
                                          "connect_to_cr_rate")) == 20


# ---------------------------------------------------------------------
# Advisor level
# ---------------------------------------------------------------------


def test_the_advisor_level_uses_a_team_size_of_one(org):
    """CR=2, teamSize=1, workingDays=5 -> 2/(1*2*5)*100 = 20."""
    assert round(aggregation.metric_value(org, "advisor", "Advisor 1", "cr_rate")) == 20


def test_the_advisor_ytd_rate_uses_ytd_working_days(org):
    """CR=20, teamSize=1, workingDays=200 -> 20/400*100 = 5."""
    assert round(aggregation.metric_value(org, "advisor", "Advisor 1",
                                          "ytd_cr_rate")) == 5
