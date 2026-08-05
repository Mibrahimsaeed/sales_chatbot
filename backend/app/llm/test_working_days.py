"""Working-day calendar + the KPIs that divide by it.

The business rule is Monday-Saturday with no holiday list, and the three
rates it unblocks are the ones that shipped in metric_aliases.UNAVAILABLE
for want of it:

    CR %      = CR            / (teamSize x 2   x workingDays) x 100
    Connect % = AnsweredCalls / (teamSize x 10  x workingDays) x 100
    Meeting % = Meetings      / (teamSize x 0.6 x workingDays) x 100

Every date here is explicit. A test that reads the clock proves nothing
on a Sunday.
"""

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, SalesFunnel,
)
from app.database.session import Base
from app.llm import aggregation, entity_extractor, nlu_pipeline, working_days
from app.llm.metric_ontology import METRICS, daily_target_rate
from app.services.chat_service import handle_chat_message

# August 2026: the 1st is a Saturday, so the 2nd is a Sunday.
MON = date(2026, 8, 3)
SAT = date(2026, 8, 8)
SUN = date(2026, 8, 9)


# ---------------------------------------------------------------------
# The calendar
# ---------------------------------------------------------------------


@pytest.mark.parametrize("day,expected", [
    (MON, True), (date(2026, 8, 4), True), (date(2026, 8, 5), True),
    (date(2026, 8, 6), True), (date(2026, 8, 7), True), (SAT, True),
    (SUN, False),
])
def test_monday_through_saturday_are_working_days(day, expected):
    assert working_days.is_working_day(day) is expected


@pytest.mark.parametrize("start,end,expected", [
    (MON, MON, 1),                    # Monday only
    (SAT, SAT, 1),                    # Saturday only
    (MON, SUN, 6),                    # Mon-Sun spans one Sunday
    (SAT, SUN, 1),                    # Saturday + Sunday
    (MON, date(2026, 8, 16), 12),     # two full weeks, two Sundays
])
def test_working_days_in_range(start, end, expected):
    assert working_days.working_days_in_range(start, end) == expected


def test_a_sunday_alone_returns_the_floor_not_zero():
    """The spec's own `max(1, count)`. A denominator of zero is never the
    right answer to divide by, and this module does not get to invent a
    new period semantic for Sundays."""
    assert working_days.working_days_in_range(SUN, SUN) == 1


def test_an_inverted_range_returns_the_floor():
    assert working_days.working_days_in_range(SUN, MON) == 1


def test_month_to_date_counts_from_the_first(monkeypatch):
    # 1 Aug Sat (1), 2 Aug Sun (0), 3 Aug Mon (1) -> 2
    assert working_days.month_to_date(MON) == 2


def test_year_to_date_counts_from_january_first():
    # 1 Jan 2026 is a Thursday: Thu, Fri, Sat count; Sun 4th does not.
    assert working_days.year_to_date(date(2026, 1, 5)) == 4


def test_a_full_week_is_six_working_days():
    assert working_days.working_days_in_range(MON, date(2026, 8, 9)) == 6


@pytest.mark.parametrize("period,expected", [
    ("MTD", 2), ("DAILY", 1),
])
def test_for_period_uses_the_right_window(period, expected):
    assert working_days.for_period(period, MON) == expected


def test_for_period_ytd_spans_the_year():
    assert working_days.for_period("YTD", date(2026, 1, 5)) == 4


def test_for_period_accepts_the_enum_and_the_string():
    assert (working_days.for_period(PerformancePeriod.MTD, MON)
            == working_days.for_period("MTD", MON))


def test_an_unknown_period_falls_back_to_month_to_date():
    assert working_days.for_period("NONSENSE", MON) == working_days.month_to_date(MON)


# ---------------------------------------------------------------------
# One owner
# ---------------------------------------------------------------------


def test_the_working_day_rule_has_exactly_one_implementation():
    """A second copy of a one-line rule is how two layers come to
    disagree about the same day."""
    import pathlib

    owners = set()
    for path in pathlib.Path("app").rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        # `weekday()` is the only way to ask "is this a Sunday", so any
        # module calling it is deciding the working-day rule.
        if "weekday()" in path.read_text():
            owners.add(path.name)
    assert owners == {"working_days.py"}, owners


@pytest.mark.parametrize("key,rate", [
    ("cr_rate", 2.0), ("answered_calls_rate", 10.0), ("meeting_rate", 0.6),
])
def test_the_per_day_target_is_declared_on_the_metric(key, rate):
    """The spec's per-advisor-per-day figures, declared once each."""
    assert daily_target_rate(key) == rate


# ---------------------------------------------------------------------
# CR % ground truth
# ---------------------------------------------------------------------


def _org(n_advisors, cr_each, calls_each=0, meetings_each=0):
    from conftest import _ADVISOR_PROFILE_VIEW

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    for wid in range(1, n_advisors + 1):
        s.add(Advisor(wid=wid, name=f"Advisor {wid}", team="Blue Area",
                      company="Graana", in_master_sheet=True))
        s.add(SalesFunnel(wid=wid, mtd_cr=cr_each, mtd_new_meeting=meetings_each,
                          mtd_followup_meeting=0))
        s.add(Calls(wid=wid, answered_calls_mtd=calls_each))
        s.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                          target=100, cleared=50, pct=50))
    s.commit()
    with engine.begin() as conn:
        conn.exec_driver_sql(_ADVISOR_PROFILE_VIEW)
    entity_extractor._cache["loaded_at"] = 0
    return s


def _cr_pct(db, working_days_value, monkeypatch):
    """CR % with the working-day count pinned, so the assertion is about
    the FORMULA rather than about today's date."""
    monkeypatch.setattr(working_days, "for_period",
                        lambda period, today=None: working_days_value)
    return aggregation.metric_value(db, "team", "Blue Area", "cr_rate")


def test_the_spec_worked_example(monkeypatch):
    """CR=20, teamSize=10, workingDays=5 -> 20 / (10*2*5) * 100 = 20."""
    db = _org(n_advisors=10, cr_each=2)          # 10 x 2 = 20 CRs
    assert round(_cr_pct(db, 5, monkeypatch)) == 20


def test_cr_of_zero_is_zero(monkeypatch):
    db = _org(n_advisors=10, cr_each=0)
    assert round(_cr_pct(db, 5, monkeypatch)) == 0


def test_one_working_day_scales_the_denominator(monkeypatch):
    """CR=20, teamSize=10, workingDays=1 -> 20 / 20 * 100 = 100."""
    db = _org(n_advisors=10, cr_each=2)
    assert round(_cr_pct(db, 1, monkeypatch)) == 100


def test_half_the_target_is_half_the_percentage(monkeypatch):
    """CR=10, teamSize=10, workingDays=5 -> 10 / 100 * 100 = 10."""
    db = _org(n_advisors=10, cr_each=1)
    assert round(_cr_pct(db, 5, monkeypatch)) == 10


def test_an_empty_team_does_not_divide_by_zero(monkeypatch):
    """teamSize 0 means no rows, so the SUM denominator is NULL rather
    than 0 — the existing convention, rendered as "no data"."""
    db = _org(n_advisors=0, cr_each=0)
    assert _cr_pct(db, 5, monkeypatch) is None


def test_the_denominator_scales_with_team_size(monkeypatch):
    """Same CR per advisor, twice the team: the RATE is unchanged, which
    is what makes teams of different sizes comparable at all."""
    small = round(_cr_pct(_org(5, cr_each=2), 5, monkeypatch))
    big = round(_cr_pct(_org(10, cr_each=2), 5, monkeypatch))
    assert small == big == 20


def test_the_advisor_level_uses_a_team_size_of_one(monkeypatch):
    """CR=2, teamSize=1, workingDays=5 -> 2 / 10 * 100 = 20."""
    db = _org(n_advisors=3, cr_each=2)
    monkeypatch.setattr(working_days, "for_period", lambda period, today=None: 5)
    assert round(aggregation.metric_value(db, "advisor", "Advisor 1", "cr_rate")) == 20


def test_the_two_paths_agree_on_one_advisor(monkeypatch):
    """The aggregation engine and the advisor service must not disagree —
    the advisor path read the binding's NUMERATOR as if it were the
    value, giving 500 where the engine said 83.3."""
    from app.services import advisor_service

    db = _org(n_advisors=3, cr_each=2)
    monkeypatch.setattr(working_days, "for_period", lambda period, today=None: 5)
    assert (advisor_service.get_advisor_metric(db, 1, "cr_rate")
            == aggregation.metric_value(db, "advisor", "Advisor 1", "cr_rate"))


# ---------------------------------------------------------------------
# Connect % and Meeting % ride the same mechanism
# ---------------------------------------------------------------------


def test_connect_percent_uses_ten_per_day(monkeypatch):
    """AnsweredCalls=100, teamSize=10, workingDays=2 -> 100/(10*10*2)*100 = 50."""
    db = _org(n_advisors=10, cr_each=0, calls_each=10)
    monkeypatch.setattr(working_days, "for_period", lambda period, today=None: 2)
    assert round(aggregation.metric_value(db, "team", "Blue Area",
                                          "answered_calls_rate")) == 50


def test_meeting_percent_uses_point_six_per_day(monkeypatch):
    """Meetings=12, teamSize=10, workingDays=2 -> 12/(10*0.6*2)*100 = 100."""
    db = _org(n_advisors=10, cr_each=0, meetings_each=1.2)
    monkeypatch.setattr(working_days, "for_period", lambda period, today=None: 2)
    assert round(aggregation.metric_value(db, "team", "Blue Area",
                                          "meeting_rate")) == 100


@pytest.mark.parametrize("key", ["cr_rate", "answered_calls_rate", "meeting_rate"])
def test_each_rate_keeps_the_eighty_five_sixty_bands(key):
    bands = METRICS[key].thresholds
    assert bands.green == 85.0 and bands.yellow == 60.0
    assert bands.status(90) == "green"
    assert bands.status(70) == "yellow"
    assert bands.status(50) == "red"


# ---------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query,metric", [
    ("What is the CR% for Blue Area this month?", "cr_rate"),
    ("What is the client registration rate for Blue Area this month?", "cr_rate"),
    ("Show Blue Area's CR rate", "cr_rate"),
    ("What is Blue Area's connect %?", "answered_calls_rate"),
    ("What is Blue Area's meeting rate?", "meeting_rate"),
])
def test_a_natural_language_rate_question_reaches_the_rate(monkeypatch, query, metric):
    db = _org(n_advisors=10, cr_each=2, calls_each=10, meetings_each=1)
    monkeypatch.setattr(working_days, "for_period", lambda period, today=None: 5)
    resolution = nlu_pipeline.resolve(query, db, session_id=None)

    assert resolution.kind != "clarify", f"{query!r} still refuses"
    found = (resolution.ir.metric.key if resolution.kind == "ir" and resolution.ir.metric
             else resolution.plan.metric)
    assert found == metric


def test_the_end_to_end_number_matches_the_formula(monkeypatch):
    """CR=20 over teamSize 10 and 5 working days is 20% — asserted on the
    rendered reply, not just on the metric key."""
    db = _org(n_advisors=10, cr_each=2)
    monkeypatch.setattr(working_days, "for_period", lambda period, today=None: 5)
    response = handle_chat_message(db, "What is the CR% for Blue Area this month?",
                                   session_id=None)
    assert "20" in response["reply"]
    assert response["type"] == "metric_value"


def test_a_comparison_on_cr_percent_computes_both_sides(monkeypatch):
    db = _org(n_advisors=10, cr_each=2)
    # Downtown: 5 advisors, 1 CR each -> 5 / (5*2*5) * 100 = 10%
    for wid in range(11, 16):
        db.add(Advisor(wid=wid, name=f"Advisor {wid}", team="Downtown",
                       company="Graana", in_master_sheet=True))
        db.add(SalesFunnel(wid=wid, mtd_cr=1, mtd_new_meeting=0,
                           mtd_followup_meeting=0))
        db.add(Calls(wid=wid, answered_calls_mtd=0))
    db.commit()
    entity_extractor._cache["loaded_at"] = 0
    monkeypatch.setattr(working_days, "for_period", lambda period, today=None: 5)

    assert round(aggregation.metric_value(db, "team", "Blue Area", "cr_rate")) == 20
    assert round(aggregation.metric_value(db, "team", "Downtown", "cr_rate")) == 10
