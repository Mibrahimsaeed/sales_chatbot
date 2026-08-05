"""THE working-day calendar.

Several KPIs in the dashboard spec are a count measured against a
working-day target rather than against a stored target:

    CR %       = CR            / (teamSize x 2   x workingDays) x 100
    Connect %  = AnsweredCalls / (teamSize x 10  x workingDays) x 100
    Meeting %  = Meetings      / (teamSize x 0.6 x workingDays) x 100

Every part of that was already present except one. `teamSize` is
aggregation.headcount(); the per-advisor-per-day figure is
MetricDef.daily_target_rate (declared 10.0 for calls and 0.6 for
meetings since Phase 5); the numerators are ordinary funnel columns.
Only `workingDays` had no source, which is why the three rates shipped in
metric_aliases.UNAVAILABLE — declared, refused with a written reason, and
answered with the underlying COUNT instead.

This module is that source, and the only one. It is deliberately a pure
calendar function rather than a table: the business rule is Monday
through Saturday with no holiday list, so a stored calendar would be a
second copy of a rule that fits in one line.

    Monday .. Saturday  -> working day
    Sunday              -> not a working day

Nothing here reads the database, the request, or the clock unless the
caller passes `today`, which keeps it testable without freezing time
globally and keeps the same query reproducible within a day.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

# date.weekday(): Monday is 0 ... Sunday is 6.
_SUNDAY = 6

# The floor the spec states: "return max(1, count)". A window containing
# no working day still divides by one rather than by zero, so a
# single-Sunday range yields the day's own figure instead of NULL. Stated
# here once so no caller re-invents it.
_MINIMUM = 1


def is_working_day(day: date) -> bool:
    """Monday-Saturday. No holiday calendar exists in the business rule."""
    return day.weekday() != _SUNDAY


def working_days_in_range(start: date, end: date) -> int:
    """Working days from `start` through `end`, both inclusive.

    Counted rather than approximated by weeks: a 7-day window contains
    exactly one Sunday, but an arbitrary range does not, and the ranges
    this is asked for (month-to-date, year-to-date) are arbitrary by
    construction. Ranges are short enough — a year at most — that
    iterating is not worth optimising away.

    An inverted range (end before start) yields the floor rather than a
    negative count: it means the caller's window is empty, and a
    denominator of zero or less is never the right answer to divide by.
    """
    if end < start:
        return _MINIMUM
    total = sum(
        1 for offset in range((end - start).days + 1)
        if is_working_day(start + timedelta(days=offset))
    )
    return max(_MINIMUM, total)


def month_to_date(today: Optional[date] = None) -> int:
    """First of the current month through today, inclusive."""
    today = today or date.today()
    return working_days_in_range(today.replace(day=1), today)


def year_to_date(today: Optional[date] = None) -> int:
    """January 1st of the current year through today, inclusive."""
    today = today or date.today()
    return working_days_in_range(today.replace(month=1, day=1), today)


def trailing_three_months(today: Optional[date] = None) -> int:
    """The 3M window: roughly 92 days back through today, inclusive.

    The spec defines MTD and YTD and is silent on 3M, which this system
    supports as a period. Ninety-two days is used rather than "three
    calendar months back" because 3M here is a rolling window, not a
    calendar-aligned one, and the arithmetic must not depend on which
    month it is asked in.
    """
    today = today or date.today()
    return working_days_in_range(today - timedelta(days=92), today)


def for_period(period, today: Optional[date] = None) -> int:
    """Working days in `period`, for the periods this system models.

    `period` is either a PerformancePeriod or its string value, so both
    the ontology (which stores the enum) and the IR (which stores the
    string) can ask without converting first.

    DAILY is one day, and a Sunday still yields 1 via the floor above.
    That is the spec's own `max(1, count)` rather than a new rule: the
    alternative — refusing a daily rate on Sundays — would be a period
    semantic invented here, and this module does not own period meaning.
    """
    name = str(getattr(period, "value", period) or "").upper()
    if name == "YTD":
        return year_to_date(today)
    if name == "3M":
        return trailing_three_months(today)
    if name == "DAILY":
        today = today or date.today()
        return working_days_in_range(today, today)
    # MTD is the system's default period, and the right default here too:
    # an unrecognised period is far more likely to be a month-shaped
    # window than a year-shaped one, and over-counting working days
    # understates a rate rather than overstating it.
    return month_to_date(today)
