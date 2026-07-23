"""
Temporal parser (Part 8 of the NLU rework). Fixes a real silent-answer bug:
entity_extractor.PERIOD_KEYWORDS used to map bare "month" -> MTD and bare
"year" -> YTD, which meant "revenue last month" silently resolved to THIS
month's MTD number — a wrong answer presented as a right one.

Two outcomes only:
- "equivalent": the phrase genuinely means one of the periods the
  Performance table already stores (MTD/YTD/3M) — same behavior as before
  for these phrasings.
- "unsupported": the phrase names a real calendar window (last month,
  yesterday, this week, past N days, a date range) that the data model
  cannot answer correctly today. This must NEVER silently fall back to
  MTD — same honest-rejection style as ir_validator._UNSUPPORTED_INTENTS.

Scope boundary: only the Performance-table period enum (MTD/YTD/3M).
Attendance's "today" is a separate table/concept and untouched here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Period = Literal["MTD", "YTD", "3M"]


@dataclass
class TemporalMatch:
    kind: Literal["equivalent", "unsupported"]
    period: Period | None = None
    confidence: float = 1.0
    reason: str | None = None


# Longest phrase first so "this month" matches before a bare "month" could.
_EQUIVALENT_PATTERNS: list[tuple[str, Period]] = [
    (r"\bmonth to date\b", "MTD"),
    (r"\bthis month\b", "MTD"),
    (r"\bmtd\b", "MTD"),
    (r"\byear to date\b", "YTD"),
    (r"\bthis year\b", "YTD"),
    (r"\bytd\b", "YTD"),
    (r"\bthis quarter\b", "3M"),
    (r"\bthree month(s)?\b", "3M"),
    (r"\bquarter\b", "3M"),
    (r"\b3m\b", "3M"),
    (r"\b3[\s-]?month\b", "3M"),
]

_UNSUPPORTED_PATTERNS: list[tuple[str, str]] = [
    (r"\blast month\b|\bprevious month\b",
     "I only have month-to-date totals for the current month, not a separate 'last month' figure yet"),
    (r"\byesterday\b",
     "I don't have day-by-day historical data yet — only the current month-to-date totals"),
    (r"\bthis week\b|\blast week\b",
     "I don't have week-level data yet — only month-to-date, year-to-date, and 3-month totals"),
    (r"\bpast\s+\d+\s+days?\b|\blast\s+\d+\s+days?\b",
     "I don't have a rolling day-window view yet — only month-to-date, year-to-date, and 3-month totals"),
    (r"\bbetween\s+.+\s+and\s+.+\b(?=.*\d)",
     "I can't compare arbitrary custom date ranges yet — only month-to-date, year-to-date, and 3-month totals"),
    (r"\blast quarter\b|\bprevious quarter\b",
     "I only have the current 3-month total, not a separate 'last quarter' figure yet"),
]


def parse_period(text: str) -> TemporalMatch | None:
    """Returns a TemporalMatch if the text names a time window at all,
    else None (no temporal expression found — caller's existing behavior
    is unaffected)."""
    q = text.lower()

    # unsupported checked first: "last month" must never be caught by a
    # bare "month" style equivalent pattern.
    for pattern, reason in _UNSUPPORTED_PATTERNS:
        if re.search(pattern, q):
            return TemporalMatch(kind="unsupported", reason=reason)

    for pattern, period in _EQUIVALENT_PATTERNS:
        if re.search(pattern, q):
            return TemporalMatch(kind="equivalent", period=period, confidence=1.0)

    return None
