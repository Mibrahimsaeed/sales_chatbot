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

Part 12 (semantic retrieval expansion) added a semantic widening step,
tried only after BOTH pattern lists above find nothing at all — so it can
never override an "unsupported" verdict (checked first, returned
immediately) or an "equivalent" one. The exemplar corpus is deliberately
narrow and steers clear of anything resembling a DISCRETE prior period
("last quarter", "previous month") — those are exactly the unsupported
cases above, and embeddings alone can't reliably tell "the past 3 months"
(a rolling window, genuinely equivalent to 3M) apart from "last quarter"
(a specific prior calendar quarter, genuinely unsupported) since they're
semantically close. Getting a period wrong is a silently-wrong-answer bug,
not a missing-answer one, so this uses a higher floor than entity linking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.llm import entity_linker

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


# Deliberately narrow and unambiguous — see the module docstring for why
# "last quarter"-shaped phrasings must never appear here.
_TEMPORAL_EXEMPLARS: list[tuple[str, str]] = [
    ("since the start of the month", "MTD"), ("from the beginning of the month", "MTD"),
    ("so far this month", "MTD"), ("current month so far", "MTD"), ("month so far", "MTD"),
    ("so far this year", "YTD"), ("since january", "YTD"), ("year so far", "YTD"),
    ("annual to date", "YTD"), ("this year so far", "YTD"),
    ("past three months", "3M"), ("past 3 months", "3M"), ("last 3 months", "3M"),
    ("rolling three months", "3M"), ("trailing three months", "3M"), ("last three months", "3M"),
]

entity_linker.register_exemplar_type("temporal", lambda: _TEMPORAL_EXEMPLARS)

# Cheap pre-filter: only pay for an embedding call when the text has SOME
# time-adjacent vocabulary at all — the vast majority of messages ("top 5
# by revenue") have none, and must never pay this cost.
_TEMPORAL_HINT_RE = re.compile(
    r"\b(month|year|quarter|week|day|since|so far|to date|period|recent|annual|ytd|mtd)\b", re.I
)
_SEMANTIC_TEMPORAL_FLOOR = 0.85


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

    # Part 12: nothing matched deterministically — try semantic retrieval
    # against the narrow exemplar corpus above before giving up. Only
    # reached when neither pattern list matched anything, so it can never
    # override either verdict.
    if _TEMPORAL_HINT_RE.search(q):
        semantic = entity_linker.semantic_classify(q, "temporal", top_k=1, floor=_SEMANTIC_TEMPORAL_FLOOR)
        if semantic:
            return TemporalMatch(kind="equivalent", period=semantic[0]["value"], confidence=semantic[0]["score"])

    return None
