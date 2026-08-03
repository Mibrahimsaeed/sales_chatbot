"""
Join integrity audit (Area 4 of the reliability work).

Every user-facing number in this system comes from joining `advisors` to a
fact table on `wid`. When that join misses, nothing errors — the advisor
simply contributes nothing to the metric, and the chatbot reports a
confidently wrong total. The live audit found 45 master-sheet advisors with
no row in sales_funnel, performance, OR attendance: invisible in every
leaderboard, silently absent from every rollup.

This module reports those instead of letting them disappear:

  - per fact table: which master-sheet advisors have no row
  - fully unmatched: advisors with no row in ANY fact table
  - reverse direction: fact rows pointing at a non-master-sheet advisor,
    which is the population `in_master_sheet` deliberately filters out of
    user-facing queries (expected, but worth counting so a sudden jump is
    visible)

Read-only and side-effect free. `audit_joins()` is called once per sync
(recorded on SyncLog) and on demand by the health endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import (
    Advisor, Attendance, Bookings, Calls, Performance, Pipeline,
    Portfolio, SalesFunnel,
)

# The fact tables that carry the metrics the chatbot actually reports.
# Ordered most- to least- important for the "which advisors are invisible"
# question, since the sample lists below are truncated.
FACT_TABLES = (
    ("sales_funnel", SalesFunnel),
    ("performance", Performance),
    ("attendance", Attendance),
    ("pipeline", Pipeline),
    ("portfolio", Portfolio),
    ("bookings", Bookings),
    ("calls", Calls),
)

# Tables whose absence makes an advisor effectively invisible in the
# chatbot's core answers (revenue / activity / attendance questions).
_CORE_TABLES = ("sales_funnel", "performance", "attendance")

_SAMPLE_LIMIT = 10


@dataclass
class TableJoinResult:
    table: str
    matched: int
    unmatched: int
    unmatched_sample: list[dict] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        total = self.matched + self.unmatched
        return round(self.matched / total, 4) if total else 1.0


@dataclass
class JoinIntegrityReport:
    advisors_total: int = 0
    per_table: list[TableJoinResult] = field(default_factory=list)
    fully_unmatched_count: int = 0
    fully_unmatched_sample: list[dict] = field(default_factory=list)
    ghost_fact_rows: dict = field(default_factory=dict)

    @property
    def worst_match_rate(self) -> float:
        return min((r.match_rate for r in self.per_table), default=1.0)

    def to_dict(self) -> dict:
        return {
            "advisors_total": self.advisors_total,
            "fully_unmatched_count": self.fully_unmatched_count,
            "fully_unmatched_sample": self.fully_unmatched_sample,
            "worst_match_rate": self.worst_match_rate,
            "per_table": [
                {**asdict(r), "match_rate": r.match_rate} for r in self.per_table
            ],
            "ghost_fact_rows": self.ghost_fact_rows,
        }


def audit_joins(db: Session) -> JoinIntegrityReport:
    """Full advisors<->facts join audit, scoped to master-sheet advisors
    (the population every user-facing query already filters to — auditing
    the raw-only ghosts would report thousands of expected misses and
    drown the real signal)."""
    report = JoinIntegrityReport()

    master_advisors = (
        db.query(Advisor.wid, Advisor.name, Advisor.team)
        .filter(Advisor.in_master_sheet.is_(True))
        .all()
    )
    report.advisors_total = len(master_advisors)
    if not master_advisors:
        return report

    master_wids = {a.wid for a in master_advisors}
    by_wid = {a.wid: a for a in master_advisors}

    present_by_table: dict[str, set[int]] = {}
    for label, model in FACT_TABLES:
        present = {w for (w,) in db.query(model.wid).distinct()}
        present_by_table[label] = present

        matched_wids = master_wids & present
        unmatched_wids = master_wids - present
        report.per_table.append(TableJoinResult(
            table=label,
            matched=len(matched_wids),
            unmatched=len(unmatched_wids),
            unmatched_sample=[
                {"wid": w, "name": by_wid[w].name, "team": by_wid[w].team}
                for w in sorted(unmatched_wids)[:_SAMPLE_LIMIT]
            ],
        ))

        # reverse direction: rows keyed to a wid that is NOT a master-sheet
        # advisor. Expected (that's what in_master_sheet filters), but a
        # sudden change here means the source sheets shifted.
        report.ghost_fact_rows[label] = len(present - master_wids)

    core_present: set[int] = set()
    for label in _CORE_TABLES:
        core_present |= present_by_table.get(label, set())
    fully_unmatched = sorted(master_wids - core_present)
    report.fully_unmatched_count = len(fully_unmatched)
    report.fully_unmatched_sample = [
        {"wid": w, "name": by_wid[w].name, "team": by_wid[w].team}
        for w in fully_unmatched[:_SAMPLE_LIMIT]
    ]

    return report
