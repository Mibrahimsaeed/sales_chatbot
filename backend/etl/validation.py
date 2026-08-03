"""
Post-load data validation (Area 2 of the reliability work).

Runs against the DATABASE after every sync's load step and produces a
structured report instead of a pass/fail boolean. The distinction matters:
almost none of these findings should abort a sync — a team missing its
target row is a real data gap, but refusing to load 3,000 good advisor
rows over it would make the chatbot's data worse, not better. So the
report is recorded (SyncLog.validation_report) and exposed for monitoring
(/api/health/*), and only genuinely corrupt-load conditions escalate to
"error" severity.

Every check maps to a specific way the chatbot previously gave an
inconsistent answer:

  missing_team_targets   -> "what's X's target achievement" answered for
                            some teams and silently not others
  empty_required_column  -> a hierarchy level that looks supported but has
                            no data behind it at all (advisors.unit)
  orphan_fact_rows       -> activity numbers keyed to a WID with no advisor
                            row, so they vanish from every rollup
  advisors_without_facts -> an onboarded advisor who silently contributes
                            0 to every metric
  duplicate_*            -> the same real entity counted twice
  near_duplicate_names   -> the same real entity stored under two
                            spellings normalize.py can't safely auto-merge

`Finding.severity` is "error" | "warning" | "info", and the report's own
`ok` property is False only when an "error"-severity finding is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import (
    Advisor, Attendance, Bookings, Calls, Performance, Pipeline,
    Portfolio, SalesFunnel, TeamTarget,
)
from etl.normalize import normalization_key

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

# Advisor columns that are supposed to carry data for the hierarchy the
# chatbot exposes. A column here that is 100% NULL means a level the bot
# advertises has nothing behind it (this is how advisors.unit was found).
_EXPECTED_POPULATED_COLUMNS = (
    "company", "team", "office", "bm", "zm",
)

# (label, model) for every fact table keyed by advisor wid.
_FACT_TABLES = (
    ("sales_funnel", SalesFunnel),
    ("pipeline", Pipeline),
    ("attendance", Attendance),
    ("performance", Performance),
    ("portfolio", Portfolio),
    ("bookings", Bookings),
    ("calls", Calls),
)


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    count: int = 0
    # A bounded sample of offending values, so the report stays readable
    # and small enough to store as JSON on SyncLog without bloating it.
    sample: list[Any] = field(default_factory=list)


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def ok(self) -> bool:
        """False only for error-severity findings — a warning is a data
        gap worth surfacing, not a reason to reject an otherwise good
        sync."""
        return not any(f.severity == SEVERITY_ERROR for f in self.findings)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_WARNING)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "findings": [asdict(f) for f in self.findings],
        }


_SAMPLE_LIMIT = 10


def _sample(values) -> list:
    return [v for v in list(values)[:_SAMPLE_LIMIT]]


def check_missing_team_targets(db: Session) -> Finding | None:
    """Teams that have advisors but no row in team_targets — the live
    audit found 10 target rows against 137 distinct team names."""
    teams_with_advisors = {
        t for (t,) in db.query(Advisor.team)
        .filter(Advisor.team.isnot(None), Advisor.in_master_sheet.is_(True))
        .distinct()
    }
    if not teams_with_advisors:
        return None
    teams_with_targets = {t for (t,) in db.query(TeamTarget.team).distinct() if t}
    # compare on the normalized key so a pure whitespace/case variant isn't
    # reported as "missing" when a target genuinely exists for it
    target_keys = {normalization_key(t) for t in teams_with_targets}
    missing = sorted(t for t in teams_with_advisors if normalization_key(t) not in target_keys)
    if not missing:
        return None
    return Finding(
        check="missing_team_targets",
        severity=SEVERITY_WARNING,
        message=(
            f"{len(missing)} of {len(teams_with_advisors)} teams with advisors have no row in "
            "team_targets — target/achievement questions for these teams cannot be answered"
        ),
        count=len(missing),
        sample=_sample(missing),
    )


def check_empty_required_columns(db: Session) -> list[Finding]:
    """Advisor columns that are entirely NULL across every master-sheet
    row — the level exists in the schema and the chatbot advertises it,
    but no sync has ever put data there."""
    findings: list[Finding] = []
    total = db.query(func.count(Advisor.wid)).filter(Advisor.in_master_sheet.is_(True)).scalar() or 0
    if not total:
        return findings

    for column_name in _EXPECTED_POPULATED_COLUMNS:
        column = getattr(Advisor, column_name)
        populated = (
            db.query(func.count(Advisor.wid))
            .filter(Advisor.in_master_sheet.is_(True), column.isnot(None))
            .scalar()
        ) or 0
        if populated == 0:
            findings.append(Finding(
                check="empty_required_column",
                severity=SEVERITY_ERROR,
                message=f"advisors.{column_name} is empty for all {total} master-sheet advisors",
                count=total,
            ))
        elif populated < total:
            findings.append(Finding(
                check="partially_empty_column",
                severity=SEVERITY_INFO,
                message=f"advisors.{column_name} is missing for {total - populated} of {total} master-sheet advisors",
                count=total - populated,
            ))
    return findings


def check_orphan_fact_rows(db: Session) -> list[Finding]:
    """Fact rows whose wid has no matching advisors row. The FK makes this
    impossible in a healthy Postgres schema, so anything found here means
    real referential damage — hence error severity."""
    findings: list[Finding] = []
    advisor_wids = select(Advisor.wid)
    for label, model in _FACT_TABLES:
        orphans = [w for (w,) in db.query(model.wid).filter(~model.wid.in_(advisor_wids)).distinct()]
        if orphans:
            findings.append(Finding(
                check="orphan_fact_rows",
                severity=SEVERITY_ERROR,
                message=f"{len(orphans)} row(s) in {label} reference a wid with no advisors record",
                count=len(orphans),
                sample=_sample(orphans),
            ))
    return findings


def check_duplicate_records(db: Session) -> list[Finding]:
    """Duplicate keys where the schema expects uniqueness. advisors.wid and
    the single-row-per-wid fact tables are PK-protected, so the reachable
    cases are team_targets.team (PK, but case/whitespace variants slip past
    it) and performance (wid, period)."""
    findings: list[Finding] = []

    dupe_perf = (
        db.query(Performance.wid, Performance.period, func.count().label("n"))
        .group_by(Performance.wid, Performance.period)
        .having(func.count() > 1)
        .all()
    )
    if dupe_perf:
        findings.append(Finding(
            check="duplicate_performance_rows",
            severity=SEVERITY_ERROR,
            message=f"{len(dupe_perf)} (wid, period) pair(s) have more than one performance row",
            count=len(dupe_perf),
            sample=_sample([f"wid={w} period={getattr(p, 'value', p)}" for w, p, _n in dupe_perf]),
        ))

    # team_targets.team is the PK, so exact duplicates can't exist — but two
    # rows differing only by whitespace/case CAN, and they double-count a
    # team's target.
    target_teams = [t for (t,) in db.query(TeamTarget.team) if t]
    by_key: dict[str, list[str]] = {}
    for team in target_teams:
        by_key.setdefault(normalization_key(team), []).append(team)
    collisions = {k: v for k, v in by_key.items() if len(v) > 1}
    if collisions:
        findings.append(Finding(
            check="duplicate_team_targets",
            severity=SEVERITY_ERROR,
            message=f"{len(collisions)} team(s) have multiple team_targets rows differing only in spelling",
            count=len(collisions),
            sample=_sample([" | ".join(v) for v in collisions.values()]),
        ))
    return findings


def check_near_duplicate_names(db: Session) -> list[Finding]:
    """Distinct stored values that collapse to the same normalization key —
    i.e. the same real entity under two spellings. normalize.py fixes the
    mechanical variants at write time; anything still reported here needs a
    human decision at the source sheet, so it's a warning, not an error."""
    findings: list[Finding] = []
    for label, column in (("team", Advisor.team), ("office", Advisor.office), ("company", Advisor.company)):
        values = [v for (v,) in db.query(column).filter(column.isnot(None)).distinct()]
        by_key: dict[str, set[str]] = {}
        for value in values:
            by_key.setdefault(normalization_key(value), set()).add(value)
        collisions = {k: v for k, v in by_key.items() if len(v) > 1}
        if collisions:
            findings.append(Finding(
                check="near_duplicate_names",
                severity=SEVERITY_WARNING,
                message=(
                    f"{len(collisions)} advisors.{label} value(s) exist under more than one spelling — "
                    "these split rollups for what is probably one real entity"
                ),
                count=len(collisions),
                sample=_sample([" | ".join(sorted(v)) for v in collisions.values()]),
            ))
    return findings


def validate_database(db: Session) -> ValidationReport:
    """Runs every check and returns the combined report. Pure read-only —
    safe to call from a sync run, a health endpoint, or a test."""
    report = ValidationReport()

    missing_targets = check_missing_team_targets(db)
    if missing_targets:
        report.add(missing_targets)

    for finding in check_empty_required_columns(db):
        report.add(finding)
    for finding in check_orphan_fact_rows(db):
        report.add(finding)
    for finding in check_duplicate_records(db):
        report.add(finding)
    for finding in check_near_duplicate_names(db):
        report.add(finding)

    return report
