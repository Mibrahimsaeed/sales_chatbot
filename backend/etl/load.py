"""
Upserts each table produced by etl.transform.transform(). Order matters:
`advisors` must load first since every other table has a FK to it.
"""

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from app.core.logger import get_logger
from app.database.models import (
    Advisor, SalesFunnel, Pipeline, Attendance, Performance,
    Portfolio, Bookings, Calls, TeamTarget,
)
from app.database.session import SessionLocal

log = get_logger("etl.load")

# The share of the currently-flagged roster a MasterSheet payload must
# cover before it is allowed to deactivate anybody.
#
# Reconciliation reads "absent from the payload" as "no longer on the
# sheet", which is only sound when the payload is COMPLETE. A truncated
# fetch — a partial API page, a sheet mid-edit, a tab renamed — looks
# exactly like a mass departure, and acting on it would deactivate the
# roster in one run and silently empty every leaderboard.
#
# 0.8 keeps normal attrition well inside the floor: 107 stale rows had
# accumulated over roughly a month here, ~15% of the roster, and a single
# sync's real departures are far smaller than that. Anything below the
# floor is reported and skipped, never guessed at.
_RECONCILE_MIN_COVERAGE = 0.8

# (table_key in transform() output, model, conflict target)
TABLE_MAP = [
    ("advisors", Advisor, ["wid"]),
    ("sales_funnel", SalesFunnel, ["wid"]),
    ("pipeline", Pipeline, ["wid"]),
    ("attendance", Attendance, ["wid"]),
    ("portfolio", Portfolio, ["wid"]),
    ("bookings", Bookings, ["wid"]),
    ("calls", Calls, ["wid"]),
    ("team_targets", TeamTarget, ["team"]),
]


def _upsert_rows(db, model, rows: list[dict], conflict_cols: list[str]):
    if not rows:
        return 0
    for row in rows:
        stmt = insert(model).values(**row)
        update_cols = {
            c.name: getattr(stmt.excluded, c.name)
            for c in model.__table__.columns
            if c.name not in conflict_cols
        }
        stmt = stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
        db.execute(stmt)
    return len(rows)


def reconcile_master_sheet(db, advisors: list[dict]) -> int:
    """Deactivate advisors the MasterSheet no longer lists. Returns the count.

    THE GAP THIS CLOSES. Loading is upsert-only, so a row is touched only
    when the payload contains it. An advisor who leaves the sheet is
    therefore never emitted again and never updated — their row stays
    `in_master_sheet=True` forever, carrying the hierarchy they had on
    their last day. 107 such rows had accumulated here, inflating one
    Unit Head's reported headcount from 77 to 89.

    It self-heals only for someone still visible in an ACTIVITY tab:
    they are re-emitted with the flag False (ensure_advisor's default)
    and the upsert corrects them. Every one of the 107 appeared in no tab
    at all, which is exactly why they survived.

    DEACTIVATE, NEVER DELETE. `in_master_sheet` is already the filter the
    whole system reads — query_compiler, the team/company/attendance
    services, the gazetteer and the advisor_profile view all honour it —
    so flipping it removes these people from every scope without touching
    a row, a fact table, or AdvisorHistory. Their past remains
    inspectable, and the change is reversible by their reappearing on the
    sheet.

    Only the flag is written. Hierarchy fields are deliberately left as
    they were: they record where the person sat when they left, which is
    true, and rewriting them would destroy the only record of it.
    """
    on_sheet = {a["wid"] for a in advisors if a.get("in_master_sheet")}
    flagged = {w for (w,) in db.query(Advisor.wid).filter(Advisor.in_master_sheet.is_(True))}
    if not flagged:
        return 0

    # THE FLOOR. Absence is only evidence of departure when the payload is
    # complete; a short fetch is indistinguishable from everyone leaving.
    coverage = len(on_sheet & flagged) / len(flagged)
    if coverage < _RECONCILE_MIN_COVERAGE:
        log.warning(
            "Skipping advisor reconciliation: the MasterSheet payload covers "
            f"{len(on_sheet & flagged)} of {len(flagged)} flagged advisors "
            f"({coverage:.0%}), below the {_RECONCILE_MIN_COVERAGE:.0%} floor. "
            "A truncated fetch looks identical to a mass departure, so nothing "
            "is deactivated."
        )
        return 0

    departed = flagged - on_sheet
    if not departed:
        return 0

    db.execute(
        update(Advisor).where(Advisor.wid.in_(departed)).values(in_master_sheet=False)
    )
    log.info(
        f"Advisor reconciliation: {len(departed)} advisor(s) no longer on the "
        "MasterSheet marked in_master_sheet=False (rows and history kept)"
    )
    return len(departed)


def load_all(data: dict) -> dict:
    """data is the dict from etl.transform.transform(). Returns row counts per table."""
    db = SessionLocal()
    counts = {}
    try:
        # advisors first — everything else FKs into it
        for key, model, conflict_cols in TABLE_MAP:
            counts[key] = _upsert_rows(db, model, data.get(key, []), conflict_cols)

        # performance is many-rows-per-advisor (wid, period) — separate handling
        perf_rows = data.get("performance", [])
        for row in perf_rows:
            stmt = insert(Performance).values(**row)
            update_cols = {
                c.name: getattr(stmt.excluded, c.name)
                for c in Performance.__table__.columns
                if c.name not in ("id", "wid", "period")
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["wid", "period"], set_=update_cols
            )
            db.execute(stmt)
        counts["performance"] = len(perf_rows)

        # AFTER the upserts, so an advisor who is on the sheet has already
        # been (re)written with the flag True and cannot be caught by the
        # sweep. Inside the same transaction, so a failure below leaves
        # the roster exactly as it was.
        counts["advisors_deactivated"] = reconcile_master_sheet(db, data.get("advisors", []))

        db.commit()
        return counts
    finally:
        db.close()