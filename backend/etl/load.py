"""
Upserts each table produced by etl.transform.transform(). Order matters:
`advisors` must load first since every other table has a FK to it.
"""

from sqlalchemy.dialects.postgresql import insert
from app.database.models import (
    Advisor, SalesFunnel, Pipeline, Attendance, Performance,
    Portfolio, Bookings, Calls, TeamTarget,
)
from app.database.session import SessionLocal

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

        db.commit()
        return counts
    finally:
        db.close()