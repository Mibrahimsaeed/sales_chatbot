"""Writes one AdvisorHistory row per advisor per sync run, from the
already-transformed data etl.transform.transform() produced (not
re-derived from raw sheet rows) — Phase 2a of the trend-analytics
roadmap. AdvisorHistory is append-only: every call adds new rows, never
upserts, so later trend queries can diff any two snapshot_at timestamps.

Runs as a best-effort step after the core sync succeeds — a bug here
must not fail the primary Advisor/Performance/etc. sync that the rest of
the chatbot depends on today.
"""

import datetime

from sqlalchemy.orm import Session

from app.database.models import AdvisorHistory, PerformancePeriod
from app.database.session import SessionLocal
from app.core.logger import get_logger

log = get_logger("etl.history_snapshot")


def write_snapshot(data: dict, db: Session | None = None) -> int:
    """data is the dict from etl.transform.transform(). Returns the number
    of AdvisorHistory rows written. `db` is injectable for tests; the
    real run_sync() call site relies on the default (its own session)."""
    mtd_performance = {
        row["wid"]: row
        for row in data.get("performance", [])
        if row.get("period") == PerformancePeriod.MTD
    }
    pipeline_by_wid = {row["wid"]: row for row in data.get("pipeline", [])}
    funnel_by_wid = {row["wid"]: row for row in data.get("sales_funnel", [])}
    wids = [a["wid"] for a in data.get("advisors", [])]
    snapshot_at = datetime.datetime.utcnow()

    owns_session = db is None
    if db is None:
        db = SessionLocal()
    try:
        for wid in wids:
            perf = mtd_performance.get(wid, {})
            funnel = funnel_by_wid.get(wid, {})
            pipeline = pipeline_by_wid.get(wid, {})
            db.add(AdvisorHistory(
                wid=wid,
                snapshot_at=snapshot_at,
                mtd_cleared=perf.get("cleared"),
                mtd_target=perf.get("target"),
                connects=(
                    funnel.get("mtd_new_connect", 0) + funnel.get("mtd_followup_connect", 0)
                    if funnel else None
                ),
                meetings=(
                    funnel.get("mtd_new_meeting", 0) + funnel.get("mtd_followup_meeting", 0)
                    if funnel else None
                ),
                overdue=pipeline.get("overdue"),
            ))
        db.commit()
        return len(wids)
    finally:
        if owns_session:
            db.close()


def write_snapshot_safe(data: dict) -> int:
    """Fail-soft wrapper for the run_sync() call site: history is additive
    and non-critical, so a failure here is logged and swallowed rather than
    taking down the sync run that everything else depends on."""
    try:
        return write_snapshot(data)
    except Exception:
        log.exception("AdvisorHistory snapshot failed; core sync is unaffected")
        return 0
