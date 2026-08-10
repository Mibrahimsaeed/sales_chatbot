"""
ETL sync entrypoint (Area 1 of the reliability work).

What changed and why — the failure this addresses was NOT a crash. The
sync simply stopped happening, and nothing noticed for four days while the
chatbot kept answering from stale data as if it were current:

1. RETRY. extract_all() hits the Google Sheets API 13 times; any one
   transient 429/5xx/timeout used to abort the entire run and re-raise,
   leaving data stale until the next scheduled fire. Now each attempt is
   retried with backoff (settings.sync_max_attempts), and the attempt
   count is recorded even on success so absorbed-but-recurring flakiness
   is visible rather than hidden.

2. FAILURE DETECTION. A run that dies hard (process killed mid-sync) used
   to leave a SyncLog row stuck at status='running' forever, which every
   "last successful sync" query would either ignore or mistake for
   progress. reap_stuck_runs() marks those 'failed' so monitoring sees the
   truth.

3. DETAILED STATUS. duration, per-table row counts, attempt count,
   trigger, and the full validation + join-integrity reports are all
   persisted per run (migration 0008) instead of a single advisors count.

4. VALIDATION + JOIN AUDIT run after every load, and their reports are
   stored. Neither aborts the sync — a data gap must not block loading
   otherwise-good rows — but both are surfaced to /api/health.

run_sync() still raises on terminal failure, so a supervisor/cron wrapper
can act on a non-zero exit; the SyncLog row is written BEFORE the raise.
"""

from __future__ import annotations

import datetime
import json
import time

from app.core.config import settings
from app.core.logger import get_logger
from app.database.models import SyncLog
from app.database.session import SessionLocal
from etl.extract import extract_all
from etl.history_snapshot import write_snapshot_safe
from etl.join_integrity import audit_joins
from etl.load import load_all
from etl.transform import transform
from etl.validation import validate_database

log = get_logger("etl.sync")

# A run still marked 'running' after this long is assumed dead (process
# killed, machine rebooted) rather than genuinely in progress — real runs
# take ~25-30s against the current data volume.
STUCK_RUN_TIMEOUT_MINUTES = 60


def reap_stuck_runs(db, timeout_minutes: int = STUCK_RUN_TIMEOUT_MINUTES) -> int:
    """Mark long-abandoned status='running' rows as failed. Without this a
    hard-killed sync leaves a phantom 'running' row forever, and any
    monitoring that trusts status can't tell a live run from a dead one.
    Returns how many rows were reaped."""
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(minutes=timeout_minutes)
    stuck = db.query(SyncLog).filter(SyncLog.status == "running", SyncLog.started_at < cutoff).all()
    for row in stuck:
        row.status = "failed"
        row.error = f"abandoned — still 'running' after {timeout_minutes}m, assumed dead"
        row.finished_at = datetime.datetime.utcnow()
    if stuck:
        db.commit()
        log.warning(f"Reaped {len(stuck)} stuck sync run(s)")
    return len(stuck)


def _extract_transform_load() -> tuple[dict, dict]:
    """One full attempt. Returns (transformed data, per-table load counts)."""
    raw = extract_all()
    data = transform(raw)
    counts = load_all(data)
    return data, counts


def _run_with_retries() -> tuple[dict, dict, int]:
    """Retries the extract/transform/load cycle on transient failure.
    Returns (data, counts, attempts_used); re-raises the last exception if
    every attempt fails."""
    attempts = max(1, settings.sync_max_attempts)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            data, counts = _extract_transform_load()
            if attempt > 1:
                log.warning(f"Sync succeeded on attempt {attempt}/{attempts} after transient failure(s)")
            return data, counts, attempt
        except Exception as e:
            last_error = e
            if attempt < attempts:
                backoff = settings.sync_retry_backoff_seconds * attempt
                log.warning(f"Sync attempt {attempt}/{attempts} failed ({e!r}) — retrying in {backoff:.1f}s")
                time.sleep(backoff)
            else:
                log.error(f"Sync attempt {attempt}/{attempts} failed — no attempts remaining")

    raise last_error  # type: ignore[misc]


def run_sync(trigger: str = "manual") -> dict:
    """Full sync run. `trigger` is recorded on the SyncLog row so a gap in
    SCHEDULED runs (the scheduler being down) is distinguishable from a
    period with no manual runs. Returns a summary dict."""
    started_at = datetime.datetime.utcnow()
    monotonic_start = time.monotonic()

    db = SessionLocal()
    try:
        reap_stuck_runs(db)
        entry = SyncLog(started_at=started_at, status="running", trigger=trigger, attempts=0)
        db.add(entry)
        db.commit()
        db.refresh(entry)
        entry_id = entry.id
    finally:
        db.close()

    try:
        data, counts, attempts = _run_with_retries()
    except Exception as e:
        duration = time.monotonic() - monotonic_start
        db = SessionLocal()
        try:
            row = db.get(SyncLog, entry_id)
            row.finished_at = datetime.datetime.utcnow()
            row.status = "failed"
            row.error = f"{type(e).__name__}: {e}"[:2000]
            row.duration_seconds = round(duration, 2)
            row.attempts = max(1, settings.sync_max_attempts)
            db.commit()
        finally:
            db.close()
        log.exception(f"Sync failed after {settings.sync_max_attempts} attempt(s)")
        raise

    history_rows = write_snapshot_safe(data)
    log.info(f"AdvisorHistory snapshot: {history_rows} rows")

    # Advisor lifecycle: how many people the MasterSheet stopped listing
    # this run. Logged beside the row counts because a deactivation
    # changes every headcount and roster, and a silent one is the reason
    # 107 departed advisors sat in the roster unnoticed.
    deactivated = counts.get("advisors_deactivated", 0)
    if deactivated:
        log.info(
            f"Advisor reconciliation: {deactivated} advisor(s) no longer on the "
            "MasterSheet were deactivated (rows and history kept)"
        )

    # Validation + join audit are best-effort REPORTING steps: a failure
    # here must not turn an otherwise-successful data load into a failed
    # sync, since the loaded data is already live and correct.
    validation_dict: dict | None = None
    join_dict: dict | None = None
    db = SessionLocal()
    try:
        try:
            # The payload's MasterSheet WID set, so the lifecycle check
            # can tell "absent from the sheet" from "absent from the DB".
            report = validate_database(db, master_sheet_wids={
                a["wid"] for a in data.get("advisors", []) if a.get("in_master_sheet")
            })
            validation_dict = report.to_dict()
            level = "warning" if not report.ok or report.warning_count else "info"
            getattr(log, level)(
                f"Validation: {report.error_count} error(s), {report.warning_count} warning(s)"
            )
            for finding in report.findings:
                log.info(f"  [{finding.severity}] {finding.check}: {finding.message}")
        except Exception:
            log.exception("Validation step failed — sync itself is unaffected")

        try:
            join_report = audit_joins(db)
            join_dict = join_report.to_dict()
            if join_report.fully_unmatched_count:
                log.warning(
                    f"Join integrity: {join_report.fully_unmatched_count} master-sheet advisor(s) "
                    "have no rows in any core fact table — they contribute nothing to any metric"
                )
        except Exception:
            log.exception("Join integrity step failed — sync itself is unaffected")

        duration = time.monotonic() - monotonic_start
        row = db.get(SyncLog, entry_id)
        row.finished_at = datetime.datetime.utcnow()
        row.status = "success"
        row.rows_synced = counts.get("advisors", 0)
        row.duration_seconds = round(duration, 2)
        row.attempts = attempts
        row.rows_by_table = json.dumps(counts)
        row.validation_report = json.dumps(validation_dict) if validation_dict else None
        row.join_report = json.dumps(join_dict) if join_dict else None
        db.commit()
    finally:
        db.close()

    log.info(f"Sync complete in {duration:.1f}s (attempt {attempts}): {counts}")
    return {
        "sync_id": entry_id,
        "status": "success",
        "attempts": attempts,
        "duration_seconds": round(duration, 2),
        "counts": counts,
        "validation": validation_dict,
        "join_integrity": join_dict,
    }


if __name__ == "__main__":
    run_sync(trigger="manual")
