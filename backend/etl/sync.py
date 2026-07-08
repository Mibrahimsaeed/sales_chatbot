import datetime
from app.database.session import SessionLocal
from app.database.models import SyncLog
from app.core.logger import get_logger
from etl.extract import extract_all
from etl.transform import transform
from etl.load import load_all

log = get_logger("etl.sync")


def run_sync():
    db = SessionLocal()
    entry = SyncLog(started_at=datetime.datetime.utcnow(), status="running")
    db.add(entry)
    db.commit()
    db.refresh(entry)
    entry_id = entry.id
    db.close()

    try:
        raw = extract_all()
        data = transform(raw)
        counts = load_all(data)
        total = counts.get("advisors", 0)

        db = SessionLocal()
        row = db.get(SyncLog, entry_id)
        row.finished_at = datetime.datetime.utcnow()
        row.status = "success"
        row.rows_synced = total
        db.commit()
        db.close()
        log.info(f"Sync complete: {counts}")
    except Exception as e:
        db = SessionLocal()
        row = db.get(SyncLog, entry_id)
        row.finished_at = datetime.datetime.utcnow()
        row.status = "failed"
        row.error = str(e)
        db.commit()
        db.close()
        log.exception("Sync failed")
        raise


if __name__ == "__main__":
    run_sync()