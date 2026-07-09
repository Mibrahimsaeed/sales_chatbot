"""
In-process daily scheduler. Use this if you're deploying without access to
system cron (e.g. inside a single long-running container). If you DO have
system cron or a Kubernetes CronJob available, prefer that instead — see
the crontab line and CronJob YAML in docs/data-model.md — since an external
scheduler keeps a sync crash from being tied to a process that also needs
to stay up for other reasons. This file is the fallback, not the default.

Run with: python -m etl.scheduler
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from app.core.config import settings
from app.core.logger import get_logger
from etl.sync import run_sync

log = get_logger("etl.scheduler")


def start():
    scheduler = BlockingScheduler()
    trigger = CronTrigger(hour=settings.sync_hour, minute=settings.sync_minute)
    scheduler.add_job(run_sync, trigger, id="daily_sheets_sync", misfire_grace_time=3600)

    log.info(f"Scheduler started — daily sync at {settings.sync_hour:02d}:{settings.sync_minute:02d}")

    # run once immediately on startup so data isn't stale until the first
    # scheduled fire (e.g. if the container restarts at 3pm, don't wait until 2am)
    try:
        run_sync()
    except Exception:
        log.exception("Startup sync failed — scheduler will still run on schedule")

    scheduler.start()


if __name__ == "__main__":
    start()