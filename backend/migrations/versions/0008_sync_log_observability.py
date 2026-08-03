"""add observability columns to sync_log

The reliability rework (ETL retry/validation/join-integrity) needs a sync
run to record more than started/finished/status/rows/error:

  duration_seconds   — how long the run took, so a slow-degrading sync is
                       visible before it starts timing out
  attempts           — how many tries the run needed; >1 means transient
                       failures are happening even when status='success'
  trigger            — 'scheduled' | 'manual' | 'startup', so a gap in
                       scheduled runs is distinguishable from a quiet period
  rows_by_table      — JSON per-table load counts (previously only the
                       advisors count survived, as `rows_synced`)
  validation_report  — JSON from etl/validation.py
  join_report        — JSON from etl/join_integrity.py

All nullable: existing rows keep working, and a run that dies before
producing a report still records what it managed.

Also adds an index on (status, started_at) — every monitoring query is
"most recent successful run", which was a full scan of an ever-growing
table.

Revision ID: 0008
Revises: 0007
"""

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("sync_log", sa.Column("duration_seconds", sa.Float(), nullable=True))
    op.add_column("sync_log", sa.Column("attempts", sa.Integer(), nullable=True))
    op.add_column("sync_log", sa.Column("trigger", sa.String(), nullable=True))
    op.add_column("sync_log", sa.Column("rows_by_table", sa.String(), nullable=True))
    op.add_column("sync_log", sa.Column("validation_report", sa.String(), nullable=True))
    op.add_column("sync_log", sa.Column("join_report", sa.String(), nullable=True))
    op.create_index("ix_sync_log_status_started_at", "sync_log", ["status", "started_at"])


def downgrade():
    op.drop_index("ix_sync_log_status_started_at", table_name="sync_log")
    op.drop_column("sync_log", "join_report")
    op.drop_column("sync_log", "validation_report")
    op.drop_column("sync_log", "rows_by_table")
    op.drop_column("sync_log", "trigger")
    op.drop_column("sync_log", "attempts")
    op.drop_column("sync_log", "duration_seconds")
