"""add request trace columns to chat_log

Phase 7 (request tracing). Before this, a chat request left behind only
the final intent label, a confidence float, and a QueryIR — which is
enough to know THAT something went wrong and nothing about WHY. Every
wrong-person and wrong-hierarchy bug in the pipeline audit had to be
diagnosed by hand-instrumenting the pipeline in a REPL, because the
decisive information (which identity candidates were considered, what the
planner decided the question was, what SQL actually ran) was never
recorded anywhere.

`trace` holds the full chain as JSON. The four scalars beside it are
denormalized out of that JSON so the common triage questions are
indexable instead of requiring a JSON scan of every row:

    trace_id      correlate a user report with its log line
    resolved_wid  "every request that resolved to this advisor"
    row_count     "which requests returned nothing"
    duration_ms   "what got slow"

Revision ID: 0010
Revises: 0009
"""

import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("chat_log", sa.Column("trace", sa.String(), nullable=True))
    op.add_column("chat_log", sa.Column("trace_id", sa.String(), nullable=True))
    op.add_column("chat_log", sa.Column("resolved_wid", sa.Integer(), nullable=True))
    op.add_column("chat_log", sa.Column("row_count", sa.Integer(), nullable=True))
    op.add_column("chat_log", sa.Column("duration_ms", sa.Float(), nullable=True))
    op.create_index("ix_chat_log_trace_id", "chat_log", ["trace_id"])
    op.create_index("ix_chat_log_resolved_wid", "chat_log", ["resolved_wid"])


def downgrade():
    op.drop_index("ix_chat_log_resolved_wid", table_name="chat_log")
    op.drop_index("ix_chat_log_trace_id", table_name="chat_log")
    op.drop_column("chat_log", "duration_ms")
    op.drop_column("chat_log", "row_count")
    op.drop_column("chat_log", "resolved_wid")
    op.drop_column("chat_log", "trace_id")
    op.drop_column("chat_log", "trace")
