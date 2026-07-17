"""add resolved_ir column to chat_log

Stores the full QueryIR (as JSON text) that was compiled and run for a
chat message, when resolution went through the new IR pipeline
(nlu_pipeline.Resolution.kind == "ir"). Null for shortcut/plan-kind
resolutions, which never produce a QueryIR. This is what makes a
production query failure debuggable after the fact instead of only
having the final intent label and a confidence float — Part 6 of the
NLU architecture redesign.

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("chat_log", sa.Column("resolved_ir", sa.String(), nullable=True))


def downgrade():
    op.drop_column("chat_log", "resolved_ir")
