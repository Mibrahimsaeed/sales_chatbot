"""add confidence_metadata column to chat_log

Stores the JSON-serialized per-field confidence breakdown (intent/metric/
entities/filters/time), confidence_level, and ambiguity_reasons for the
QueryIR resolved for a chat message — Part 10 of the NLU redesign
(confidence-aware QueryIR generation). Pulled into its own column rather
than left nested inside resolved_ir so confidence-quality questions ("how
often do we reject for low confidence", "which dimension is weakest most
often") are queryable without parsing resolved_ir's JSON on every row.
Null for shortcut/plan-kind resolutions, same as resolved_ir.

Revision ID: 0007
Revises: 0006
"""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("chat_log", sa.Column("confidence_metadata", sa.String(), nullable=True))


def downgrade():
    op.drop_column("chat_log", "confidence_metadata")
