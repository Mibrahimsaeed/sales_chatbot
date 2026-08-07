"""add DAILY to the performanceperiod enum type

Revision ID: 0012
Revises: 0011_sheet_audit_import_gaps
"""

from alembic import op


revision = "0012"
down_revision = "0011_sheet_audit_import_gaps"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TYPE performanceperiod ADD VALUE IF NOT EXISTS 'DAILY'"
    )


def downgrade() -> None:
    pass
