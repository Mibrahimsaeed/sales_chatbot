"""add in_master_sheet column to advisors

The advisors table is a union of every WID seen across ALL source sheets
(MasterSheet, CCMC DATA MTD, biometric, login report, connect session,
etc.) — a WID that only ever appears in a raw activity sheet, never on
the MasterSheet, was still getting a full Advisor row and showing up in
every leaderboard/summary/lookup right alongside real advisors. This
column lets every query path filter down to real MasterSheet advisors.
Defaults to true so existing rows aren't hidden before the next sync (see
etl/transform.py) correctly recomputes the real value for every WID.

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "advisors",
        sa.Column("in_master_sheet", sa.Boolean(), nullable=False, server_default="true"),
    )


def downgrade():
    op.drop_column("advisors", "in_master_sheet")
