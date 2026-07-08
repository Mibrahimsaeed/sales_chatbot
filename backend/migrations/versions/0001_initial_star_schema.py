"""initial star schema tables

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "advisors",
        sa.Column("wid", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False, index=True),
        sa.Column("company", sa.String, index=True),
        sa.Column("region", sa.String),
        sa.Column("team", sa.String, index=True),
        sa.Column("office", sa.String),
        sa.Column("unit", sa.String),
        sa.Column("portfolio_lead", sa.String, index=True),
        sa.Column("management_lead", sa.String),
        sa.Column("bm", sa.String),
        sa.Column("zm", sa.String),
        sa.Column("rm", sa.String),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "sales_funnel",
        sa.Column("wid", sa.Integer, sa.ForeignKey("advisors.wid"), primary_key=True),
        sa.Column("mtd_new_connect", sa.Float, server_default="0"),
        sa.Column("mtd_followup_connect", sa.Float, server_default="0"),
        sa.Column("system_connect", sa.Float, server_default="0"),
        sa.Column("mtd_cr", sa.Float, server_default="0"),
        sa.Column("mtd_new_meeting", sa.Float, server_default="0"),
        sa.Column("mtd_followup_meeting", sa.Float, server_default="0"),
        sa.Column("mtd_todo", sa.Float, server_default="0"),
        sa.Column("mtd_booking_stored", sa.Float, server_default="0"),
        sa.Column("mtd_conversion", sa.Float, server_default="0"),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "pipeline",
        sa.Column("wid", sa.Integer, sa.ForeignKey("advisors.wid"), primary_key=True),
        sa.Column("pipeline", sa.Float, server_default="0"),
        sa.Column("overdue", sa.Float, server_default="0"),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "attendance",
        sa.Column("wid", sa.Integer, sa.ForeignKey("advisors.wid"), primary_key=True),
        sa.Column("biometric_time", sa.String),
        sa.Column("biometric_status", sa.String),
        sa.Column("biometric_mtd_ontime", sa.Float, server_default="0"),
        sa.Column("biometric_mtd_late", sa.Float, server_default="0"),
        sa.Column("biometric_mtd_not_marked", sa.Float, server_default="0"),
        sa.Column("login_time", sa.String),
        sa.Column("login_status", sa.String),
        sa.Column("login_mtd_ontime", sa.Float, server_default="0"),
        sa.Column("login_mtd_late", sa.Float, server_default="0"),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    performance_period = sa.Enum("MTD", "YTD", "3M", name="performanceperiod")
    op.create_table(
        "performance",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("wid", sa.Integer, sa.ForeignKey("advisors.wid"), nullable=False, index=True),
        sa.Column("period", performance_period, nullable=False),
        sa.Column("target", sa.Float, server_default="0"),
        sa.Column("cleared", sa.Float, server_default="0"),
        sa.Column("pct", sa.Float, server_default="0"),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint("wid", "period", name="uq_performance_wid_period"),
    )

    op.create_table(
        "portfolio",
        sa.Column("wid", sa.Integer, sa.ForeignKey("advisors.wid"), primary_key=True),
        sa.Column("value", sa.Float, server_default="0"),
        sa.Column("returned", sa.Float, server_default="0"),
        sa.Column("retention_pct", sa.Float, server_default="0"),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "bookings",
        sa.Column("wid", sa.Integer, sa.ForeignKey("advisors.wid"), primary_key=True),
        sa.Column("confirmed", sa.Float, server_default="0"),
        sa.Column("expected", sa.Float, server_default="0"),
        sa.Column("token", sa.Float, server_default="0"),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "calls",
        sa.Column("wid", sa.Integer, sa.ForeignKey("advisors.wid"), primary_key=True),
        sa.Column("answered_calls_mtd", sa.Float, server_default="0"),
        sa.Column("answered_calls_daily", sa.Float, server_default="0"),
        sa.Column("connects_mtd", sa.Float, server_default="0"),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "team_targets",
        sa.Column("team", sa.String, primary_key=True),
        sa.Column("target", sa.Float, server_default="0"),
        sa.Column("achieved", sa.Float, server_default="0"),
        sa.Column("achievement_pct", sa.Float, server_default="0"),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "morning_meeting_compliance",
        sa.Column("wid", sa.Integer, sa.ForeignKey("advisors.wid"), primary_key=True),
        sa.Column("team", sa.String),
        sa.Column("zonal_head", sa.String),
        sa.Column("status", sa.String),
        sa.Column("mtd_ontime", sa.Float, server_default="0"),
        sa.Column("mtd_late", sa.Float, server_default="0"),
        sa.Column("mtd_not_submitted", sa.Float, server_default="0"),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "advisor_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("wid", sa.Integer, index=True, nullable=False),
        sa.Column("snapshot_at", sa.DateTime, server_default=sa.func.now(), index=True),
        sa.Column("mtd_cleared", sa.Float),
        sa.Column("mtd_target", sa.Float),
        sa.Column("connects", sa.Float),
        sa.Column("meetings", sa.Float),
        sa.Column("overdue", sa.Float),
    )

    op.create_table(
        "sync_log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("started_at", sa.DateTime, nullable=False),
        sa.Column("finished_at", sa.DateTime),
        sa.Column("status", sa.String),
        sa.Column("rows_synced", sa.Integer),
        sa.Column("error", sa.String),
    )


def downgrade():
    op.drop_table("sync_log")
    op.drop_table("advisor_history")
    op.drop_table("morning_meeting_compliance")
    op.drop_table("team_targets")
    op.drop_table("calls")
    op.drop_table("bookings")
    op.drop_table("portfolio")
    op.drop_table("performance")
    sa.Enum(name="performanceperiod").drop(op.get_bind(), checkfirst=True)
    op.drop_table("attendance")
    op.drop_table("pipeline")
    op.drop_table("sales_funnel")
    op.drop_table("advisors")