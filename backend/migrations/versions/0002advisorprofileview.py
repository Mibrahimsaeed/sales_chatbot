"""create advisor_profile view

Joins every star-schema fact table for the chatbot's most common read
pattern ("tell me about advisor X") so the API issues one query instead
of six. The normalized tables stay the source of truth and sync
independently; this view is a read convenience, not a duplicate store.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


CREATE_VIEW = """
CREATE OR REPLACE VIEW advisor_profile AS
SELECT
    a.wid,
    a.name,
    a.company,
    a.team,
    a.region,
    a.office,
    a.portfolio_lead,
    a.management_lead,

    sf.mtd_new_connect,
    sf.mtd_followup_connect,
    sf.system_connect,
    sf.mtd_cr,
    sf.mtd_new_meeting,
    sf.mtd_followup_meeting,
    sf.mtd_conversion,

    p.pipeline,
    p.overdue,

    att.biometric_time,
    att.biometric_status,
    att.login_time,
    att.login_status,

    mtd.target  AS mtd_target,
    mtd.cleared AS mtd_cleared,
    mtd.pct     AS mtd_pct,
    ytd.target  AS ytd_target,
    ytd.cleared AS ytd_cleared,
    ytd.pct     AS ytd_pct,

    port.value AS portfolio_value,
    port.retention_pct AS portfolio_retention_pct,

    b.confirmed AS npr_confirmed,
    b.expected  AS npr_expected,

    c.answered_calls_mtd

FROM advisors a
LEFT JOIN sales_funnel sf ON sf.wid = a.wid
LEFT JOIN pipeline p ON p.wid = a.wid
LEFT JOIN attendance att ON att.wid = a.wid
LEFT JOIN performance mtd ON mtd.wid = a.wid AND mtd.period = 'MTD'
LEFT JOIN performance ytd ON ytd.wid = a.wid AND ytd.period = 'YTD'
LEFT JOIN portfolio port ON port.wid = a.wid
LEFT JOIN bookings b ON b.wid = a.wid
LEFT JOIN calls c ON c.wid = a.wid;
"""

DROP_VIEW = "DROP VIEW IF EXISTS advisor_profile;"


def upgrade():
    op.execute(CREATE_VIEW)


def downgrade():
    op.execute(DROP_VIEW)