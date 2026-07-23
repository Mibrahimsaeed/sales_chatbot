"""filter advisor_profile view to master-sheet advisors only

Same policy as 0005, applied to the one read path (advisor_service.
find_advisor_by_name) that queries this view directly instead of the
advisors table — a name lookup for a raw-data-only WID should say "not
found", not surface a ghost record that was never actually onboarded.

Revision ID: 0006
Revises: 0005
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


CREATE_VIEW_FILTERED = """
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
LEFT JOIN calls c ON c.wid = a.wid
WHERE a.in_master_sheet = true;
"""

CREATE_VIEW_UNFILTERED = """
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


def upgrade():
    op.execute(CREATE_VIEW_FILTERED)


def downgrade():
    op.execute(CREATE_VIEW_UNFILTERED)
