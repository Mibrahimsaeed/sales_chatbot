"""expose bm/zm/rm on the advisor_profile view

Phase 6 (reverse hierarchy). The view carried portfolio_lead and
management_lead but omitted bm, zm and rm entirely — so "who is X's BM?"
was not merely unrouted, the data was not reachable from the chatbot's
main read path at all, and an advisor profile could never show who the
person reports to.

hierarchy_service.get_manager_of() queries the `advisors` table directly
and therefore already worked without this change; this migration is what
lets the PROFILE reply and any /advisor API consumer see the same fields,
so the two paths can't disagree about who someone reports to.

Revision ID: 0009
Revises: 0008
"""

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


_COLUMNS_WITH_HIERARCHY = """
    a.wid,
    a.name,
    a.company,
    a.team,
    a.region,
    a.office,
    a.portfolio_lead,
    a.management_lead,
    a.bm,
    a.zm,
    a.rm,
"""

_COLUMNS_WITHOUT_HIERARCHY = """
    a.wid,
    a.name,
    a.company,
    a.team,
    a.region,
    a.office,
    a.portfolio_lead,
    a.management_lead,
"""

_VIEW_BODY = """
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


def _create_view(columns: str) -> str:
    # CREATE OR REPLACE cannot drop columns, so a rename-free rebuild
    # needs an explicit DROP first — the view has no dependents.
    return f"DROP VIEW IF EXISTS advisor_profile;\nCREATE VIEW advisor_profile AS\nSELECT{columns}{_VIEW_BODY}"


def upgrade():
    op.execute(_create_view(_COLUMNS_WITH_HIERARCHY))


def downgrade():
    op.execute(_create_view(_COLUMNS_WITHOUT_HIERARCHY))
