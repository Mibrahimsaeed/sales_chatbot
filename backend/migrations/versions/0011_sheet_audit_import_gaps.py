"""import the sheet columns and tabs the audit found unimported

A live audit of the two spreadsheets found 24 tabs, of which the ETL
fetched 13. Three unfetched tabs and three unread columns held data the
dashboard KPI formulas require:

    1 Unit            -> Advisor.unit          (declared since day one,
                                                never populated)
    YTD CCMC          -> SalesFunnel.ytd_*
    YTD P1 & Overdue  -> Pipeline.ytd_*
    MasterSheet       -> mtd_meetings_planned / mtd_meetings_conducted
    Login Report      -> login_mtd_not_marked
    Answered Calls    -> connects_daily

WHY YTD IS PARALLEL COLUMNS, NOT PERIOD ROWS. `Performance` models
periods as rows, and that is the better shape in general. It cannot be
used here for two structural reasons:

  1. These columns are NAMED for their period (`mtd_new_connect`), so a
     YTD row would hold year-to-date figures under "mtd" names.
  2. Every binding on sales_funnel/pipeline carries `period=None` and
     applies no period filter, so adding rows would make
     SUM(mtd_new_connect) sum MTD *and* YTD — silently doubling every
     funnel metric. Correcting that means changing bindings, which is
     outside the ETL.

Parallel columns are additive and inert: nothing reads `ytd_*` yet, and
nothing reading `mtd_*` can see them.

Revision ID: 0011_sheet_audit_import_gaps
Revises: 0010_add_request_trace_to_chat_log
"""

import sqlalchemy as sa
from alembic import op

revision = "0011_sheet_audit_import_gaps"
down_revision = "0010_add_request_trace_to_chat_log"
branch_labels = None
depends_on = None

# (table, column) — every added field defaults to 0.0 so existing rows
# read as "no data" rather than NULL, matching how _num() loads a blank
# cell everywhere else in the ETL.
_NUMERIC_COLUMNS = [
    ("sales_funnel", "mtd_meetings_planned"),
    ("sales_funnel", "mtd_meetings_conducted"),
    ("sales_funnel", "ytd_new_connect"),
    ("sales_funnel", "ytd_followup_connect"),
    ("sales_funnel", "ytd_cr"),
    ("sales_funnel", "ytd_new_meeting"),
    ("sales_funnel", "ytd_followup_meeting"),
    ("sales_funnel", "ytd_todo"),
    ("sales_funnel", "ytd_booking_stored"),
    ("sales_funnel", "ytd_conversion"),
    ("pipeline", "ytd_pipeline"),
    ("pipeline", "ytd_overdue"),
    ("attendance", "login_mtd_not_marked"),
    ("calls", "connects_daily"),
]


def upgrade() -> None:
    for table, column in _NUMERIC_COLUMNS:
        op.add_column(table, sa.Column(column, sa.Float(), server_default="0"))


def downgrade() -> None:
    for table, column in reversed(_NUMERIC_COLUMNS):
        op.drop_column(table, column)
