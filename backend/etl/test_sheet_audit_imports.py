"""Regression tests for the sheet-audit import gaps.

A live audit of the two spreadsheets found 24 tabs against 13 fetched.
Three unfetched tabs and three unread columns held data the dashboard KPI
formulas need — each had been classified as "no source exists" before the
sheets were actually read.

Every field imported here gets a test, because the failure mode for all
of them is silent: a tab that stops being fetched, or a header that gets
renamed, produces zeros rather than an error. `_num()` turns a missing
column into `0.0`, so "column gone" and "value is genuinely zero" look
identical downstream.
"""

import pytest

from etl.extract import extract_all
from etl.transform import _FUNNEL_COLUMNS, _funnel_fields, transform


def _src(**overrides):
    """A complete, empty source payload with the given tabs filled in.

    Every key extract_all() produces must be present, or transform's
    `src[...]` lookups raise. Building it here means a test names only
    the tab it cares about.
    """
    base = {
        "master_sheet": [], "ccmc_mtd": [], "p1_overdue": [], "connect_session": [],
        "biometric": [], "login_report": [], "mtd_perf": [], "ytd_perf": [],
        "three_m_perf": [], "portfolio": [], "npr": [], "answered_calls": [],
        "target_achievement": [], "one_unit": [], "ytd_ccmc": [], "ytd_p1_overdue": [],
    }
    base.update(overrides)
    return base


def _one(result, table):
    rows = result[table]
    assert len(rows) == 1, f"{table}: expected 1 row, got {len(rows)}"
    return rows[0]


# ---------------------------------------------------------------------
# Tabs the ETL must fetch
# ---------------------------------------------------------------------

def test_extract_requests_every_audited_tab(monkeypatch):
    """The three tabs the audit found unfetched. Asserted against the
    real extract_all() with the network stubbed, so a tab silently
    dropped from the fetch list fails here."""
    requested = []

    def fake_fetch(spreadsheet_id, tab_name):
        requested.append(tab_name)
        return []

    monkeypatch.setattr("etl.extract.fetch_tab", fake_fetch)
    extract_all()

    for tab in ("1 Unit", "YTD CCMC", "YTD P1 & Overdue"):
        assert tab in requested, f"'{tab}' is no longer fetched"


def test_the_previously_imported_tabs_are_still_fetched(monkeypatch):
    """Backward compatibility: adding tabs must not drop any."""
    requested = []
    monkeypatch.setattr("etl.extract.fetch_tab",
                        lambda sid, tab: requested.append(tab) or [])
    extract_all()

    for tab in ("MasterSheet", "CCMC DATA MTD", "P1 & Overdue", "Connect Session",
                "Biometric", "Login Report", "MTD Performance", "YTD Performance",
                "3M Performance", "Portfolio", "Answered Calls", "NPR",
                "Target Achievement"):
        assert tab in requested, tab
    assert len(requested) == 16


# ---------------------------------------------------------------------
# 1 Unit
# ---------------------------------------------------------------------

def test_one_unit_populates_advisor_unit():
    """`Advisor.unit` was declared from the start and permanently NULL,
    because the tab it names was never fetched."""
    result = transform(_src(
        one_unit=[{"SAP ID": "1", "Advisor Name": "Adv A", "Unit": "2", "Count": "1"}],
    ))
    assert _one(result, "advisors")["unit"] == "2"


def test_one_unit_stores_the_tally_not_the_flag():
    """The tab carries `Unit` (observed 0-4) and `Count` (1 where
    Unit > 0). The tally is stored because the flag is derivable from it
    and the reverse is not."""
    result = transform(_src(
        one_unit=[{"SAP ID": "1", "Advisor Name": "Adv A", "Unit": "3", "Count": "1"}],
    ))
    assert _one(result, "advisors")["unit"] == "3"


def test_one_unit_is_keyed_by_sap_id():
    """This tab has no WID column — it identifies advisors by SAP ID.
    Keying it on WID would silently import nothing."""
    result = transform(_src(
        one_unit=[{"SAP ID": "77", "Advisor Name": "Adv A", "Unit": "1", "Count": "1"}],
    ))
    advisor = _one(result, "advisors")
    assert advisor["wid"] == 77
    assert advisor["unit"] == "1"


def test_a_zero_unit_advisor_still_imports():
    """387 of 579 rows have Unit=0. They must load as "0", not be
    skipped — "no units" is data, not absence."""
    result = transform(_src(
        one_unit=[{"SAP ID": "1", "Advisor Name": "Adv A", "Unit": "0", "Count": "0"}],
    ))
    assert _one(result, "advisors")["unit"] == "0"


# ---------------------------------------------------------------------
# YTD CCMC / YTD P1 & Overdue
# ---------------------------------------------------------------------

YTD_ROW = {
    "WID": "1", "Name": "Adv A",
    "New Connect": "100", "Follow-up Connect": "20", "CR": "30",
    "New Meeting": "40", "Follow-up Meeting": "10", "Todo": "90",
    "Booking Stored": "20", "Conversion": "10",
}
MTD_ROW = {
    "WID": "1", "Name": "Adv A",
    "New Connect": "10", "Follow-up Connect": "2", "CR": "3",
    "New Meeting": "4", "Follow-up Meeting": "1", "Todo": "9",
    "Booking Stored": "2", "Conversion": "1",
}


@pytest.mark.parametrize("field", [f for field, _col in _FUNNEL_COLUMNS for f in (field,)])
def test_every_funnel_field_imports_at_both_periods(field):
    """Parametrised over the shared column table, so a field added there
    is covered without editing this test."""
    result = transform(_src(ccmc_mtd=[MTD_ROW], ytd_ccmc=[YTD_ROW]))
    funnel = _one(result, "sales_funnel")

    assert f"mtd_{field}" in funnel, field
    assert f"ytd_{field}" in funnel, field
    assert funnel[f"ytd_{field}"] == funnel[f"mtd_{field}"] * 10, field


def test_ytd_does_not_overwrite_mtd():
    """The core backward-compatibility property. Both periods live on
    ONE row in parallel columns; if YTD were written as a period row, or
    into the mtd_* fields, every existing funnel metric would change."""
    result = transform(_src(ccmc_mtd=[MTD_ROW], ytd_ccmc=[YTD_ROW]))
    funnel = _one(result, "sales_funnel")

    assert funnel["mtd_new_connect"] == 10.0
    assert funnel["ytd_new_connect"] == 100.0


def test_ytd_alone_still_produces_a_row():
    """An advisor present in YTD CCMC but not in CCMC DATA MTD must not
    be dropped."""
    result = transform(_src(ytd_ccmc=[YTD_ROW]))
    funnel = _one(result, "sales_funnel")
    assert funnel["ytd_cr"] == 30.0
    assert funnel.get("mtd_cr", 0.0) in (0.0, None)


def test_ytd_pipeline_and_overdue_import():
    result = transform(_src(
        p1_overdue=[{"WID": "1", "Name": "Adv A", "Pipeline": "500", "Total Overdue": "3"}],
        ytd_p1_overdue=[{"WID": "1", "Name": "Adv A", "Pipeline": "5000", "Overdue": "30"}],
    ))
    pipeline = _one(result, "pipeline")

    assert pipeline["pipeline"] == 500.0
    assert pipeline["ytd_pipeline"] == 5000.0
    assert pipeline["overdue"] == 3.0
    assert pipeline["ytd_overdue"] == 30.0


def test_the_ytd_overdue_header_differs_from_mtd():
    """"P1 & Overdue" calls it "Total Overdue"; "YTD P1 & Overdue" calls
    it "Overdue". Reading the MTD header against the YTD tab would load
    0.0 for every row and look like real data."""
    result = transform(_src(
        ytd_p1_overdue=[{"WID": "1", "Name": "Adv A", "Pipeline": "5000",
                         "Overdue": "30", "Total Overdue": "999"}],
    ))
    assert _one(result, "pipeline")["ytd_overdue"] == 30.0


# ---------------------------------------------------------------------
# The shared funnel mapper
# ---------------------------------------------------------------------

def test_the_funnel_mapping_is_declared_once():
    """"CCMC DATA MTD" and "YTD CCMC" have identical headers. Writing the
    mapping twice is precisely how the two would drift."""
    mtd = _funnel_fields(MTD_ROW, "mtd")
    ytd = _funnel_fields(YTD_ROW, "ytd")

    assert len(mtd) == len(ytd) == len(_FUNNEL_COLUMNS)
    assert {k[4:] for k in mtd} == {k[4:] for k in ytd}


def test_the_mapper_covers_every_ccmc_metric_column():
    columns = {column for _field, column in _FUNNEL_COLUMNS}
    assert columns == {
        "New Connect", "Follow-up Connect", "CR", "New Meeting",
        "Follow-up Meeting", "Todo", "Booking Stored", "Conversion",
    }


# ---------------------------------------------------------------------
# MasterSheet IBD columns
# ---------------------------------------------------------------------

def test_masersheet_supplies_planned_and_conducted_meetings():
    """Verified against the "P/C Meeting" tab: 0 disagreements across all
    608 shared WIDs. MasterSheet is already fetched, so it is the source
    and no new tab is needed."""
    result = transform(_src(master_sheet=[{
        "User ID": "1", "Advisor Name": "Adv A",
        "Meetings Planned": "7", "Meetings Conducted": "5",
    }]))
    funnel = _one(result, "sales_funnel")

    assert funnel["mtd_meetings_planned"] == 7.0
    assert funnel["mtd_meetings_conducted"] == 5.0


def test_the_ibd_columns_do_not_disturb_the_funnel_row():
    """MasterSheet runs BEFORE CCMC DATA MTD. Seeding the sales_funnel
    row there must not shadow the funnel numbers written afterwards."""
    result = transform(_src(
        master_sheet=[{"User ID": "1", "Advisor Name": "Adv A",
                       "Meetings Planned": "7", "Meetings Conducted": "5"}],
        ccmc_mtd=[MTD_ROW],
    ))
    funnel = _one(result, "sales_funnel")

    assert funnel["mtd_meetings_planned"] == 7.0
    assert funnel["mtd_new_connect"] == 10.0
    assert funnel["mtd_conversion"] == 1.0


def test_masersheet_org_fields_are_unchanged():
    """Backward compatibility for the columns this tab already fed."""
    result = transform(_src(master_sheet=[{
        "User ID": "1", "Advisor Name": "Adv A", "Company": "Graana",
        "Regional": "North", "Teams": "Blue Area",
        "Portfolio Lead": "Fawad Hafeez", "Management Lead": "Usman Ghani",
    }]))
    advisor = _one(result, "advisors")

    assert advisor["in_master_sheet"] is True
    assert advisor["team"] == "Blue Area"
    assert advisor["company"] == "Graana"
    # Person/org names pass through etl/normalize.py — asserted on
    # realistic values so this test pins the IMPORT, not the casing rule.
    assert advisor["portfolio_lead"] == "Fawad Hafeez"
    assert advisor["management_lead"] == "Usman Ghani"


# ---------------------------------------------------------------------
# Login Report / Answered Calls columns
# ---------------------------------------------------------------------

def test_login_not_marked_imports():
    """"Login Report" always carried this column; only the Biometric half
    read its equivalent, so the two attendance sources could not share a
    denominator shape."""
    result = transform(_src(login_report=[{
        "WID": "1", "Advisor Name": "Adv A",
        "MTD On Time": "18", "MTD Late": "2", "MTD Not Marked": "1",
    }]))
    attendance = _one(result, "attendance")

    assert attendance["login_mtd_ontime"] == 18.0
    assert attendance["login_mtd_late"] == 2.0
    assert attendance["login_mtd_not_marked"] == 1.0


def test_login_and_biometric_now_have_the_same_shape():
    """Both tabs have identical 13-column schemas. After this import both
    halves carry ontime/late/not_marked."""
    result = transform(_src(
        biometric=[{"WID": "1", "Advisor Name": "Adv A", "Time": "09:00",
                    "Comment": "On Time", "MTD On Time": "20",
                    "MTD Late": "1", "MTD Not Marked": "0"}],
        login_report=[{"WID": "1", "Advisor Name": "Adv A",
                       "Login In Time": "09:05", "Comment": "On Time",
                       "MTD On Time": "18", "MTD Late": "2", "MTD Not Marked": "1"}],
    ))
    attendance = _one(result, "attendance")

    for half in ("biometric", "login"):
        for part in ("ontime", "late", "not_marked"):
            assert f"{half}_mtd_{part}" in attendance, f"{half}_{part}"


def test_connects_daily_imports():
    result = transform(_src(answered_calls=[{
        "WID": "1", "Advisor Name": "Adv A",
        "Answered Calls MTD": "120", "Answered Calls Daily": "6",
        "Connects MTD": "100", "Connects Daily": "5",
    }]))
    calls = _one(result, "calls")

    assert calls["answered_calls_mtd"] == 120.0
    assert calls["answered_calls_daily"] == 6.0
    assert calls["connects_mtd"] == 100.0
    assert calls["connects_daily"] == 5.0


# ---------------------------------------------------------------------
# The load contract
# ---------------------------------------------------------------------

@pytest.mark.parametrize("table,field", [
    ("sales_funnel", "mtd_meetings_planned"),
    ("sales_funnel", "mtd_meetings_conducted"),
    ("sales_funnel", "ytd_new_connect"),
    ("sales_funnel", "ytd_conversion"),
    ("pipeline", "ytd_pipeline"),
    ("pipeline", "ytd_overdue"),
    ("attendance", "login_mtd_not_marked"),
    ("calls", "connects_daily"),
])
def test_every_new_field_has_a_model_column(table, field):
    """transform can emit a key the model lacks and the loader will
    raise only at insert time, against a real database. Checked here
    instead."""
    from sqlalchemy import inspect

    from etl.load import TABLE_MAP

    model = next(m for key, m, _cols in TABLE_MAP if key == table)
    assert field in [c.key for c in inspect(model).columns]


def test_no_new_table_was_introduced():
    """The whole import fits existing tables, so the loader needed no
    change and every existing upsert key still applies."""
    from etl.load import TABLE_MAP

    assert {key for key, _m, _c in TABLE_MAP} == {
        "advisors", "sales_funnel", "pipeline", "attendance",
        "portfolio", "bookings", "calls", "team_targets",
    }


def test_transform_still_returns_every_existing_key():
    """Backward compatibility for load_all()."""
    result = transform(_src())
    assert set(result) == {
        "advisors", "sales_funnel", "pipeline", "attendance", "performance",
        "portfolio", "bookings", "calls", "team_targets",
    }
