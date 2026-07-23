"""in_master_sheet is the only field transform() computes without a real
sheet cell backing it — True for a WID actually present in the MasterSheet
tab, False for a WID that only shows up in some other raw source sheet
(CCMC DATA MTD, biometric, etc.). See models.py's in_master_sheet column
docstring for why this matters: those raw-only WIDs used to get a full
Advisor row and show up in every leaderboard/summary/lookup."""

from etl.transform import transform

_EMPTY_SRC = {
    "master_sheet": [], "ccmc_mtd": [], "p1_overdue": [], "connect_session": [],
    "biometric": [], "login_report": [], "mtd_perf": [], "ytd_perf": [],
    "three_m_perf": [], "portfolio": [], "answered_calls": [], "npr": [],
    "target_achievement": [],
}


def _src(**overrides) -> dict:
    src = {k: list(v) for k, v in _EMPTY_SRC.items()}
    src.update(overrides)
    return src


def test_master_sheet_advisor_is_flagged_true():
    src = _src(master_sheet=[{"User ID": "1", "Advisor Name": "Real Advisor"}])
    advisors = {a["wid"]: a for a in transform(src)["advisors"]}
    assert advisors[1]["in_master_sheet"] is True


def test_raw_data_only_advisor_is_flagged_false():
    # WID 2 only ever appears in CCMC DATA MTD, never MasterSheet
    src = _src(ccmc_mtd=[{"WID": "2", "Name": "Raw Data Ghost"}])
    advisors = {a["wid"]: a for a in transform(src)["advisors"]}
    assert advisors[2]["in_master_sheet"] is False


def test_advisor_in_both_sources_is_flagged_true():
    src = _src(
        master_sheet=[{"User ID": "3", "Advisor Name": "Both"}],
        ccmc_mtd=[{"WID": "3", "Name": "Both"}],
    )
    advisors = {a["wid"]: a for a in transform(src)["advisors"]}
    assert advisors[3]["in_master_sheet"] is True


def test_advisor_appearing_in_master_sheet_after_other_sheets_still_flagged_true():
    # order independence: MasterSheet runs first in transform(), but the
    # flag must end up True regardless of dict-creation order
    src = _src(
        biometric=[{"WID": "4", "Advisor Name": "Later Confirmed"}],
        master_sheet=[{"User ID": "4", "Advisor Name": "Later Confirmed"}],
    )
    advisors = {a["wid"]: a for a in transform(src)["advisors"]}
    assert advisors[4]["in_master_sheet"] is True
