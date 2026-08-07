"""A populated source tab must never transform to zero rows in silence.

THE BUG THIS ENCODES. `ensure_advisor` returns None for a row whose WID
does not parse and every loop then does `continue`, so a renamed id
header does not raise — it produces NOTHING. `_num()` turns the now-
missing value columns into 0.0 further down, which means "this tab
stopped importing" and "everyone genuinely scored zero" are the same
observation everywhere downstream, including in the reply the user gets.

It had already happened twice, live, undetected:

    "Answered Calls" (667 rows)  id header had become ' '  -> 0 rows
    "P1 & Overdue"   (739 rows)  id header had become 'x'  -> 0 rows

The first was hiding the daily call data this phase exists to expose; the
second was the MTD half of pipeline/overdue, leaving that table filled
only by its YTD loop. Neither appeared in a log, a sync status, or a
test — the sync reported success both times.

Two defences, and the tests below cover both. `_identity` accepts the
spellings the live sheets actually use, so today's headers work. And
`_check_identity_columns` refuses to transform a populated tab it cannot
identify at all, so tomorrow's rename fails the sync instead of emptying
a table.
"""

import pytest

from etl.transform import (
    SourceIdentityError, _WID_HEADERS, _WID_KEYED_TABS, _identity, transform,
)

# Every key extract_all() produces; a test names only the tab it exercises.
_EMPTY = {
    "master_sheet": [], "ccmc_mtd": [], "p1_overdue": [], "connect_session": [],
    "biometric": [], "login_report": [], "mtd_perf": [], "ytd_perf": [],
    "three_m_perf": [], "portfolio": [], "npr": [], "answered_calls": [],
    "target_achievement": [], "one_unit": [], "ytd_ccmc": [], "ytd_p1_overdue": [],
}


def _src(**tabs):
    return {**_EMPTY, **tabs}


# The production headers as they stand today, verified against the live
# sheets. Written out rather than fetched so this test needs no network
# and fails loudly if the mapping stops covering them.
_LIVE_ANSWERED_CALLS = {
    " ": "2853", "Name": "Zainab Riaz",
    "Answered Calls MTD": "200", "Answered Calls Daily": "8",
    "Connects MTD": "500", "Connects Daily": "17",
}
_LIVE_P1_OVERDUE = {
    "x": "2853", "Name": "Zainab Riaz", "Pipeline": "7500", "Total Overdue": "2",
}


# ---------------------------------------------------------------------
# Today's headers work
# ---------------------------------------------------------------------


def test_the_answered_calls_tab_imports_under_its_current_headers():
    """Its id column is headed with a single space in the live sheet."""
    out = transform(_src(answered_calls=[_LIVE_ANSWERED_CALLS]))
    assert len(out["calls"]) == 1
    row = out["calls"][0]
    assert row["wid"] == 2853
    assert row["answered_calls_mtd"] == 200.0
    assert row["answered_calls_daily"] == 8.0
    assert row["connects_mtd"] == 500.0
    assert row["connects_daily"] == 17.0


def test_the_answered_calls_tab_names_its_advisor():
    out = transform(_src(answered_calls=[_LIVE_ANSWERED_CALLS]))
    assert [a["name"] for a in out["advisors"]] == ["Zainab Riaz"]


def test_the_p1_overdue_tab_imports_under_its_current_headers():
    """Its id column is headed 'x' in the live sheet."""
    out = transform(_src(p1_overdue=[_LIVE_P1_OVERDUE]))
    assert len(out["pipeline"]) == 1
    assert out["pipeline"][0] == {"wid": 2853, "pipeline": 7500.0, "overdue": 2.0}


@pytest.mark.parametrize("header", ["WID", " ", "x"])
def test_the_documented_and_the_observed_id_headers_all_resolve(header):
    """'WID' is the documented spelling; the other two are what the live
    tabs carry. All three must reach the same advisor."""
    wid_raw, name = _identity({header: "2853", "Name": "Zainab Riaz"})
    assert wid_raw == "2853"
    assert name == "Zainab Riaz"


def test_an_explicit_wid_column_wins_over_a_blank_one():
    """A tab carrying both must not be decided by dict ordering."""
    wid_raw, _ = _identity({" ": "999", "WID": "2853", "Name": "Zainab Riaz"})
    assert wid_raw == "2853"


@pytest.mark.parametrize("header", ["Advisor Name", "Name"])
def test_both_name_headers_resolve(header):
    _, name = _identity({"WID": "1", header: "Zainab Riaz"})
    assert name == "Zainab Riaz"


# ---------------------------------------------------------------------
# Tomorrow's rename fails loudly
# ---------------------------------------------------------------------


def test_a_populated_tab_with_an_unknown_id_header_refuses_to_transform():
    """The whole point. Silently returning zero rows is what let two tabs
    stop importing without anyone noticing."""
    broken = dict(_LIVE_ANSWERED_CALLS)
    broken["Employee Ref"] = broken.pop(" ")

    with pytest.raises(SourceIdentityError) as excinfo:
        transform(_src(answered_calls=[broken]))

    message = str(excinfo.value)
    assert "answered_calls" in message
    assert "Employee Ref" in message, "the message must show the headers it saw"


def test_the_refusal_names_every_broken_tab_not_just_the_first():
    broken_calls = {"Ref": "1", "Name": "A", "Answered Calls Daily": "1"}
    broken_pipe = {"Ref": "1", "Name": "A", "Pipeline": "1"}
    with pytest.raises(SourceIdentityError) as excinfo:
        transform(_src(answered_calls=[broken_calls], p1_overdue=[broken_pipe]))
    assert "answered_calls" in str(excinfo.value)
    assert "p1_overdue" in str(excinfo.value)


def test_an_id_column_holding_no_parseable_wid_is_caught_too():
    """A header that survives a rename but whose VALUES stop being WIDs
    empties the table just as completely."""
    with pytest.raises(SourceIdentityError):
        transform(_src(answered_calls=[{"WID": "n/a", "Name": "A"}]))


def test_an_empty_tab_is_not_an_error():
    """A tab with no rows has nothing to identify — that is a data state,
    not a broken mapping, and the sync must survive it."""
    assert transform(_src(answered_calls=[]))["calls"] == []


def test_a_tab_keyed_by_something_other_than_wid_is_not_checked():
    """biometric/login_report/one_unit resolve identity by SAP ID or User
    ID. Including them would fail the sync for tabs that were never
    WID-keyed to begin with."""
    for tab in ("biometric", "login_report", "one_unit", "master_sheet",
                "target_achievement"):
        assert tab not in _WID_KEYED_TABS


def test_every_checked_tab_is_a_real_source_key():
    assert set(_WID_KEYED_TABS) <= set(_EMPTY)


def test_the_documented_header_is_tried_first():
    """Order is the tie-break in _identity, so it is part of the contract."""
    assert _WID_HEADERS[0] == "WID"
