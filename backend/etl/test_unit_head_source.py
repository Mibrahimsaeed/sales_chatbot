"""Phase 25 — the Unit Head relationship comes from MasterSheet.

THE DEFECT. MasterSheet carries the whole org chart, and three of its
four levels were mapped correctly:

    "Management Lead" -> management_lead -> bcm         596/596
    "Portfolio Lead"  -> portfolio_lead  -> zonal_head  596/596
    "Regional"        -> region          -> a GEOGRAPHY  ← wrong
    "Teams"           -> team            -> team        596/596

"Regional" holds 11 distinct PEOPLE for all 596 rows — it is the unit
head. Written into `region`, it left `unit_head` to fall back on the RM
column of the ACTIVITY tabs, populated for 686 of 3,171 rows. Every Unit
Head scope was therefore built from a partial column: 170 advisors and
10,355 connects missing across 9 of 11 unit heads, and Chairman's MTD
revenue under-reported by 216 million (225,312,212 vs 8,655,336).

It also made `region` a list of people's names, which is why "Region"
appeared as an option when disambiguating a person.

WHY THE CARDINALITY IS THE PROOF. 182 management leads -> 88 portfolio
leads -> 11 regionals -> 9 teams nests exactly as
advisor -> bcm -> zonal_head -> unit_head -> team. A geography column
would not.

These tests run against transform() with hand-built source rows, so they
pin the MAPPING rather than today's sheet contents.
"""

import pytest

from etl.transform import transform

_EMPTY = {
    "master_sheet": [], "ccmc_mtd": [], "p1_overdue": [], "connect_session": [],
    "biometric": [], "login_report": [], "mtd_perf": [], "ytd_perf": [],
    "three_m_perf": [], "portfolio": [], "npr": [], "answered_calls": [],
    "target_achievement": [], "one_unit": [], "ytd_ccmc": [], "ytd_p1_overdue": [],
}


def _src(**tabs):
    return {**_EMPTY, **tabs}


def _master_row(wid, **over):
    row = {
        "User ID": str(wid), "Advisor Name": f"Advisor {wid}",
        "Company": "Graana", "Teams": "Blue Area",
        "Regional": "Haseeb Arslan",          # the UNIT HEAD
        "Region": "",                          # empty in the real sheet
        "Portfolio Lead": "Waqas Mehdi",       # zonal head
        "Management Lead": "Ali Asghar",       # bcm
    }
    row.update(over)
    return row


def _ccmc_row(wid, **over):
    row = {"WID": str(wid), "Name": f"Advisor {wid}", "Team": "Blue Area",
           "RM": "Kaleem Satti", "Region": "North", "Company": "Graana",
           "BM": "Some Bm", "ZM": "Some Zm", "Office": "HQ"}
    row.update(over)
    return row


def _advisor(out, wid):
    return next(a for a in out["advisors"] if a["wid"] == wid)


# ---------------------------------------------------------------------
# The mapping
# ---------------------------------------------------------------------


def test_the_unit_head_comes_from_master_sheet_regional():
    """THE fix. `rm` is the column behind the `unit_head` level."""
    out = transform(_src(master_sheet=[_master_row(1)]))
    assert _advisor(out, 1)["rm"] == "Haseeb Arslan"


def test_regional_is_not_written_to_region():
    """It is a person. Storing it as a geography is what put people's
    names in the `region` level."""
    out = transform(_src(master_sheet=[_master_row(1)]))
    assert _advisor(out, 1).get("region") != "Haseeb Arslan"


def test_region_takes_the_real_geography_from_the_activity_tab():
    """MasterSheet's own "Region" column is empty for every row, and CCMC
    carries North/Center/South — which the old mapping shadowed."""
    out = transform(_src(master_sheet=[_master_row(1)], ccmc_mtd=[_ccmc_row(1)]))
    assert _advisor(out, 1)["region"] == "North"


def test_the_other_three_levels_are_unchanged():
    out = transform(_src(master_sheet=[_master_row(1)]))
    advisor = _advisor(out, 1)
    assert advisor["management_lead"] == "Ali Asghar"      # bcm
    assert advisor["portfolio_lead"] == "Waqas Mehdi"      # zonal_head
    assert advisor["team"] == "Blue Area"


# ---------------------------------------------------------------------
# Precedence — MasterSheet wins, the activity tab fills gaps
# ---------------------------------------------------------------------


def test_master_sheet_wins_over_the_activity_rm():
    """The WID 6965 case: its activity RM said "Kaleem Satti" while
    MasterSheet said "Haseeb Arslan", and its zonal head independently
    rolls up to Haseeb Arslan. The authoritative source decides."""
    out = transform(_src(master_sheet=[_master_row(6965)],
                         ccmc_mtd=[_ccmc_row(6965, RM="Kaleem Satti")]))
    assert _advisor(out, 6965)["rm"] == "Haseeb Arslan"


def test_the_activity_rm_still_fills_an_advisor_master_sheet_omits():
    """The fallback requirement — someone absent from MasterSheet keeps
    the only unit head there is. This is _assign's existing "first
    non-blank wins, MasterSheet runs first" ordering, not a new rule."""
    out = transform(_src(ccmc_mtd=[_ccmc_row(2, RM="Kaleem Satti")]))
    assert _advisor(out, 2)["rm"] == "Kaleem Satti"


def test_a_blank_regional_does_not_erase_an_activity_value():
    out = transform(_src(master_sheet=[_master_row(3, Regional="")],
                         ccmc_mtd=[_ccmc_row(3, RM="Kaleem Satti")]))
    assert _advisor(out, 3)["rm"] == "Kaleem Satti"


# ---------------------------------------------------------------------
# The hierarchy nests
# ---------------------------------------------------------------------


def test_the_levels_nest_from_most_granular_to_least():
    """182 bcm >= 88 zonal_head >= 11 unit_head >= 9 team in production.
    Asserted here as an ORDER over a fixture that mirrors that shape, so
    a remapping that inverted two levels is caught."""
    rows = [
        _master_row(1, **{"Management Lead": "B1", "Portfolio Lead": "Z1",
                          "Regional": "U1", "Teams": "T1"}),
        _master_row(2, **{"Management Lead": "B2", "Portfolio Lead": "Z1",
                          "Regional": "U1", "Teams": "T1"}),
        _master_row(3, **{"Management Lead": "B3", "Portfolio Lead": "Z2",
                          "Regional": "U1", "Teams": "T1"}),
    ]
    out = transform(_src(master_sheet=rows))
    advisors = out["advisors"]

    def distinct(field):
        return len({a[field] for a in advisors if a.get(field)})

    assert distinct("management_lead") >= distinct("portfolio_lead")
    assert distinct("portfolio_lead") >= distinct("rm")
    assert distinct("rm") >= distinct("team")


def test_every_master_sheet_advisor_gets_a_unit_head():
    """The population guarantee. `rm` used to be the sparse column; it is
    now as complete as the other three, which is what makes a flat unit
    head filter trustworthy."""
    rows = [_master_row(wid) for wid in range(1, 6)]
    out = transform(_src(master_sheet=rows))
    master = [a for a in out["advisors"] if a.get("in_master_sheet")]

    assert len(master) == 5
    for field in ("rm", "portfolio_lead", "management_lead", "team"):
        missing = [a["wid"] for a in master if not a.get(field)]
        assert missing == [], f"{field} missing for {missing}"
