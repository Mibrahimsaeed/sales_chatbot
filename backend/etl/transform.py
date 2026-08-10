# """
# Splits the merged-by-WID rows into one list per star-schema table, matching
# app/database/models.py. Each function below owns exactly the columns that
# belong to its table — this is the enforcement point for "one table per
# business entity, not per sheet tab."
# """
# from app.database.models import PerformancePeriod


# def _num(value):
#     if value is None:
#         return 0.0

#     if isinstance(value, (int, float)):
#         return float(value)

#     value = str(value).strip()

#     if value in ["-", "", "N/A", "nan"]:
#         return 0.0

#     # remove commas and percentage signs
#     value = value.replace(",", "")
#     value = value.replace("%", "")

#     try:
#         return float(value)
#     except ValueError:
#         return 0.0


# def _wid(v):
#     try:
#         return int(float(v))
#     except (TypeError, ValueError):
#         return None


# def transform(src: dict) -> dict:
#     """src is the dict returned by etl.extract.extract_all().
#     Returns {"advisors": [...], "sales_funnel": [...], ...} ready for load.py.
#     """
#     advisors: dict[int, dict] = {}
#     sales_funnel: dict[int, dict] = {}
#     pipeline: dict[int, dict] = {}
#     attendance: dict[int, dict] = {}
#     performance: list[dict] = []
#     portfolio: dict[int, dict] = {}
#     bookings: dict[int, dict] = {}
#     calls: dict[int, dict] = {}

#     def ensure_advisor(wid_raw, name=None):
#         wid = _wid(wid_raw)
#         if wid is None:
#             return None
#         a = advisors.setdefault(wid, {"wid": wid})
#         if name:
#             a["name"] = name
#         return wid, a

#     # ---- sales_funnel + org columns on advisors (CCMC DATA MTD) ----
#     for row in src["ccmc_mtd"]:
#         res = ensure_advisor(row.get("WID"), row.get("Name"))
#         if not res:
#             continue
#         wid, a = res
#         a.update({
#             "team": row.get("Team"), "bm": row.get("BM"), "zm": row.get("ZM"), "rm": row.get("RM"),
#             "company": row.get("Company"), "region": row.get("Region"), "office": row.get("Office"),
#         })
#         sales_funnel[wid] = {
#             "wid": wid,
#             "mtd_new_connect": _num(row.get("New Connect")),
#             "mtd_followup_connect": _num(row.get("Follow-up Connect")),
#             "mtd_cr": _num(row.get("CR")),
#             "mtd_new_meeting": _num(row.get("New Meeting")),
#             "mtd_followup_meeting": _num(row.get("Follow-up Meeting")),
#             "mtd_todo": _num(row.get("Todo")),
#             "mtd_booking_stored": _num(row.get("Booking Stored")),
#             "mtd_conversion": _num(row.get("Conversion")),
#         }

#     # ---- system-verified connect count layered onto sales_funnel (Connect Session) ----
#     for row in src.get("connect_session", []):
#         res = ensure_advisor(row.get("WID"), row.get("Name"))
#         if not res:
#             continue
#         wid, _ = res
#         sales_funnel.setdefault(wid, {"wid": wid})
#         sales_funnel[wid]["system_connect"] = _num(row.get("Total Connect Through System"))

#     # ---- org hierarchy + leads (MasterSheet) ----
#     for row in src["master_sheet"]:
#         res = ensure_advisor(row.get("User ID"), row.get("Advisor Name"))
#         if not res:
#             continue
#         _, a = res
#         a.setdefault("company", row.get("Company"))
#         a.setdefault("region", row.get("Regional"))
#         a.setdefault("team", row.get("Teams"))
#         a["portfolio_lead"] = row.get("Portfolio Lead")
#         a["management_lead"] = row.get("Management Lead")

#     # ---- pipeline (P1 & Overdue) ----
#     for row in src["p1_overdue"]:
#         res = ensure_advisor(row.get("WID"), row.get("Name"))
#         if not res:
#             continue
#         wid, _ = res
#         pipeline[wid] = {
#             "wid": wid,
#             "pipeline": _num(row.get("Pipeline")),
#             "overdue": _num(row.get("Total Overdue")),
#         }

#     # ---- attendance: biometric half ----
#     for row in src["biometric"]:
#         res = ensure_advisor(row.get("WID"), row.get("Advisor Name"))
#         if not res:
#             continue
#         wid, _ = res
#         attendance.setdefault(wid, {"wid": wid})
#         attendance[wid].update({
#             "biometric_time": row.get("Time"),
#             "biometric_status": row.get("Comment"),
#             "biometric_mtd_ontime": _num(row.get("MTD On Time")),
#             "biometric_mtd_late": _num(row.get("MTD Late")),
#             "biometric_mtd_not_marked": _num(row.get("MTD Not Marked")),
#         })

#     # ---- attendance: login half ----
#     for row in src["login_report"]:
#         res = ensure_advisor(row.get("WID"), row.get("Advisor Name"))
#         if not res:
#             continue
#         wid, _ = res
#         attendance.setdefault(wid, {"wid": wid})
#         attendance[wid].update({
#             "login_time": row.get("Login In Time"),
#             "login_status": row.get("Comment"),
#             "login_mtd_ontime": _num(row.get("MTD On Time")),
#             "login_mtd_late": _num(row.get("MTD Late")),
#         })

#     # ---- performance: one row per (wid, period) ----
#     # ---- performance: one row per (wid, period) ----
#     def perf_rows(rows, period: PerformancePeriod):
#         for row in rows:
#             res = ensure_advisor(row.get("WID"), row.get("Advisor Name"))
#             if not res:
#                 continue

#             wid, _ = res

#             performance.append({
#                 "wid": wid,
#                 "period": period,
#                 "target": _num(row.get("Target")),
#                 "cleared": _num(row.get("Cleared")),
#                 "pct": _num(row.get("%")),
#             })

#     perf_rows(src["mtd_perf"], PerformancePeriod.MTD)
#     perf_rows(src["ytd_perf"], PerformancePeriod.YTD)
#     perf_rows(src["three_m_perf"], PerformancePeriod.THREE_M)
    



#     # ---- portfolio ----
#     for row in src["portfolio"]:
#         res = ensure_advisor(row.get("WID"), row.get("Advisor Name"))
#         if not res:
#             continue
#         wid, _ = res
#         portfolio[wid] = {
#             "wid": wid,
#             "value": _num(row.get("Portfolio")),
#             "returned": _num(row.get("Returned")),
#             "retention_pct": _num(row.get("Retention %")),
#         }

#     # ---- bookings (NPR) ----
#     for row in src["npr"]:
#         res = ensure_advisor(row.get("WID"), row.get("Name"))
#         if not res:
#             continue
#         wid, _ = res
#         bookings[wid] = {
#             "wid": wid,
#             "confirmed": _num(row.get("Confirmed")),
#             "expected": _num(row.get("Expected")),
#             "token": _num(row.get("Token")),
#         }

#     # ---- calls (Answered Calls) ----
#     for row in src["answered_calls"]:
#         res = ensure_advisor(row.get("WID"), row.get("Advisor Name"))
#         if not res:
#             continue
#         wid, _ = res
#         calls[wid] = {
#             "wid": wid,
#             "answered_calls_mtd": _num(row.get("Answered Calls MTD")),
#             "answered_calls_daily": _num(row.get("Answered Calls Daily")),
#             "connects_mtd": _num(row.get("Connects MTD")),
#         }

#     # ---- team_targets (Target Achievement) — standalone, not keyed by wid ----
#     team_targets = []
#     for row in src.get("target_achievement", []):
#         team = row.get("Team")
#         if not team:
#             continue
#         team_targets.append({
#             "team": team,
#             "target": _num(row.get("Target")),
#             "achieved": _num(row.get("Total Achieved")),
#             "achievement_pct": _num(row.get("Achievement %")),
#         })

#     return {
#         "advisors": [a for a in advisors.values() if a.get("name")],
#         "sales_funnel": list(sales_funnel.values()),
#         "pipeline": list(pipeline.values()),
#         "attendance": list(attendance.values()),
#         "performance": performance,
#         "portfolio": list(portfolio.values()),
#         "bookings": list(bookings.values()),
#         "calls": list(calls.values()),
#         "team_targets": team_targets,
#     }



"""
Splits the merged-by-WID rows into one list per star-schema table, matching
app/database/models.py.

FIX (2026-07-13): advisors.team / company / region / portfolio_lead /
management_lead were being lost for advisors whose CCMC DATA MTD row had a
blank Team cell, because the old code did:

    a.update({"team": row.get("Team"), ...})   # ccmc_mtd runs first, writes "" unconditionally
    ...
    a.setdefault("team", row.get("Teams"))     # master_sheet runs later, but setdefault
                                                # is a no-op once the key exists — even if it's ""

That silently discarded MasterSheet's correct value for any WID where CCMC
DATA MTD had a blank Team. It also meant advisors who exist ONLY in
Biometric/Login Report (not in CCMC DATA MTD or MasterSheet) never got a
team at all, even though the Biometric tab has its own Team/Company/RM/
Portfolio Lead/Management Lead columns that were never read.

Fix: a single `_assign()` helper that (1) never overwrites a real value with
a blank one, and (2) never lets an early blank value block a later real one.
Processing order is also changed so MasterSheet (your documented authoritative
source for org hierarchy) runs before CCMC DATA MTD, and Biometric/Login
Report are added as a last-resort fallback source for org columns.
"""
from app.database.models import PerformancePeriod
from etl.normalize import normalize_field, normalize_org_name


def _num(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if value in ["-", "", "N/A", "nan"]:
        return 0.0
    value = value.replace(",", "").replace("%", "")
    try:
        return float(value)
    except ValueError:
        return 0.0


def _wid(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# The header spellings the WID column actually appears under, in priority
# order. "WID" is the documented one; the rest are what the live tabs
# carry today — "Answered Calls" heads its id column with a single space,
# "P1 & Overdue" with a lowercase "x". Both are unlabelled id columns as
# far as the sheet author is concerned.
#
# Listed rather than sniffed positionally: a positional rule silently
# picks up whatever ends up in column A when someone inserts one, which
# is the same class of silent-wrong-data failure this replaces. An
# unrecognised spelling is caught loudly by _check_identity_columns
# below instead of being guessed at.
_WID_HEADERS: tuple[str, ...] = ("WID", "Wid", "wid", " ", "", "x")

# Likewise for the name column.
_NAME_HEADERS: tuple[str, ...] = ("Advisor Name", "Name", "name")


def _identity(row: dict) -> tuple:
    """(wid_raw, name) for a WID-keyed source row.

    Every WID-keyed loop reads its identity through here, so a tab whose
    id header is renamed is one edit to _WID_HEADERS rather than one per
    loop — and so the check below can ask the same question the loops do
    rather than a lookalike of it.
    """
    wid_raw = next((row[key] for key in _WID_HEADERS if key in row), None)
    name = next((row[key] for key in _NAME_HEADERS if key in row), None)
    return wid_raw, name


# Source tabs keyed by WID, and the loop that consumes each. Tabs keyed by
# SAP ID (biometric, login_report, one_unit) or by name/team
# (master_sheet, target_achievement) are deliberately absent — they
# resolve identity another way and are not what this checks.
_WID_KEYED_TABS: tuple[str, ...] = (
    "ccmc_mtd", "connect_session", "p1_overdue", "mtd_perf", "ytd_perf",
    "three_m_perf", "portfolio", "npr", "answered_calls", "ytd_ccmc",
    "ytd_p1_overdue",
)


class SourceIdentityError(RuntimeError):
    """A source tab has rows but none of them yields a WID."""


def _check_identity_columns(src: dict) -> None:
    """Fail loudly when a populated tab can no longer be identified.

    THE FAILURE THIS EXISTS FOR. `ensure_advisor` returns None for a row
    whose WID does not parse, and every loop then does `continue` — so a
    renamed id header does not raise, it produces ZERO rows. `_num()`
    turns the missing columns into 0.0 further down, which means "the tab
    stopped importing" and "everyone genuinely scored zero" are the same
    observation downstream.

    That is not hypothetical: "Answered Calls" (667 rows) and "P1 &
    Overdue" (739 rows) were both importing nothing, because their id
    headers had become " " and "x". The daily call data was in the sheet
    and in neither the database's future nor any error log.

    Checked BEFORE the loops run, so a broken source fails the sync
    instead of emptying a table — load.py's job is to write what
    transform produced, and "produced nothing" is indistinguishable there
    from "there is nothing".
    """
    broken = [
        tab for tab in _WID_KEYED_TABS
        if src.get(tab) and not any(_wid(_identity(row)[0]) is not None for row in src[tab])
    ]
    if broken:
        raise SourceIdentityError(
            "no WID column found in " + ", ".join(
                f"{tab!r} ({len(src[tab])} rows, headers "
                f"{list(src[tab][0].keys())})" for tab in broken
            )
            + f" — the id header must be one of {_WID_HEADERS}. Refusing to "
            "transform: every row would be skipped and the table would be "
            "emptied rather than left alone."
        )


def _clean(value):
    """Normalize a raw cell value: strip whitespace, treat blank/placeholder as missing."""
    if value is None:
        return None
    value = str(value).strip()
    if value in ("", "-", "N/A", "n/a", "nan", "None"):
        return None
    return value


# The CCMC funnel column layout, declared ONCE. "CCMC DATA MTD" and
# "YTD CCMC" have identical headers, so the only difference between the
# two loops below is the destination prefix — writing the mapping twice
# is exactly how the two would drift.
_FUNNEL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("new_connect", "New Connect"),
    ("followup_connect", "Follow-up Connect"),
    ("cr", "CR"),
    ("new_meeting", "New Meeting"),
    ("followup_meeting", "Follow-up Meeting"),
    ("todo", "Todo"),
    ("booking_stored", "Booking Stored"),
    ("conversion", "Conversion"),
)


def _funnel_fields(row: dict, prefix: str) -> dict:
    """The funnel numbers from one CCMC-shaped row, keyed `<prefix>_<field>`."""
    return {f"{prefix}_{field}": _num(row.get(column))
            for field, column in _FUNNEL_COLUMNS}


def _assign(a: dict, key: str, value):
    """Write `value` into advisor dict `a[key]` UNLESS value is empty, and
    UNLESS `a[key]` already holds a real (non-empty) value from a
    higher-priority source that ran earlier. This is the single choke point
    that replaces every ad-hoc `.update()` / `.setdefault()` call below, so
    field-loss like the team/company bug can't reoccur silently.

    Also the single place canonical normalization is applied (etl/
    normalize.py) — collapsing the whitespace/separator/casing variants
    that were splitting one real office or person across several stored
    spellings. Being the one choke point is exactly why it belongs here:
    every org/person field on `advisors` already flows through this
    function, so no source-sheet loop needs its own normalization call."""
    value = normalize_field(key, _clean(value))
    if value is None:
        return
    if not _clean(a.get(key)):
        a[key] = value


def transform(src: dict) -> dict:
    """src is the dict returned by etl.extract.extract_all().
    Returns {"advisors": [...], "sales_funnel": [...], ...} ready for load.py.
    """
    # Before anything is built: a populated tab nobody can identify would
    # otherwise transform to zero rows and empty its table on load.
    _check_identity_columns(src)

    advisors: dict[int, dict] = {}
    sales_funnel: dict[int, dict] = {}
    pipeline: dict[int, dict] = {}
    attendance: dict[int, dict] = {}
    performance: list[dict] = []
    portfolio: dict[int, dict] = {}
    bookings: dict[int, dict] = {}
    calls: dict[int, dict] = {}

    def ensure_advisor(wid_raw, name=None):
        wid = _wid(wid_raw)
        if wid is None:
            return None
        # in_master_sheet defaults False here — only the MasterSheet loop
        # below ever sets it True. Every other source sheet can introduce
        # a WID that was never onboarded (raw activity data, terminated
        # employees, etc.); those rows must not silently look identical
        # to a real advisor in every leaderboard/summary/lookup.
        a = advisors.setdefault(wid, {"wid": wid, "in_master_sheet": False})
        _assign(a, "name", name)
        return wid, a

    # ---- 1. MasterSheet FIRST — documented authoritative source for org
    #         hierarchy (Company / Regional / Teams / Portfolio Lead /
    #         Management Lead). Runs before CCMC DATA MTD so a blank cell
    #         downstream can never shadow a good value set here. Also the
    #         ONLY loop that sets in_master_sheet — a real (not empty)
    #         assignment, not run through _assign()'s don't-overwrite
    #         semantics, since every WID here genuinely IS on the sheet. ----
    for row in src["master_sheet"]:
        res = ensure_advisor(row.get("User ID"), row.get("Advisor Name"))
        if not res:
            continue
        wid_ms, a = res
        a["in_master_sheet"] = True
        _assign(a, "company", row.get("Company"))
        # "Regional" is the UNIT HEAD, not a place. It holds 11 distinct
        # PEOPLE for all 596 rows, and the level cardinality confirms the
        # chain it belongs to: 182 management leads -> 88 portfolio leads
        # -> 11 regionals -> 9 teams, nesting exactly as
        # advisor -> bcm -> zonal_head -> unit_head -> team.
        #
        # It was written to `region`, so `unit_head` fell back to the RM
        # column on the ACTIVITY tabs — populated for 686 of 3,171 rows
        # against MasterSheet's 596 of 596. Every Unit Head scope was
        # therefore built from a partial column: 170 advisors and 10,355
        # connects went missing across 9 of 11 unit heads, and Chairman's
        # revenue was under-reported by 216 million. `region`
        # simultaneously became a list of people's names, which is what
        # put "Region" in the disambiguation prompt for a person.
        #
        # MasterSheet runs FIRST, so this now WINS and the activity RM
        # below fills only the advisors MasterSheet does not list — the
        # fallback ordering _assign already provides, and the same
        # arrangement portfolio_lead and management_lead have always had.
        _assign(a, "rm", row.get("Regional"))
        _assign(a, "team", row.get("Teams"))
        _assign(a, "portfolio_lead", row.get("Portfolio Lead"))
        _assign(a, "management_lead", row.get("Management Lead"))
        # `region` is deliberately NOT written here. MasterSheet's own
        # "Region" column is empty for all 596 rows, and CCMC DATA MTD
        # carries the real geography (North/Center/South) — which this
        # line used to shadow with a person's name.
        # MasterSheet carries the IBD meeting figures too. Verified equal
        # to the "P/C Meeting" tab on all 608 shared WIDs, and this tab is
        # already fetched — so it is the source, per the audit.
        sales_funnel.setdefault(wid_ms, {"wid": wid_ms})
        sales_funnel[wid_ms].update({
            "mtd_meetings_planned": _num(row.get("Meetings Planned")),
            "mtd_meetings_conducted": _num(row.get("Meetings Conducted")),
        })

    # ---- 2. sales_funnel + org columns on advisors (CCMC DATA MTD) ----
    for row in src["ccmc_mtd"]:
        res = ensure_advisor(*_identity(row))
        if not res:
            continue
        wid, a = res
        _assign(a, "team", row.get("Team"))
        _assign(a, "bm", row.get("BM"))
        _assign(a, "zm", row.get("ZM"))
        _assign(a, "rm", row.get("RM"))
        _assign(a, "company", row.get("Company"))
        _assign(a, "region", row.get("Region"))
        _assign(a, "office", row.get("Office"))
        sales_funnel.setdefault(wid, {"wid": wid})
        sales_funnel[wid].update(_funnel_fields(row, "mtd"))

    # ---- 3. system-verified connect count layered onto sales_funnel (Connect Session) ----
    for row in src.get("connect_session", []):
        res = ensure_advisor(*_identity(row))
        if not res:
            continue
        wid, _ = res
        sales_funnel.setdefault(wid, {"wid": wid})
        sales_funnel[wid]["system_connect"] = _num(row.get("Total Connect Through System"))

    # ---- 4. pipeline (P1 & Overdue) ----
    for row in src["p1_overdue"]:
        res = ensure_advisor(*_identity(row))
        if not res:
            continue
        wid, _ = res
        pipeline[wid] = {
            "wid": wid,
            "pipeline": _num(row.get("Pipeline")),
            "overdue": _num(row.get("Total Overdue")),
        }

    # ---- 5. attendance: biometric half.
    #         FIX: Biometric also carries Team/Company/RM/Portfolio Lead/
    #         Management Lead — capture them as a fallback for advisors who
    #         never appear in MasterSheet or CCMC DATA MTD. ----
    for row in src["biometric"]:
        res = ensure_advisor(*_identity(row))
        if not res:
            continue
        wid, a = res
        _assign(a, "team", row.get("Team"))
        _assign(a, "company", row.get("Company"))
        _assign(a, "rm", row.get("RM"))
        _assign(a, "portfolio_lead", row.get("Portfolio Lead"))
        _assign(a, "management_lead", row.get("Management Lead"))
        attendance.setdefault(wid, {"wid": wid})
        attendance[wid].update({
            "biometric_time": row.get("Time"),
            "biometric_status": row.get("Comment"),
            "biometric_mtd_ontime": _num(row.get("MTD On Time")),
            "biometric_mtd_late": _num(row.get("MTD Late")),
            "biometric_mtd_not_marked": _num(row.get("MTD Not Marked")),
        })

    # ---- 6. attendance: login half (same fallback treatment if Login Report
    #         carries org columns too — harmless no-op via _assign if not) ----
    for row in src["login_report"]:
        res = ensure_advisor(*_identity(row))
        if not res:
            continue
        wid, a = res
        _assign(a, "team", row.get("Team"))
        _assign(a, "company", row.get("Company"))
        attendance.setdefault(wid, {"wid": wid})
        attendance[wid].update({
            "login_time": row.get("Login In Time"),
            "login_status": row.get("Comment"),
            "login_mtd_ontime": _num(row.get("MTD On Time")),
            "login_mtd_late": _num(row.get("MTD Late")),
            "login_mtd_not_marked": _num(row.get("MTD Not Marked")),
        })

    # ---- 7. performance: one row per (wid, period) ----
    def perf_rows(rows, period: PerformancePeriod):
        for row in rows:
            res = ensure_advisor(*_identity(row))
            if not res:
                continue
            wid, _ = res
            performance.append({
                "wid": wid,
                "period": period,
                "target": _num(row.get("Target")),
                "cleared": _num(row.get("Cleared")),
                "pct": _num(row.get("%")),
            })

    perf_rows(src["mtd_perf"], PerformancePeriod.MTD)
    perf_rows(src["ytd_perf"], PerformancePeriod.YTD)
    perf_rows(src["three_m_perf"], PerformancePeriod.THREE_M)

    # ---- 8. portfolio ----
    for row in src["portfolio"]:
        res = ensure_advisor(*_identity(row))
        if not res:
            continue
        wid, _ = res
        portfolio[wid] = {
            "wid": wid,
            "value": _num(row.get("Portfolio")),
            "returned": _num(row.get("Returned")),
            "retention_pct": _num(row.get("Retention %")),
        }

    # ---- 9. bookings (NPR) ----
    for row in src["npr"]:
        res = ensure_advisor(*_identity(row))
        if not res:
            continue
        wid, _ = res
        bookings[wid] = {
            "wid": wid,
            "confirmed": _num(row.get("Confirmed")),
            "expected": _num(row.get("Expected")),
            "token": _num(row.get("Token")),
        }

    # ---- 10. calls (Answered Calls) ----
    for row in src["answered_calls"]:
        res = ensure_advisor(*_identity(row))
        if not res:
            continue
        wid, _ = res
        calls[wid] = {
            "wid": wid,
            "answered_calls_mtd": _num(row.get("Answered Calls MTD")),
            "answered_calls_daily": _num(row.get("Answered Calls Daily")),
            "connects_mtd": _num(row.get("Connects MTD")),
            "connects_daily": _num(row.get("Connects Daily")),
        }

    # ---- 10b. unit ownership (1 Unit) — the "1 Unit" leaderboard's source.
    #          `Advisor.unit` was declared for this tab from the start
    #          (models.py) and the tab was never fetched, so the column
    #          was permanently NULL.
    #
    #          The tab carries BOTH `Unit` (a tally, observed 0-4) and
    #          `Count` (a flag, 1 where Unit > 0). The tally is stored
    #          because a flag is derivable from it and the reverse is not.
    #          Keyed by SAP ID — this tab has no WID column. ----
    for row in src.get("one_unit", []):
        res = ensure_advisor(row.get("SAP ID"), row.get("Advisor Name"))
        if not res:
            continue
        _, a = res
        _assign(a, "unit", row.get("Unit"))

    # ---- 10c. YTD mirrors. Same sheet layout as their MTD counterparts,
    #          so both reuse the shared mappers rather than restating the
    #          column names. Written to parallel `ytd_*` fields, never as
    #          period rows — see the SalesFunnel/Pipeline docstrings. ----
    for row in src.get("ytd_ccmc", []):
        res = ensure_advisor(*_identity(row))
        if not res:
            continue
        wid, _ = res
        sales_funnel.setdefault(wid, {"wid": wid})
        sales_funnel[wid].update(_funnel_fields(row, "ytd"))

    for row in src.get("ytd_p1_overdue", []):
        res = ensure_advisor(*_identity(row))
        if not res:
            continue
        wid, _ = res
        pipeline.setdefault(wid, {"wid": wid})
        pipeline[wid].update({
            "ytd_pipeline": _num(row.get("Pipeline")),
            # NOTE the header differs from the MTD tab: "P1 & Overdue"
            # calls it "Total Overdue", "YTD P1 & Overdue" calls it
            # "Overdue".
            "ytd_overdue": _num(row.get("Overdue")),
        })

    # ---- 11. team_targets (Target Achievement) — standalone, not keyed by wid ----
    team_targets = []
    for row in src.get("target_achievement", []):
        # normalize_org_name, not bare _clean: team_targets.team is joined
        # to advisors.team BY NAME (there's no FK), so the two sides must
        # be normalized identically or a whitespace variant here silently
        # becomes a team with "no target on file".
        team = normalize_org_name(_clean(row.get("Team")))
        if not team:
            continue
        team_targets.append({
            "team": team,
            # FIX: real column header in the "Target Achievement" tab is
            # "Targets" (plural) — "Target" always returned None, so this
            # value was silently loading as 0.0 for every row.
            "target": _num(row.get("Targets")),
            "achieved": _num(row.get("Total Achieved")),
            "achievement_pct": _num(row.get("Achievement %")),
        })

    return {
        "advisors": [a for a in advisors.values() if a.get("name")],
        "sales_funnel": list(sales_funnel.values()),
        "pipeline": list(pipeline.values()),
        "attendance": list(attendance.values()),
        "performance": performance,
        "portfolio": list(portfolio.values()),
        "bookings": list(bookings.values()),
        "calls": list(calls.values()),
        "team_targets": team_targets,
    }