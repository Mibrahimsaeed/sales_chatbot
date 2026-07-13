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


def _clean(value):
    """Normalize a raw cell value: strip whitespace, treat blank/placeholder as missing."""
    if value is None:
        return None
    value = str(value).strip()
    if value in ("", "-", "N/A", "n/a", "nan", "None"):
        return None
    return value


def _assign(a: dict, key: str, value):
    """Write `value` into advisor dict `a[key]` UNLESS value is empty, and
    UNLESS `a[key]` already holds a real (non-empty) value from a
    higher-priority source that ran earlier. This is the single choke point
    that replaces every ad-hoc `.update()` / `.setdefault()` call below, so
    field-loss like the team/company bug can't reoccur silently."""
    value = _clean(value)
    if value is None:
        return
    if not _clean(a.get(key)):
        a[key] = value


def transform(src: dict) -> dict:
    """src is the dict returned by etl.extract.extract_all().
    Returns {"advisors": [...], "sales_funnel": [...], ...} ready for load.py.
    """
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
        a = advisors.setdefault(wid, {"wid": wid})
        _assign(a, "name", name)
        return wid, a

    # ---- 1. MasterSheet FIRST — documented authoritative source for org
    #         hierarchy (Company / Regional / Teams / Portfolio Lead /
    #         Management Lead). Runs before CCMC DATA MTD so a blank cell
    #         downstream can never shadow a good value set here. ----
    for row in src["master_sheet"]:
        res = ensure_advisor(row.get("User ID"), row.get("Advisor Name"))
        if not res:
            continue
        _, a = res
        _assign(a, "company", row.get("Company"))
        _assign(a, "region", row.get("Regional"))
        _assign(a, "team", row.get("Teams"))
        _assign(a, "portfolio_lead", row.get("Portfolio Lead"))
        _assign(a, "management_lead", row.get("Management Lead"))

    # ---- 2. sales_funnel + org columns on advisors (CCMC DATA MTD) ----
    for row in src["ccmc_mtd"]:
        res = ensure_advisor(row.get("WID"), row.get("Name"))
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
        sales_funnel[wid] = {
            "wid": wid,
            "mtd_new_connect": _num(row.get("New Connect")),
            "mtd_followup_connect": _num(row.get("Follow-up Connect")),
            "mtd_cr": _num(row.get("CR")),
            "mtd_new_meeting": _num(row.get("New Meeting")),
            "mtd_followup_meeting": _num(row.get("Follow-up Meeting")),
            "mtd_todo": _num(row.get("Todo")),
            "mtd_booking_stored": _num(row.get("Booking Stored")),
            "mtd_conversion": _num(row.get("Conversion")),
        }

    # ---- 3. system-verified connect count layered onto sales_funnel (Connect Session) ----
    for row in src.get("connect_session", []):
        res = ensure_advisor(row.get("WID"), row.get("Name"))
        if not res:
            continue
        wid, _ = res
        sales_funnel.setdefault(wid, {"wid": wid})
        sales_funnel[wid]["system_connect"] = _num(row.get("Total Connect Through System"))

    # ---- 4. pipeline (P1 & Overdue) ----
    for row in src["p1_overdue"]:
        res = ensure_advisor(row.get("WID"), row.get("Name"))
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
        res = ensure_advisor(row.get("WID"), row.get("Advisor Name"))
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
        res = ensure_advisor(row.get("WID"), row.get("Advisor Name"))
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
        })

    # ---- 7. performance: one row per (wid, period) ----
    def perf_rows(rows, period: PerformancePeriod):
        for row in rows:
            res = ensure_advisor(row.get("WID"), row.get("Advisor Name"))
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
        res = ensure_advisor(row.get("WID"), row.get("Advisor Name"))
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
        res = ensure_advisor(row.get("WID"), row.get("Name"))
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
        res = ensure_advisor(row.get("WID"), row.get("Advisor Name"))
        if not res:
            continue
        wid, _ = res
        calls[wid] = {
            "wid": wid,
            "answered_calls_mtd": _num(row.get("Answered Calls MTD")),
            "answered_calls_daily": _num(row.get("Answered Calls Daily")),
            "connects_mtd": _num(row.get("Connects MTD")),
        }

    # ---- 11. team_targets (Target Achievement) — standalone, not keyed by wid ----
    team_targets = []
    for row in src.get("target_achievement", []):
        team = _clean(row.get("Team"))
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