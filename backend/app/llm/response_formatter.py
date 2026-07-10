def _pct(cleared, target):
    if not target:
        return "n/a"
    return f"{(cleared / target * 100):.0f}%"


def format_advisor_reply(advisor: dict) -> str:
    name = advisor.get("name", "This advisor")
    connects = (advisor.get("mtd_new_connect") or 0) + (advisor.get("mtd_followup_connect") or 0)
    cleared = advisor.get("mtd_cleared") or 0
    target = advisor.get("mtd_target") or 0
    overdue = advisor.get("overdue") or 0
    overdue_note = f" They have {overdue:.0f} overdue items in pipeline." if overdue else ""
    return (
        f"{name} has {connects:.0f} MTD connects and has cleared {cleared:,.0f} "
        f"against a target of {target:,.0f} ({_pct(cleared, target)})."
        f"{overdue_note}"
    )


def format_team_reply(summary: dict) -> str:
    team = summary.get("team")
    advisors = summary.get("advisors", 0)
    cleared = summary.get("mtd_cleared") or 0
    target = summary.get("target")
    if target:
        return (
            f"{team} has {advisors} advisors, {summary.get('connects', 0):.0f} MTD connects, "
            f"and is at {_pct(summary.get('achieved') or cleared, target)} of target "
            f"({(summary.get('achieved') or cleared):,.0f} of {target:,.0f})."
        )
    return f"{team} has {advisors} advisors and {summary.get('connects', 0):.0f} MTD connects. No target on file for this team."


def format_company_reply(summary: dict) -> str:
    company = summary.get("company")
    return (
        f"{company} has {summary.get('advisors', 0)} advisors, "
        f"{summary.get('connects', 0):.0f} MTD connects, and has cleared "
        f"{summary.get('mtd_cleared', 0):,.0f} of a {summary.get('mtd_target', 0):,.0f} target "
        f"({_pct(summary.get('mtd_cleared', 0), summary.get('mtd_target', 0))})."
    )


def format_leaderboard_reply(metric: str, rows: list[dict]) -> str:
    if not rows:
        return "No data available for that leaderboard yet."
    label = {"mtd_cleared": "MTD revenue cleared", "mtd_new_connect": "MTD connects", "overdue": "overdue items"}.get(metric, metric)
    top = rows[0]
    return f"Top by {label}: {top['name']} ({top.get('value', 0):,.0f}). {len(rows)} advisors shown."


def format_attendance_reply(rows: list[dict]) -> str:
    if not rows:
        return "No attendance issues found — everyone's on time."
    names = ", ".join(r["name"] for r in rows[:5])
    more = f" and {len(rows) - 5} more" if len(rows) > 5 else ""
    return f"{len(rows)} advisor(s) with attendance issues: {names}{more}."


CLARIFICATION_PROMPTS = {
    "advisor_name": "which advisor did you mean",
    "team": "which team",
    "company": "which company \u2014 IMARAT, Graana, or Agency21",
    "metric": "which metric \u2014 revenue, connects, or overdue",
}


def format_clarification_reply(intent: str, missing_slots: list[str]) -> str:
    asks = [CLARIFICATION_PROMPTS.get(slot, slot) for slot in missing_slots]
    return "I need a bit more detail \u2014 " + " and ".join(asks) + "?"
def format_response(result: dict) -> str:
    """
    Main response formatter.

    Takes output from query_executor.py
    and converts it into a user-friendly reply.
    """

    query = result.get("query")
    data = result.get("data")


    # ----------------------------
    # Advisor Profile
    # ----------------------------
    if query == "advisor_profile":

        if not data:
            return "I could not find this advisor."

        return format_advisor_reply(data)



    # ----------------------------
    # Leaderboard
    # ----------------------------
    if query == "leaderboard":

        return format_leaderboard_reply(
            result.get("metric", "mtd_cleared"),
            data or []
        )



    # ----------------------------
    # Attendance
    # ----------------------------
    if query == "attendance_summary":

        return format_attendance_reply(
            data or []
        )



    # ----------------------------
    # Team Performance
    # ----------------------------
    if query == "team_performance":

        if not data:
            return "No team performance data found."

        return format_team_reply(data)



    # ----------------------------
    # Company Performance
    # ----------------------------
    if query == "company_performance":

        if not data:
            return "No company performance data found."

        return format_company_reply(data)



    # ----------------------------
    # Clarification
    # ----------------------------
    if query == "clarification":

        return format_clarification_reply(
            result.get("intent"),
            result.get("missing_slots", [])
        )



    # ----------------------------
    # Greeting
    # ----------------------------
    if query == "greeting":

        return (
            "Hello! I can help you with sales performance, "
            "advisor details, team analysis, company metrics, "
            "and attendance information."
        )



    # ----------------------------
    # Unknown
    # ----------------------------
    return (
        "I could not understand that request. "
        "Please ask about sales, performance, "
        "attendance, teams, companies, or advisors."
    )