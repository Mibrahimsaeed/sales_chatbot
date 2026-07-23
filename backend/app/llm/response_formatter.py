"""
Response formatting.

format_advisor_reply / format_team_reply / format_company_reply /
format_attendance_reply are unchanged — they still serve the rule-based
"lookup" / "summary" / "attendance_filter" plan path in chat_service.py,
which was intentionally left alone (see nlu_pipeline.py's docstring).

format_ir_reply is new: it formats by QueryIR.intent shape (leaderboard /
comparison / filtered_list) rather than by an ad hoc response "type"
string scattered through chat_service.py, per Part 5.6 of the redesign —
a comparison needs a side-by-side template, a filtered_list needs a plain
list, a leaderboard needs a ranked list; each is one function here, not a
new pipeline stage.
"""

from app.llm.metric_ontology import METRICS
from app.llm.query_ir import QueryIR
from app.llm.response_planner import plan_response


def _pct(cleared, target):
    if not target:
        return "n/a"
    return f"{(cleared / target * 100):.0f}%"


def format_advisor_reply(advisor: dict) -> str:
    name = advisor.get("name", "This advisor")
    connects = (advisor.get("mtd_new_connect") or 0) + (advisor.get("mtd_followup_connect") or 0)
    mtd_cleared = advisor.get("mtd_cleared") or 0
    mtd_target = advisor.get("mtd_target") or 0
    ytd_cleared = advisor.get("ytd_cleared") or 0
    ytd_target = advisor.get("ytd_target") or 0
    overdue = advisor.get("overdue") or 0

    # bug fix: advisor_profile already carries ytd_cleared/ytd_target (see
    # migrations/0002advisorprofileview.py) but this reply used to only
    # ever mention MTD, silently dropping YTD regardless of what was
    # asked. Shown whenever there's a nonzero YTD figure to report.
    ytd_note = ""
    if ytd_cleared or ytd_target:
        ytd_note = (
            f" Year to date, they've cleared {ytd_cleared:,.0f} against a YTD target of "
            f"{ytd_target:,.0f} ({_pct(ytd_cleared, ytd_target)})."
        )

    overdue_note = f" They have {overdue:.0f} overdue items in pipeline." if overdue else ""
    return (
        f"{name} has {connects:.0f} MTD connects and has cleared {mtd_cleared:,.0f} "
        f"against a target of {mtd_target:,.0f} ({_pct(mtd_cleared, mtd_target)})."
        f"{ytd_note}{overdue_note}"
    )


def _ytd_note(summary: dict) -> str:
    # bug fix: team/company summaries used to never mention YTD at all —
    # get_team_summary/get_company_summary now roll up YTD Performance
    # too, shown here whenever there's a nonzero figure to report.
    ytd_cleared = summary.get("ytd_cleared") or 0
    ytd_target = summary.get("ytd_target") or 0
    if not ytd_cleared and not ytd_target:
        return ""
    return (
        f" Year to date, they've cleared {ytd_cleared:,.0f} against a YTD target of "
        f"{ytd_target:,.0f} ({_pct(ytd_cleared, ytd_target)})."
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
            f"{_ytd_note(summary)}"
        )
    return (
        f"{team} has {advisors} advisors and {summary.get('connects', 0):.0f} MTD connects. "
        f"No target on file for this team.{_ytd_note(summary)}"
    )


def format_company_reply(summary: dict) -> str:
    company = summary.get("company")
    return (
        f"{company} has {summary.get('advisors', 0)} advisors, "
        f"{summary.get('connects', 0):.0f} MTD connects, and has cleared "
        f"{summary.get('mtd_cleared', 0):,.0f} of a {summary.get('mtd_target', 0):,.0f} target "
        f"({_pct(summary.get('mtd_cleared', 0), summary.get('mtd_target', 0))})."
        f"{_ytd_note(summary)}"
    )


def format_leaderboard_reply(metric: str, rows: list[dict]) -> str:
    if not rows:
        return "No data available for that leaderboard yet."

    label = METRICS[metric].label if metric in METRICS else metric
    lines = [f"🏆 Top {len(rows)} by {label}", ""]

    for i, row in enumerate(rows, start=1):
        value = row.get("value", 0)
        lines.append(f"{i}. {row['name']} — {value:,.0f}")
        if row.get("team"):
            lines.append(f"   Team: {row['team']}")
        if row.get("company"):
            lines.append(f"   Company: {row['company']}")
        lines.append("")

    return "\n".join(lines)


def format_attendance_reply(rows: list[dict]) -> str:
    if not rows:
        return "No attendance issues found — everyone's on time."
    names = ", ".join(r["name"] for r in rows[:5])
    more = f" and {len(rows) - 5} more" if len(rows) > 5 else ""
    return f"{len(rows)} advisor(s) with attendance issues: {names}{more}."


CLARIFICATION_PROMPTS = {
    "advisor_name": "which advisor did you mean",
    "team": "which team",
    "company": "which company — IMARAT, Graana, or Agency21",
    "metric": "which metric — revenue, connects, or overdue",
}


def format_clarification_reply(intent: str, missing_slots: list[str]) -> str:
    asks = [CLARIFICATION_PROMPTS.get(slot, slot) for slot in missing_slots]
    return "I need a bit more detail — " + " and ".join(asks) + "?"


# ---------------------------------------------------------------------
# QueryIR-shaped formatting (Part 5.6)
# ---------------------------------------------------------------------

def _metric_label(metric_key: str | None) -> str:
    if not metric_key:
        return "value"
    return METRICS[metric_key].label if metric_key in METRICS else metric_key


def _filters_summary(ir: QueryIR) -> str:
    if not ir.filters:
        return ""
    parts = []
    for f in ir.filters:
        label = _metric_label(f.field) if f.field in METRICS else f.field
        parts.append(f"{label} {f.operator} {f.value}")
    return " (filtered by " + ", ".join(parts) + ")" if parts else ""


def _shown_through(rows: list[dict], start_index: int) -> int:
    return start_index - 1 + len(rows)


def format_ir_leaderboard_reply(
    ir: QueryIR, rows: list[dict], total_count: int | None = None, start_index: int = 1, paginated: bool = False
) -> str:
    if not rows:
        return "No data available for that yet."
    metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
    label = _metric_label(metric_key)
    if paginated:
        header = f"Showing {_shown_through(rows, start_index)} of {total_count} by {label}{_filters_summary(ir)}"
    else:
        header = f"🏆 Top {len(rows)} by {label}{_filters_summary(ir)}"
    lines = [header, ""]
    for i, row in enumerate(rows, start=start_index):
        value = row.get("value", 0)
        lines.append(f"{i}. {row['name']} — {value:,.0f}")
        if row.get("team"):
            lines.append(f"   Team: {row['team']}")
        if row.get("company"):
            lines.append(f"   Company: {row['company']}")
        lines.append("")
    return "\n".join(lines)


def format_ir_comparison_reply(
    ir: QueryIR, rows: list[dict], total_count: int | None = None, start_index: int = 1, paginated: bool = False
) -> str:
    if not rows:
        return "I couldn't find data for those to compare."
    metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
    label = _metric_label(metric_key)
    if paginated:
        header = f"Showing {_shown_through(rows, start_index)} of {total_count} comparing by {label}{_filters_summary(ir)}"
    else:
        header = f"📊 Comparing by {label}{_filters_summary(ir)}"
    lines = [header, ""]
    for row in rows:
        lines.append(f"• {row['name']}: {row.get('value', 0):,.0f}")
    return "\n".join(lines)


def format_ir_filtered_list_reply(
    ir: QueryIR, rows: list[dict], total_count: int | None = None, start_index: int = 1, paginated: bool = False
) -> str:
    if not rows:
        return "No results matched those conditions."
    metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
    label = _metric_label(metric_key) if metric_key else None
    if paginated:
        header = f"Showing {_shown_through(rows, start_index)} of {total_count} result(s){_filters_summary(ir)}"
    else:
        header = f"{len(rows)} result(s){_filters_summary(ir)}:"
    lines = [header, ""]
    for row in rows:
        value_note = f" — {label}: {row.get('value', 0):,.0f}" if label else ""
        lines.append(f"• {row['name']}{value_note}")
    return "\n".join(lines)


# Kept for the empty-rows case (Part 8): plan_response() collapses to
# shape="empty" whenever there are no rows, but each formatter already
# has the right no-data message for ITS intent ("I couldn't find data for
# those to compare" vs "No results matched those conditions") — so an
# empty result still dispatches by intent, not by the (now-uninformative)
# shape.
_FORMATTER_BY_INTENT = {
    "leaderboard": format_ir_leaderboard_reply,
    "comparison": format_ir_comparison_reply,
    "filtered_list": format_ir_filtered_list_reply,
}

# Part 8: explicit response-shape planning (response_planner.py) replaces
# dispatching by ir.intent alone — a leaderboard that resolved to exactly
# one row reads as a single value, not a "Top 1" list. Same formatter
# functions as before, just re-keyed by shape.
_SHAPE_FORMATTERS = {
    "single_value": format_ir_leaderboard_reply,
    "ranked_list": format_ir_leaderboard_reply,
    "comparison_table": format_ir_comparison_reply,
    "filtered_table": format_ir_filtered_list_reply,
}


def format_ir_reply(
    ir: QueryIR, rows: list[dict], total_count: int | None = None, start_index: int = 1, paginated: bool = False
) -> str:
    plan = plan_response(ir, rows)
    if plan.shape == "empty":
        formatter = _FORMATTER_BY_INTENT.get(ir.intent, format_ir_leaderboard_reply)
    else:
        formatter = _SHAPE_FORMATTERS.get(plan.shape, format_ir_leaderboard_reply)
    return formatter(ir, rows, total_count=total_count, start_index=start_index, paginated=paginated)
