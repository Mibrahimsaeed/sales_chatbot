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


def format_ir_leaderboard_reply(ir: QueryIR, rows: list[dict]) -> str:
    if not rows:
        return "No data available for that yet."
    metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
    label = _metric_label(metric_key)
    lines = [f"🏆 Top {len(rows)} by {label}{_filters_summary(ir)}", ""]
    for i, row in enumerate(rows, start=1):
        value = row.get("value", 0)
        lines.append(f"{i}. {row['name']} — {value:,.0f}")
        if row.get("team"):
            lines.append(f"   Team: {row['team']}")
        if row.get("company"):
            lines.append(f"   Company: {row['company']}")
        lines.append("")
    return "\n".join(lines)


def format_ir_comparison_reply(ir: QueryIR, rows: list[dict]) -> str:
    if not rows:
        return "I couldn't find data for those to compare."
    metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
    label = _metric_label(metric_key)
    lines = [f"📊 Comparing by {label}{_filters_summary(ir)}", ""]
    for row in rows:
        lines.append(f"• {row['name']}: {row.get('value', 0):,.0f}")
    return "\n".join(lines)


def format_ir_filtered_list_reply(ir: QueryIR, rows: list[dict]) -> str:
    if not rows:
        return "No results matched those conditions."
    metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
    label = _metric_label(metric_key) if metric_key else None
    lines = [f"{len(rows)} result(s){_filters_summary(ir)}:", ""]
    for row in rows:
        value_note = f" — {label}: {row.get('value', 0):,.0f}" if label else ""
        lines.append(f"• {row['name']}{value_note}")
    return "\n".join(lines)


_IR_FORMATTERS = {
    "leaderboard": format_ir_leaderboard_reply,
    "comparison": format_ir_comparison_reply,
    "filtered_list": format_ir_filtered_list_reply,
}


def format_ir_reply(ir: QueryIR, rows: list[dict]) -> str:
    formatter = _IR_FORMATTERS.get(ir.intent, format_ir_leaderboard_reply)
    return formatter(ir, rows)
