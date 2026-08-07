"""
Response formatting.

format_advisor_reply / format_team_reply / format_company_reply /
format_attendance_reply are unchanged — they still serve the rule-based
"lookup" / "summary" / "attendance_filter" plan path in chat_service.py,
which was intentionally left alone (see nlu_pipeline.py's docstring).

format_breakdown_reply serves the "breakdown" intent (unit_head/zonal_head/
business_center), nested by team instead of a single flat aggregate (see
hierarchy_service.get_level_breakdown) — reachable from both the rule-based
plan path and, since the hierarchy rework's phase 2, the LLM IR pipeline
(QueryIR.intent == "breakdown"). format_flat_breakdown_reply is the
explicit flat opt-in (QueryIR.flat / a "flat"/"list all" phrase).

format_ir_reply is new: it formats by QueryIR.intent shape (leaderboard /
comparison / filtered_list) rather than by an ad hoc response "type"
string scattered through chat_service.py, per Part 5.6 of the redesign —
a comparison needs a side-by-side template, a filtered_list needs a plain
list, a leaderboard needs a ranked list; each is one function here, not a
new pipeline stage.
"""

from app.llm.metric_ontology import METRICS, is_percentage_metric, metric_phrase
from app.llm.query_ir import QueryIR
from app.llm.response_planner import plan_response
from app.llm.query_compiler import effective_metric


def format_metric_value(metric_key: str | None, value) -> str:
    """One metric value, rendered with its unit.

    TWO defects in one function.

    CRASH. The leaderboard formatter did `f"{value:,.0f}"` directly, and
    a RATIO metric whose denominator sums to zero compiles to NULL by
    design — `aggregation.value_expression` documents that callers
    "render it as no data rather than as 0%". No caller did: one advisor
    with no recorded attendance made the ENTIRE reply raise TypeError, so
    a whole leaderboard failed because one row had nothing to divide by.

    UNITS. It also printed a bare number, so an achievement of 99% read
    as "99" — indistinguishable from a count. The metric already knows
    whether it is a percentage; nothing was asking it.

    Kept here rather than at each call site so a new formatter cannot
    reintroduce either half.
    """
    from app.llm.metric_ontology import is_percentage_metric

    if value is None:
        return "no data"
    if metric_key and is_percentage_metric(metric_key):
        return f"{value:,.1f}%".replace(".0%", "%")
    return f"{value:,.0f}"


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


def _qualified(label: str, value: str) -> str:
    """"Region North", but "Unit 1" rather than "Unit Unit 1".

    Some level values already carry their level's name — units are
    literally called "Unit 1", and a team can be named "Team Rashid
    Majeed". Prefixing the label again reads as a stutter. Comparing
    generically rather than special-casing a level means this holds for
    whatever levels exist, including ones added later."""
    label = label or ""
    value = str(value or "")
    if not label:
        return value
    if value.lower().startswith(label.lower()):
        return value
    return f"{label} {value}".strip()


def _breakdown_header(data: dict, group_note: str) -> str:
    """Shared top line for format_breakdown_reply/format_flat_breakdown_
    reply — same fields either way, `group_note` is the only thing that
    differs ("across N team(s)" vs a flat-list note)."""
    label = data.get("level_label") or "Group"
    value = data.get("value", "")
    return (
        f"{_qualified(label, value)} has {data.get('advisors', 0)} advisors{group_note}, "
        f"{data.get('connects', 0):.0f} MTD connects, and has cleared "
        f"{data.get('mtd_cleared', 0):,.0f} of a {data.get('mtd_target', 0):,.0f} target "
        f"({_pct(data.get('mtd_cleared', 0), data.get('mtd_target', 0))})."
        f"{_ytd_note(data)}"
    )


def format_person_disambiguation_reply(name: str, candidates: list) -> str:
    """Phase 1 identity refactor: several real people match this name, so
    ask which one instead of silently answering about the lowest WID.
    Each option carries team/company context, because the name alone is
    exactly what failed to distinguish them.

    Two distinct shapes of ambiguity share this reply, and the wording
    has to tell them apart honestly: several people with the SAME name
    ("8 people named Yasir Ali"), versus a near-tie between DIFFERENT
    names the query couldn't separate ("Ahmed Ali" vs "Ali Ahmed") —
    calling the latter "people named Ahmed Ali" would be plainly false."""
    distinct_names = {c.name for c in candidates}
    if len(distinct_names) == 1:
        header = f"I found multiple advisors named {candidates[0].name}."
    else:
        header = f"'{name}' matches more than one advisor."

    lines = [header, ""]
    for i, candidate in enumerate(candidates, start=1):
        lines.append(f"{i}. {candidate.name} — {_distinguishing_context(candidate, candidates)}")
    lines.append("")
    lines.append("Which one did you mean?")
    return "\n".join(lines)


def _distinguishing_context(candidate, candidates: list) -> str:
    """Context that actually tells this candidate apart from the others.

    Team alone is usually enough, but not always: production has 8 people
    named "Yasir Ali" of whom 6 share the team "North/KPK Region". Listing
    six identical lines asks a question the user cannot answer, which is
    no better than the silent guess this whole flow replaced. So the
    context escalates only as far as it needs to — team, then team +
    company, then the wid, which is unique by definition."""
    team = candidate.team or "no team on file"
    if sum(1 for c in candidates if (c.team or "no team on file") == team) == 1:
        return team

    with_company = f"{team} · {candidate.company}" if candidate.company else team
    if sum(
        1 for c in candidates
        if (f"{c.team or 'no team on file'} · {c.company}" if c.company else (c.team or "no team on file")) == with_company
    ) == 1:
        return with_company

    return f"{with_company} · ID {candidate.wid}"


ROSTER_PREVIEW_LIMIT = 40


def format_roster_reply(roster: dict) -> str:
    """A plain list of people — deliberately NOT the aggregate metrics an
    entity summary returns, because "all advisors in Blue Area" asks who
    they are, not how they're doing.

    Team is shown per advisor only when the roster spans more than one
    (so a team roster doesn't repeat the same team on every line, while a
    company or unit-head roster stays informative)."""
    label = roster.get("level_label") or "Group"
    value = roster.get("value", "")
    advisors = roster.get("advisors", [])
    count = roster.get("count", len(advisors))

    if not advisors:
        return f"I couldn't find any advisors in {_qualified(label, value)}."

    show_team = len({a.get("team") for a in advisors}) > 1
    lines = [f"{count} advisor(s) in {_qualified(label, value)}:", ""]
    for advisor in advisors[:ROSTER_PREVIEW_LIMIT]:
        suffix = f" — {advisor['team']}" if show_team and advisor.get("team") else ""
        lines.append(f"• {advisor['name']}{suffix}")

    remaining = count - min(count, ROSTER_PREVIEW_LIMIT)
    if remaining > 0:
        lines.append("")
        lines.append(f"…and {remaining} more.")
    return "\n".join(lines)


def format_comparison_reply(comparison: dict) -> str:
    """Side-by-side comparison — one row per KPI, one column per entity.

    Values are aligned in fixed-width columns so the numbers can actually
    be compared by eye; a bulleted list of "A: 5, B: 7" per metric is
    technically the same information and much harder to read across.
    The leader on each row is marked, since "which is better" is usually
    the actual question behind "compare A and B"."""
    entities = comparison.get("entities", [])
    rows = comparison.get("rows", [])
    winners = comparison.get("winners", {})
    if len(entities) < 2:
        return "I need two things to compare."

    names = [e["value"] for e in entities]
    # "on <measure>" only when the comparison is ABOUT one measure. With
    # several, every row is already labelled with its own, and naming
    # them again in the header would either repeat the table or — as it
    # did before this guard — render the raw key list into the sentence.
    metric_note = ""
    chosen = comparison.get("metric")
    if isinstance(chosen, str) and chosen:
        label = next((r["label"] for r in rows if r["key"] == chosen), chosen)
        metric_note = f" on {label}"

    # Column width must account for the widest VALUE as well as the
    # widest name, or a long figure (889,781,772) overflows its column
    # and shifts every column to its right out of alignment.
    def _rendered(entity, row) -> str:
        value = entity["metrics"].get(row["key"])
        if value is None:
            return "n/a"
        return f"{value:,.0f}%" if row["is_percentage"] else f"{value:,.0f}"

    label_width = max((len(r["label"]) for r in rows), default=10)
    widest_value = max(
        (len(_rendered(e, r)) for e in entities for r in rows), default=0
    )
    col_width = max(max(len(n) for n in names), widest_value, 12) + 2  # +2 for the winner marker

    header = "  ".join(n.ljust(col_width) for n in names)
    lines = [
        f"📊 Comparing {' vs '.join(names)}{metric_note}",
        "",
        " " * (label_width + 2) + header,
    ]

    for row in rows:
        cells = []
        for entity in entities:
            value = entity["metrics"].get(row["key"])
            if value is None:
                text = "n/a"
            elif row["is_percentage"]:
                text = f"{value:,.0f}%"
            else:
                text = f"{value:,.0f}"
            # pad BEFORE appending the marker — padding after it makes the
            # marked column wider than the others and skews every
            # subsequent column, which defeats the point of a table
            if winners.get(row["key"]) == entity["value"]:
                text = f"{text} ←"
            cells.append(text.ljust(col_width))
        lines.append(f"{row['label'].ljust(label_width)}  " + "  ".join(cells).rstrip())

    return "\n".join(lines)


def _bare_metric_clause(metric_key: str, value) -> str:
    """One measure WITHOUT the person's name — the second and later
    clauses of a multi-measure sentence, which already named them."""
    phrase = metric_phrase(metric_key)
    if value is None:
        return f"no {phrase} on file"
    if is_percentage_metric(metric_key):
        return f"{value:,.0f}% {phrase.replace('%', '').strip()}"
    return f"{_metric_number(value)} {phrase}"


def _one_metric_clause(name: str, metric_key: str, value) -> str:
    """One measure as a whole sentence, exactly as it always read."""
    phrase = metric_phrase(metric_key)
    if value is None:
        return f"I don't have {phrase} on file for {name}."
    if is_percentage_metric(metric_key):
        return f"{name} is at {value:,.0f}% {phrase.replace('%', '').strip()}."
    return f"{name} has {_metric_number(value)} {phrase}."


def format_advisor_metric_reply(name, answered, unavailable=None, period=None) -> str:
    """The measures one person was asked for.

    Deliberately still a sentence with nothing else in it. The full
    profile already answers this question in the sense of containing the
    numbers — the reason this exists is that containing them is not the
    same as answering them, so adding team, manager or targets "for
    context" would undo the point.

    `None` for a value means the person has no row in that metric's fact
    table, which is said plainly rather than rendered as a zero — zero is
    a real value and claiming it would be a wrong answer.

    `answered` is [(metric_key, value), ...]. A single pair produces
    byte-identical output to the one-metric version this replaces, which
    is the whole compatibility contract; several are joined into one
    sentence rather than listed, because "has 500 MTD connects and 200
    answered calls" is how the question was asked.

    `unavailable` names the measures that were REQUESTED and could not be
    served at the requested window. It is appended rather than dropped:
    silently returning the metrics that happened to work makes a partial
    answer indistinguishable from a complete one, which is the failure
    this signature exists to make impossible.
    """
    # The pre-Phase-13B call shape, kept working so nothing had to be
    # updated in step with this: format_advisor_metric_reply(name, key, value).
    if isinstance(answered, str):
        answered = [(answered, unavailable)]
        unavailable = None

    if len(answered) == 1:
        reply = _one_metric_clause(name, answered[0][0], answered[0][1])
    else:
        # Only the first clause names the person; the rest are bare
        # "N <measure>" so the whole thing reads as one answer rather than
        # as several replies about the same person.
        head = _one_metric_clause(name, answered[0][0], answered[0][1]).rstrip(".")
        tail = [_bare_metric_clause(key, value) for key, value in answered[1:]]
        parts = [head] + tail
        reply = f"{', '.join(parts[:-1])} and {parts[-1]}."

    if unavailable:
        from app.llm.metric_ontology import measure_label, supported_periods
        from app.llm.periods import label_for as period_label

        for key in unavailable:
            held = ", ".join(p.value for p in supported_periods(key)) or "no period"
            reply += (
                f" I don't have {period_label(period)} figures for "
                f"{measure_label(key)} — I hold {held} totals for it."
            )
    return reply


def _metric_number(value: float) -> str:
    """Counts read as integers, money keeps its separators, and a genuine
    fraction keeps one decimal — "2 MTD connects", not "2.0"."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def format_manager_reply(result: dict) -> str:
    """Reverse hierarchy answer — "who is X's BM/zonal head"."""
    return (
        f"{result['advisor']}'s {result['level_label']} is {result['manager']}."
    )


def format_breakdown_reply(breakdown: dict) -> str:
    """Nested-by-team reply for the new hierarchy levels (unit_head/
    zonal_head/business_center/company via hierarchy_service.get_level_
    breakdown) — decision: the default response for these levels is always
    nested by team, not a single flat aggregate number, since one Unit
    Head/Zonal Head/Business Center can span several teams."""
    teams = breakdown.get("teams", [])
    header = _breakdown_header(breakdown, f" across {len(teams)} team(s)")

    lines = [header, ""]
    for team in teams:
        lines.append(f"• {team['team']} ({team['advisor_count']} advisor(s))")
        for advisor in team["advisors"]:
            lines.append(
                f"   - {advisor['name']}: {advisor['connects']:.0f} connects, "
                f"{advisor['mtd_cleared']:,.0f} cleared"
            )
        lines.append("")

    return "\n".join(lines)


def format_flat_breakdown_reply(data: dict) -> str:
    """The explicit flat opt-in (hierarchy_service.get_level_flat_list) —
    same top line as format_breakdown_reply, an ungrouped advisor list
    (each with its own team shown inline) instead of nested team sections."""
    header = _breakdown_header(data, "")

    lines = [header, ""]
    for advisor in data.get("advisor_list", []):
        team_note = f" ({advisor['team']})" if advisor.get("team") else ""
        lines.append(
            f"• {advisor['name']}{team_note}: {advisor['connects']:.0f} connects, "
            f"{advisor['mtd_cleared']:,.0f} cleared"
        )

    return "\n".join(lines)


def format_leaderboard_reply(metric: str, rows: list[dict]) -> str:
    if not rows:
        return "No data available for that leaderboard yet."

    label = METRICS[metric].label if metric in METRICS else metric
    lines = [f"🏆 Top {len(rows)} by {label}", ""]

    for i, row in enumerate(rows, start=1):
        value = row.get("value", 0)
        lines.append(f"{i}. {row['name']} — {format_metric_value(metric_key, value)}")
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
    metric_key = effective_metric(ir)
    label = _metric_label(metric_key)
    if paginated:
        header = f"Showing {_shown_through(rows, start_index)} of {total_count} by {label}{_filters_summary(ir)}"
    else:
        header = f"🏆 Top {len(rows)} by {label}{_filters_summary(ir)}"
    lines = [header, ""]
    for i, row in enumerate(rows, start=start_index):
        value = row.get("value", 0)
        lines.append(f"{i}. {row['name']} — {format_metric_value(metric_key, value)}")
        if row.get("team"):
            lines.append(f"   Team: {row['team']}")
        if row.get("company"):
            lines.append(f"   Company: {row['company']}")
        lines.append("")
    return "\n".join(lines)


def format_ir_single_value_reply(
    ir: QueryIR, rows: list[dict], total_count: int | None = None, start_index: int = 1,
    paginated: bool = False, companion: tuple | None = None
) -> str:
    """One subject's figure, stated as a sentence.

    Phase 3. response_planner has returned shape="single_value" for a
    one-row result since Part 8, but _SHAPE_FORMATTERS mapped it to the
    LEADERBOARD formatter — so the plan was computed correctly and then
    discarded, and "What is Downtown revenue?" answered with

        🏆 Top 1 by MTD Revenue Cleared (filtered by team = Downtown)
        1. Downtown — 1,100

    A ranking of one is not a ranking. The figure was right; the KIND of
    answer was not.

    Paginated results keep the list rendering: pagination only happens
    over a set, so a paginated single row is a page of a longer list, not
    a subject's figure.
    """
    if not rows:
        return "No data available for that yet."
    if paginated:
        return format_ir_leaderboard_reply(
            ir, rows, total_count=total_count, start_index=start_index, paginated=True
        )

    row = rows[0]
    metric_key = effective_metric(ir)
    value = format_metric_value(metric_key, row.get("value", 0))

    # The narrowing the query applied, stated. Every other IR formatter
    # includes _filters_summary; this one omitted it, so a follow-up that
    # narrowed the scope read as if it had been ignored:
    #
    #   "Show Downtown pipeline"  -> "Downtown has 6,500 MTD Open Pipeline."
    #   "Now only Graana"         -> "Downtown has 6,500 MTD Open Pipeline."
    #
    # Both filters WERE applied (team=Downtown AND company=Graana, and
    # Downtown is entirely Graana, so the figure is genuinely unchanged) —
    # but an identical sentence is indistinguishable from a dropped
    # filter, which is the failure this reads as. The subject's own scope
    # filter is skipped: naming it twice ("Downtown has ... filtered by
    # team = Downtown") is noise, not information.
    scope = [f for f in ir.filters if f.value != row.get("name")]
    suffix = ""
    if scope:
        suffix = _filters_summary(ir.model_copy(update={"filters": scope}))

    # The paired measure, when the ontology declares one and the caller
    # fetched it. "How many CRs?" and "what is the CR rate?" are two
    # readings of one question, and answering only one leaves the obvious
    # follow-up unasked. The value arrives computed — this renders it.
    if companion is not None:
        companion_key, companion_value = companion
        if companion_value is not None:
            rendered = format_metric_value(companion_key, companion_value)
            # No article: the pair renders in both directions, and "a"
            # reads wrong before a count ("with a 20 Client
            # Registrations") even though it reads right before a rate.
            return (f"{row['name']} has {value} {_metric_label(metric_key)}, "
                    f"with {rendered} {_metric_label(companion_key)}{suffix}.")
    return f"{row['name']} has {value} {_metric_label(metric_key)}{suffix}."


def format_ir_comparison_reply(
    ir: QueryIR, rows: list[dict], total_count: int | None = None, start_index: int = 1, paginated: bool = False
) -> str:
    if not rows:
        return "I couldn't find data for those to compare."
    metric_key = effective_metric(ir)
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
    metric_key = effective_metric(ir)
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
    # Phase 3: single_value has its own renderer. It shared the
    # leaderboard's until now, which silently undid the planner's
    # decision — see format_ir_single_value_reply.
    "single_value": format_ir_single_value_reply,
    "ranked_list": format_ir_leaderboard_reply,
    "comparison_table": format_ir_comparison_reply,
    "filtered_table": format_ir_filtered_list_reply,
}


def format_ir_reply(
    ir: QueryIR, rows: list[dict], total_count: int | None = None, start_index: int = 1,
    paginated: bool = False, plan=None, companion: tuple | None = None
) -> str:
    """Render the response the planner chose.

    Phase 4: `plan` is supplied by the caller. This function used to call
    plan_response() itself while chat_service called it too, so the
    response mode was decided twice per request — and two callers of a
    decision are two places it can diverge, which is the shape of every
    defect these phases have removed. The formatter now RENDERS a plan
    rather than making one; the parameter stays optional so the module
    remains independently callable.
    """
    if plan is None:
        plan = plan_response(ir, rows)
    if plan.shape == "empty":
        formatter = _FORMATTER_BY_INTENT.get(ir.intent, format_ir_leaderboard_reply)
    else:
        formatter = _SHAPE_FORMATTERS.get(plan.shape, format_ir_leaderboard_reply)
    kwargs = {}
    if companion is not None and formatter is format_ir_single_value_reply:
        kwargs["companion"] = companion
    return formatter(ir, rows, total_count=total_count, start_index=start_index,
                     paginated=paginated, **kwargs)


def format_group_manager_reply(result: dict) -> str:
    """"Usman Ghani's Zonal Head is Fawad Hafeez."

    The plural branch is not defensive padding: if a group's advisors
    report to two different managers the chain is contradicted by the
    data, and naming both is the honest answer. Picking one would hide a
    data problem behind a confident sentence.
    """
    subject = f"{result['level_label']} {result['value']}"
    managers = result["managers"]
    label = result["target_level_label"]

    if len(managers) == 1:
        return f"{subject}'s {label} is {managers[0]}."
    joined = ", ".join(managers[:-1]) + f" and {managers[-1]}"
    return (
        f"{subject} spans more than one {label}: {joined}. "
        f"That usually means the source data disagrees with the reporting chain."
    )


def format_ancestry_reply(result: dict) -> str:
    """The whole reporting line, innermost first.

    Rendered as a chain rather than a sentence because that is the shape
    of the answer — four names in a list read as four unrelated facts.
    """
    header = f"Reporting line above {result['level_label']} {result['value']}:"
    lines = [header, ""]
    for depth, step in enumerate(result["ancestry"], start=1):
        lines.append(f"{'  ' * (depth - 1)}\u21b3 {step['level_label']}: {step['value']}")
    return "\n".join(lines)
