import json

from sqlalchemy.orm import Session

from app.llm import conversation_memory, hierarchy, narrative, routing
from app.llm.ir_validator import confidence_breakdown
from app.llm.nlu_pipeline import resolve, Resolution
from app.llm.metric_ontology import metric_for_period
from app.llm.query_compiler import compile_and_run, count_ir
from app.llm.query_ir import QueryIR
from app.llm.response_planner import plan_response, respond
from app.llm.response_formatter import (
    format_advisor_reply,
    format_advisor_metric_reply,
    format_team_reply,
    format_company_reply,
    format_ir_reply,
    format_attendance_reply,
    format_breakdown_reply,
    format_flat_breakdown_reply,
    format_person_disambiguation_reply,
    format_manager_reply,
    format_group_manager_reply,
    format_ancestry_reply,
    format_roster_reply,
    format_comparison_reply,
)
from app.services import (
    advisor_service,
    team_service,
    company_service,
    attendance_service,
    hierarchy_service,
    comparison_service,
)
from app.core import audit, tracing
from app.core.exception import NotFoundError
from app.core.logger import get_logger
from app.database.models import ChatLog

log = get_logger("services.chat_service")

# Part 8: pagination. A result set bigger than this shows only the first
# page plus a "Show More" control instead of dumping everything into one
# reply — see nlu_pipeline.py's typed "show more" recognition and the new
# POST /chat/more endpoint (app/api/chat.py) for the two ways a
# continuation gets triggered; both end up calling handle_show_more() below.
PAGE_SIZE = 15


def _capped_total(ir: QueryIR, true_total: int) -> int:
    """The total pagination should count toward: the true DB match count,
    unless the user explicitly asked for a bounded "top N" (ir.limit),
    in which case N is the ceiling even if more rows technically match."""
    return min(true_total, ir.limit) if ir.limit is not None else true_total


def _page_ir(ir: QueryIR, offset: int, capped_total: int) -> QueryIR:
    """A COPY of ir with limit shrunk to exactly this page's size — never
    mutates the stored ir, since later pages (and "Show More") still need
    its original limit/filters/sort untouched."""
    take = min(PAGE_SIZE, max(capped_total - offset, 0))
    return ir.model_copy(update={"limit": take})


def handle_chat_message(
    db: Session,
    message: str,
    session_id: str | None = None,
) -> dict:
    """Phase 7: the whole message is wrapped in a trace, so one structured
    log line carries the full decision chain (entities -> identity ->
    plan -> IR -> SQL -> rows -> response). The audit's bugs were all
    diagnosed by hand-instrumenting this path; production had no
    equivalent record.

    The audit_query() wrapper is the human-readable counterpart (see
    app/core/audit.py): same span, but it captures the complete LLM
    prompts verbatim and writes a readable block per query. Off unless
    CHAT_AUDIT_DEBUG is set, and it changes nothing about the answer."""
    with audit.audit_query(message, session_id=session_id):
        with tracing.traced(message, session_id=session_id) as trace:
            resolution = resolve(message, db, session_id=session_id)
            log.debug(f"resolved '{message}' -> kind={resolution.kind}")

            # entities + plan are recorded upstream in nlu_pipeline, where the
            # decision is actually made (a clarification never reaches here as
            # a plan); only the IR is added at this level.
            tracing.record_ir(resolution.ir)

            response = _dispatch(db, resolution, session_id)
            tracing.record_response(response)
            audit.record_response(response)
            _log_interaction(db, session_id, message, resolution, response, trace)
            return response


def handle_show_more(db: Session, session_id: str | None) -> dict:
    """Audit wrapper around _show_more() — behaviour is _show_more()'s,
    unchanged.

    It exists because the button-click entry point (POST /chat/more)
    never passes through handle_chat_message, so without this a "Show
    More" would produce a response with no audit block. The synthetic
    "[show more]" query stands in for the message the user didn't type.
    When this is instead reached via _dispatch's kind=="paginate",
    audit_query() is already open and nests into a no-op — one user
    message always yields exactly one audit block."""
    with audit.audit_query("[show more]", session_id=session_id):
        response = _show_more(db, session_id)
        audit.record_response(response)
        return response


def _show_more(db: Session, session_id: str | None) -> dict:
    """Part 8: the single implementation behind both "Show More" triggers
    — the dedicated POST /chat/more endpoint (button click, no re-parse)
    and nlu_pipeline recognizing typed "show more" (kind="paginate").
    Fetches the NEXT page only; the caller (frontend) appends it to what
    it already has, per the pagination cursor in conversation_memory."""
    state = conversation_memory.get_pagination(session_id)
    if state is None:
        return respond("text", "There's nothing more to show right now — try asking a new question.", None)

    # capture these BEFORE advance_pagination/clear_pagination — state is a
    # mutable reference into the stored PaginationState (get_pagination
    # doesn't copy it), so mutating it in place would silently change
    # what state.offset means out from under the rest of this function
    page_offset = state.offset
    ir = state.ir
    capped_total = state.capped_total

    page_ir = _page_ir(ir, page_offset, capped_total)
    rows = compile_and_run(db, page_ir, offset=page_offset)
    new_shown = page_offset + len(rows)
    has_more = capped_total > new_shown

    if has_more:
        conversation_memory.advance_pagination(session_id, new_shown)
    else:
        conversation_memory.clear_pagination(session_id)

    # Phase 4: the planner owns the mode here too. This returned
    # `ir.intent` — the second surviving place the QUESTION's shape was
    # reported as the ANSWER's kind, and easy to miss because a "show
    # more" never re-parses.
    response_plan = plan_response(ir, rows)
    reply = format_ir_reply(
        ir, rows, total_count=capped_total, start_index=page_offset + 1,
        paginated=True, plan=response_plan,
    )
    return respond(response_plan.mode, reply, rows, why=response_plan.trace(),
                   total_count=capped_total, shown_count=new_shown, has_more=has_more)


def _dispatch_breakdown(db: Session, level: str, value: str, flat: bool = False) -> dict:
    """Shared by the rule-based plan.action=="breakdown" path AND the LLM-
    driven QueryIR.intent=="breakdown" path (hierarchy rework phase 2) —
    same NotFoundError handling and formatter dispatch either way, so a
    phrasing that happens to reach the LLM instead of matching the simple
    rule-based bare-mention pattern still gets the exact same nested-by-
    team (or flat, if requested) response instead of a different one."""
    try:
        if flat:
            data = hierarchy_service.get_level_flat_list(db, level, value)
            audit.record_formatter("format_flat_breakdown_reply",
                                   f"flat=True — an ungrouped list of {level} {value!r}")
            reply = format_flat_breakdown_reply(data)
        else:
            data = hierarchy_service.get_level_breakdown(db, level, value)
            audit.record_formatter("format_breakdown_reply",
                                   f"flat=False — {level} {value!r} nested by team")
            reply = format_breakdown_reply(data)
    except NotFoundError:
        return respond("not_found", f"I couldn't find a {level.replace('_', ' ')} matching '{value}' — mind double-checking the spelling?", None)
    return respond("breakdown", reply, data)


def _dispatch(db: Session, resolution: Resolution, session_id: str | None = None) -> dict:
    if resolution.kind == "shortcut":
        return _dispatch_shortcut(db, resolution.shortcut_intent, resolution.entities)

    if resolution.kind == "multi":
        return _dispatch_multi(db, resolution, session_id)

    if resolution.kind == "paginate":
        return handle_show_more(db, session_id)

    if resolution.kind == "unsupported":
        # Distinct from a clarification: no answer the user could give
        # would make this answerable, so offering options would be a
        # false promise. No `options` key for the same reason.
        return respond("unsupported", resolution.clarify_message, None)

    if resolution.kind == "clarify":
        return respond("clarification", resolution.clarify_message, None, options=resolution.clarify_options or [])

    if resolution.kind == "ir" and resolution.ir.intent == "breakdown":
        # Never a metric-ranking operation (no compile_and_run/count_ir/
        # pagination) — ir_validator.validate_ir guarantees >= 1 grounded
        # subject before this IR is ever valid, so subjects[0] is safe here.
        subject = resolution.ir.subjects[0]
        return _dispatch_breakdown(
            db, resolution.ir.subject_level, subject.resolved_id or subject.value, resolution.ir.flat
        )

    if resolution.kind == "ir":
        return _dispatch_ir(db, resolution, session_id)

    # kind == "plan" — lookup / summary / attendance_filter, unchanged from
    # the previous design (see nlu_pipeline.py's docstring for why these
    # stayed on the simpler rule-based path).
    plan = resolution.plan

    if plan.action == "comparison_incomplete":
        # A comparison was asked for but only one side grounded. Naming
        # the side we DID find is what makes this actionable — the usual
        # cause is a typo or an entity that doesn't exist.
        return respond("clarification", (
            f"I could only find '{plan.entity_value}' — I need two things to compare. "
            "Could you check the other name?"
        ), None)

    if plan.action == "comparison":
        # "Compare Graana and Agency21" — side by side. Previously only
        # reachable via the LLM semantic parser, so with the LLM
        # unavailable this fell through to the metric-help message.
        try:
            comparison = comparison_service.get_comparison(
                db, plan.comparison_targets, metric=plan.metric
            )
        except NotFoundError as e:
            return respond("not_found", str(e.message), None)
        except ValueError:
            return respond("clarification", "I need two things to compare — try 'compare Graana and Agency21'.", None)
        audit.record_formatter("format_comparison_reply",
                               f"plan.action='comparison' over {plan.comparison_targets} "
                               f"on metric {plan.metric!r}")
        return respond("comparison", format_comparison_reply(comparison), comparison)

    if plan.action == "roster":
        # "All advisors in Blue Area" — enumerate the people. Previously
        # fell through to the entity-summary branch and answered with
        # connects/pipeline/overdue totals, which is a different question.
        try:
            roster = hierarchy_service.get_level_roster(db, plan.level, plan.entity_value)
        except NotFoundError:
            return respond("not_found", f"I couldn't find a {plan.level.replace('_', ' ')} matching '{plan.entity_value}'.", None)
        audit.record_formatter("format_roster_reply",
                               f"plan.action='roster' — flat enumeration of people in "
                               f"{plan.level} {plan.entity_value!r}")
        return respond("roster", format_roster_reply(roster), roster)

    if plan.action == "ancestry":
        # "The full hierarchy above X" — every level up, not one.
        chain = hierarchy_service.get_ancestry(db, plan.level, plan.entity_value)
        if not chain:
            label = hierarchy.label_for(plan.level)
            return respond("not_found",
                           f"I don't have a reporting line on file for {label.lower()} "
                           f"'{plan.entity_value}'.", None)
        audit.record_formatter("format_ancestry_reply",
                               f"plan.action='ancestry' — {len(chain['ancestry'])} levels "
                               f"above {plan.level} {plan.entity_value!r}")
        return respond("ancestry", format_ancestry_reply(chain), chain)

    if plan.action == "reverse_hierarchy" and plan.subject_level:
        # Phase 5.4: the subject is a GROUP (a BCM, a zonal head), so
        # there is no wid to key on — the manager is read off the
        # advisors in that group. See hierarchy_service.get_manager_of_group.
        result = hierarchy_service.get_manager_of_group(
            db, plan.subject_level, plan.entity_value, plan.level
        )
        if not result:
            label = hierarchy.label_for(plan.level)
            subject_label = hierarchy.label_for(plan.subject_level)
            return respond("not_found",
                           f"I don't have a {label} on file for {subject_label.lower()} "
                           f"'{plan.entity_value}'.", None)
        audit.record_formatter("format_group_manager_reply",
                               f"plan.action='reverse_hierarchy' with a {plan.subject_level} "
                               f"subject — the level ABOVE it is {plan.level!r}")
        return respond("manager", format_group_manager_reply(result), result)

    if plan.action == "reverse_hierarchy":
        # "Who is X's BM?" — the person ABOVE X, keyed by X's wid.
        if plan.entity_wid is None:
            return respond("not_found", f"I couldn't find anyone matching '{plan.entity_value}' — mind double-checking the spelling?", None)
        result = hierarchy_service.get_manager_of(db, plan.entity_wid, plan.level)
        if not result:
            # The advisor exists but has no manager recorded at this level.
            # Said plainly rather than answered with something else — the
            # audit's point that a capability gap must not silently become
            # a different, confident answer.
            label = hierarchy.label_for(plan.level)
            return respond("not_found", f"I don't have a {label} on file for {plan.entity_value}.", None)
        audit.record_formatter("format_manager_reply",
                               f"plan.action='reverse_hierarchy' — the one person ABOVE "
                               f"wid={plan.entity_wid} at level {plan.level!r}")
        return respond("manager", format_manager_reply(result), result)

    if plan.action == "clarify_person":
        # Phase 1: a name matching several real people asks which one,
        # instead of the old behavior of silently returning whichever had
        # the lowest wid.
        return respond("clarification", format_person_disambiguation_reply(plan.entity_value, plan.person_candidates), None, options=[c.label() for c in plan.person_candidates])

    if plan.action == "advisor_metric":
        # M7: the user named a metric, so answer with that metric alone.
        # Identity is resolved exactly as the profile lookup resolves it —
        # by wid when there is one, re-resolved through the same resolver
        # otherwise — so a metric question can no more return the wrong
        # person's number than a profile question can.
        if plan.entity_wid is not None:
            advisor = advisor_service.find_advisor_by_wid(db, plan.entity_wid)
        else:
            resolution = advisor_service.resolve_advisor(db, plan.entity_value)
            if resolution.is_ambiguous:
                return respond("clarification", format_person_disambiguation_reply(plan.entity_value, resolution.candidates), None, options=[c.label() for c in resolution.candidates])
            advisor = (
                advisor_service.find_advisor_by_wid(db, resolution.wid)
                if resolution.is_resolved else None
            )

        if not advisor:
            return respond("not_found", f"I couldn't find anyone matching '{plan.entity_value}' — mind double-checking the spelling?", None)

        value = advisor_service.get_advisor_metric(db, advisor["wid"], plan.metric)
        audit.record_formatter(
            "format_advisor_metric_reply",
            f"plan.action='advisor_metric' metric={plan.metric!r} for wid={advisor['wid']} — "
            "one number, not the profile",
        )
        return respond(
            "advisor_metric",
            format_advisor_metric_reply(advisor["name"], plan.metric, value),
            {"wid": advisor["wid"], "name": advisor["name"],
             "metric": plan.metric, "value": value},
        )

    if plan.action == "lookup":
        # Identity-first: fetch by WID when entity resolution produced one
        # (exact by construction — it cannot return a different person).
        # A plan built without a wid (e.g. an older cached IR) is
        # re-resolved through the SAME resolver rather than falling back
        # to a name query, so there is no path left that can silently pick
        # among several people.
        if plan.entity_wid is not None:
            advisor = advisor_service.find_advisor_by_wid(db, plan.entity_wid)
        else:
            resolution = advisor_service.resolve_advisor(db, plan.entity_value)
            if resolution.is_ambiguous:
                return respond("clarification", format_person_disambiguation_reply(plan.entity_value, resolution.candidates), None, options=[c.label() for c in resolution.candidates])
            advisor = (
                advisor_service.find_advisor_by_wid(db, resolution.wid)
                if resolution.is_resolved else None
            )

        if not advisor:
            return respond("not_found", f"I couldn't find anyone matching '{plan.entity_value}' — mind double-checking the spelling?", None)
        audit.record_formatter(
            "format_advisor_reply",
            f"plan.action='lookup' for {plan.entity_value!r} (wid={plan.entity_wid}) — ONE "
            "person's own profile row. This formatter emits that person's own metrics "
            "only; it has no team-roster or manager section, so anything the query asked "
            "about beyond this person is not represented in the reply.",
        )
        return respond("profile", format_advisor_reply(advisor), advisor)

    if plan.action == "summary" and plan.level == "team":
        try:
            summary = team_service.get_team_summary(db, plan.entity_value)
        except NotFoundError:
            return respond("not_found", f"I couldn't find a team matching '{plan.entity_value}'.", None)
        audit.record_formatter("format_team_reply",
                               f"plan.action='summary' with level='team' — aggregate "
                               f"performance for {plan.entity_value!r}, not its roster")
        return respond("hierarchy_summary", format_team_reply(summary), summary)

    if plan.action == "summary" and plan.level == "company":
        try:
            summary = company_service.get_company_summary(db, plan.entity_value)
        except NotFoundError:
            return respond("not_found", f"I couldn't find a company matching '{plan.entity_value}'.", None)
        audit.record_formatter("format_company_reply",
                               f"plan.action='summary' with level='company' — aggregate "
                               f"performance for {plan.entity_value!r}")
        return respond("company_summary", format_company_reply(summary), summary)

    if plan.action == "breakdown":
        return _dispatch_breakdown(db, plan.level, plan.entity_value, plan.flat)

    if plan.action == "attendance_filter":
        rows = attendance_service.get_attendance_by_status(
            db=db,
            team=plan.entity_value,
            status=plan.reason,
        )
        audit.record_formatter("format_attendance_reply",
                               f"plan.action='attendance_filter' status={plan.reason!r} "
                               f"team={plan.entity_value!r}")
        return respond("attendance", format_attendance_reply(rows), rows)

    audit.record_formatter(
        "(none — canned text)",
        f"plan.action={plan.action!r} matched no dispatch branch — the query reached the "
        "planner but nothing here can answer that action, so the reply is the generic "
        "'could you rephrase' text",
    )
    return respond("no_data", "Hmm, I'm not quite sure how to answer that one yet — could you rephrase it?", None)


def _unanswerable_reply(ir) -> str:
    """Why this (measure, period, level) has no answer.

    The period is named FIRST when it is the reason, because that is the
    part the user can act on: "revenue today" fails because there are no
    daily figures, not because revenue is unsupported. The old wording
    blamed the metric in every case and quoted its raw key
    ("mtd_cleared"), which read as though the measure itself were
    unknown.
    """
    from app.llm.metric_ontology import metric_label, supported_periods

    named = ir.sort.metric or (ir.metric.key if ir.metric else None)
    label = metric_label(named) if named else "that metric"
    period = getattr(ir.time_range, "period", None)

    if named and period and metric_for_period(named, period) is None:
        available = ", ".join(p.value for p in supported_periods(named)) or "no period"
        window = {"DAILY": "daily", "MTD": "month-to-date",
                  "YTD": "year-to-date", "3M": "3-month"}.get(period, period)
        return (
            f"I don't have {window} figures for {label} yet — I hold "
            f"{available} totals for it. Ask for one of those and I can answer."
        )

    return f"I don't have a way to answer that for {label} at the {ir.subject_level} level yet."


def _dispatch_ir(db: Session, resolution: Resolution, session_id: str | None = None) -> dict:
    """New path (Part 4/5.5): any query the generic compiler can answer —
    leaderboards, comparisons, and filtered/thresholded/boolean-combined
    queries — regardless of whether the QueryIR came from the rule-based
    fast path or the LLM Semantic Parser."""
    ir = resolution.ir
    true_total = count_ir(db, ir)

    if true_total is None:
        return respond("no_data", _unanswerable_reply(ir), None)

    # Part 8: cap the first page at PAGE_SIZE regardless of how many rows
    # actually match — a "Show More" cursor picks up the rest instead of
    # dumping hundreds of rows into one reply.
    capped_total = _capped_total(ir, true_total)
    rows = compile_and_run(db, _page_ir(ir, 0, capped_total), offset=0)
    has_more = capped_total > len(rows)

    if has_more:
        conversation_memory.set_pagination(session_id, ir, offset=len(rows), capped_total=capped_total)
    else:
        conversation_memory.clear_pagination(session_id)

    # THE response-mode decision, made once, here, from the IR and the
    # result set. Everything below renders it.
    response_plan = plan_response(ir, rows)
    routing.decide("Response", response_plan.mode, response_plan.trace())

    if response_plan.mode == "unsupported":
        # A capability limit is not an empty result. Saying so plainly
        # beats running a neighbouring query and presenting it as the
        # answer, which is what happened while this state was unreachable.
        return respond("unsupported", response_plan.reason, None)

    audit.record_formatter("format_ir_reply",
                           f"QueryIR path, intent={ir.intent!r} -> response mode "
                           f"{response_plan.mode!r} — compiled and executed "
                           "through query_compiler")
    reply = format_ir_reply(ir, rows, total_count=capped_total,
                            paginated=has_more, plan=response_plan)
    insights: list[str] = []
    if rows:
        # Part 11: evidence-aware explanation — 100% deterministic (every
        # number traced to `rows`), prepended ahead of the raw templated
        # list so the reply says WHY the answer is correct (ranking
        # justification, percentage interpretation, comparison
        # explanation) instead of just restating a value. polish_
        # explanation() may lightly smooth its phrasing but can't
        # originate or change a number — fails soft back to the
        # deterministic sentence unchanged. Facts/explanation/insights are
        # computed over the DISPLAYED page only, same as before pagination
        # existed — not the full (possibly 500+ row) match set.
        facts = narrative.compute_facts(ir, rows)
        explanation = narrative.build_explanation(ir, rows, total_count=capped_total)
        explanation = narrative.polish_explanation(explanation, facts)
        if explanation and response_plan.show_explanation:
            reply = f"{explanation}\n\n{reply}"
        # Part 8/11: only attach insights when the response planner judges
        # the result set large enough for "outlier"/"trend" to mean
        # anything (2-row comparisons don't have a meaningful peer group).
        # Anomaly detection and trend deltas are independent evidence
        # sources — both capped, combined cap keeps the reply concise.
        if response_plan.show_insights:
            insights = (narrative.compute_insights(ir, rows) + narrative.compute_trends(ir, rows, db))[:3]
    # Phase 3 made this the RESPONSE mode rather than the QUERY intent;
    # Phase 4 routes it through respond() like every other exit, so the
    # dispatcher has exactly one way out.
    return respond(
        response_plan.mode, reply, rows, why=response_plan.trace(),
        insights=insights,
        confidence=confidence_breakdown(ir),
        total_count=capped_total,
        shown_count=len(rows),
        has_more=has_more,
    )


def _dispatch_multi(db: Session, resolution: Resolution, session_id: str | None = None) -> dict:
    """Part 8, light multi-intent: each section of a compound utterance
    was already resolved independently by nlu_pipeline.split_subqueries()
    — dispatch each through the normal single-query path and stitch the
    replies into labeled sections. No new response type per section;
    reuses whatever _dispatch() already does for that section's kind.
    Pagination note: each section still runs through _dispatch_ir's own
    15-per-page cap, but since set_pagination() is called once per
    section, only the LAST section's "Show More" cursor survives — the
    same documented limitation multi-intent already has for
    conversation_memory in general."""
    sections = resolution.sections or []
    parts = []
    all_data = []
    for i, (_text, sub_resolution) in enumerate(sections, start=1):
        sub_response = _dispatch(db, sub_resolution, session_id)
        parts.append(f"{i}. {sub_response['reply']}")
        all_data.append(sub_response.get("data"))
    return respond("multi", "\n\n".join(parts), all_data)


def _dispatch_shortcut(db: Session, intent: str, entities: dict) -> dict:
    if intent == "greeting":
        return respond("text", "Hey! Happy to help — I can look up an advisor, a team, or a company, pull up a leaderboard, or check attendance. What would you like to know?", None)

    if intent == "thanks":
        return respond("text", "Anytime! Let me know if there's anything else you'd like to dig into.", None)

    if intent == "help":
        return respond("text", "Here's what I'm good at — try things like: 'tell me about <advisor>', 'how is <team> doing', 'top 5 by revenue', 'give me target achievement', 'who was late today', 'compare <team> with <team>', 'advisors from <company> who were late but still hit 80% of target'.", None)

    if intent == "attendance_check":
        rows = attendance_service.get_attendance_issues(db, entities.get("team"))
        audit.record_formatter("format_attendance_reply",
                               "shortcut intent 'attendance_check' — answered without the "
                               "planner, so no metric/entity beyond team was consulted")
        return respond("attendance", format_attendance_reply(rows), rows)

    audit.record_formatter("(none — canned text)",
                           f"shortcut intent {intent!r} matched no shortcut handler")
    return respond("no_data", "Hmm, I'm not quite sure how to answer that one yet — could you rephrase it?", None)


def _log_interaction(
    db: Session,
    session_id: str | None,
    message: str,
    resolution: Resolution,
    response: dict,
    trace=None,
):
    """Best-effort logging — a logging failure never breaks the chat response itself."""
    try:
        if resolution.kind == "shortcut":
            detected_intent = resolution.shortcut_intent
            confidence = 1.0
        elif resolution.kind == "plan":
            detected_intent = resolution.plan.action
            confidence = 0.6 if resolution.used_llm_fallback else 0.9
        elif resolution.kind == "ir":
            detected_intent = resolution.ir.intent
            confidence = resolution.ir.overall_confidence
        elif resolution.kind == "multi":
            detected_intent = "multi"
            sub_confidences = [
                r.ir.overall_confidence for _t, r in (resolution.sections or []) if r.ir is not None
            ]
            confidence = sum(sub_confidences) / len(sub_confidences) if sub_confidences else 0.5
        else:
            detected_intent = "clarify"
            confidence = 0.0

        # Part 6 / Phase 6 observability: persist the full QueryIR whenever
        # one was produced (kind == "ir", or kind == "clarify" that still
        # carries a partially-resolved IR from the validator) so a
        # production miss is debuggable from the filters/subjects/metric
        # the parser actually produced, not just the final label.
        resolved_ir_json = resolution.ir.model_dump_json() if resolution.ir is not None else None

        # Part 10: confidence metadata for that same IR, as its own column
        # (see ChatLog.confidence_metadata docstring) — ir.confidence_level
        # and ir.ambiguity_reasons are populated by ir_validator.validate_ir()
        # by the time any resolution.ir reaches here.
        confidence_metadata_json = None
        if resolution.ir is not None:
            confidence_metadata_json = json.dumps({
                **confidence_breakdown(resolution.ir),
                "level": resolution.ir.confidence_level,
                "ambiguity_reasons": resolution.ir.ambiguity_reasons,
            })

        # Phase 7: the full decision chain, stored as JSON alongside the
        # scalars worth querying directly. `trace_json` is what makes a
        # production failure reproducible — entities, the identity
        # candidates that were considered, the planner's decision, and
        # the literal SQL that ran.
        trace_json = json.dumps(trace.to_dict(), default=str) if trace is not None else None

        db.add(
            ChatLog(
                session_id=session_id,
                user_message=message,
                detected_intent=detected_intent,
                confidence=confidence,
                used_llm_fallback=resolution.used_llm_fallback,
                response_type=response["type"],
                resolved_ir=resolved_ir_json,
                confidence_metadata=confidence_metadata_json,
                trace=trace_json,
                trace_id=trace.trace_id if trace is not None else None,
                resolved_wid=trace.resolved_wid if trace is not None else None,
                row_count=trace.row_count if trace is not None else None,
                duration_ms=trace.duration_ms if trace is not None else None,
            )
        )
        db.commit()

    except Exception:
        db.rollback()
