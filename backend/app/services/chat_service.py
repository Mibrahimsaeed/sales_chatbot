import json

from sqlalchemy.orm import Session

from app.llm import conversation_memory, narrative
from app.llm.ir_validator import confidence_breakdown
from app.llm.nlu_pipeline import resolve, Resolution
from app.llm.query_compiler import compile_and_run, count_ir
from app.llm.query_ir import QueryIR
from app.llm.response_planner import plan_response
from app.llm.response_formatter import (
    format_advisor_reply,
    format_team_reply,
    format_company_reply,
    format_ir_reply,
    format_attendance_reply,
)
from app.services import (
    advisor_service,
    team_service,
    company_service,
    attendance_service,
)
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
    resolution = resolve(message, db, session_id=session_id)
    log.debug(f"resolved '{message}' -> kind={resolution.kind}")

    response = _dispatch(db, resolution, session_id)
    _log_interaction(db, session_id, message, resolution, response)
    return response


def handle_show_more(db: Session, session_id: str | None) -> dict:
    """Part 8: the single implementation behind both "Show More" triggers
    — the dedicated POST /chat/more endpoint (button click, no re-parse)
    and nlu_pipeline recognizing typed "show more" (kind="paginate").
    Fetches the NEXT page only; the caller (frontend) appends it to what
    it already has, per the pagination cursor in conversation_memory."""
    state = conversation_memory.get_pagination(session_id)
    if state is None:
        return {
            "type": "text",
            "reply": "There's nothing more to show right now — try asking a new question.",
            "data": None,
        }

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

    reply = format_ir_reply(
        ir, rows, total_count=capped_total, start_index=page_offset + 1, paginated=True
    )
    return {
        "type": ir.intent,
        "reply": reply,
        "data": rows,
        "total_count": capped_total,
        "shown_count": new_shown,
        "has_more": has_more,
    }


def _dispatch(db: Session, resolution: Resolution, session_id: str | None = None) -> dict:
    if resolution.kind == "shortcut":
        return _dispatch_shortcut(db, resolution.shortcut_intent, resolution.entities)

    if resolution.kind == "multi":
        return _dispatch_multi(db, resolution, session_id)

    if resolution.kind == "paginate":
        return handle_show_more(db, session_id)

    if resolution.kind == "clarify":
        return {
            "type": "clarification",
            "reply": resolution.clarify_message,
            "data": None,
            "options": resolution.clarify_options or [],
        }

    if resolution.kind == "ir":
        return _dispatch_ir(db, resolution, session_id)

    # kind == "plan" — lookup / summary / attendance_filter, unchanged from
    # the previous design (see nlu_pipeline.py's docstring for why these
    # stayed on the simpler rule-based path).
    plan = resolution.plan

    if plan.action == "lookup":
        advisor = advisor_service.find_advisor_by_name(db, plan.entity_value)
        if not advisor:
            return {
                "type": "not_found",
                "reply": f"I couldn't find an advisor matching '{plan.entity_value}'.",
                "data": None,
            }
        return {"type": "advisor", "reply": format_advisor_reply(advisor), "data": advisor}

    if plan.action == "summary" and plan.level == "team":
        try:
            summary = team_service.get_team_summary(db, plan.entity_value)
        except NotFoundError:
            return {"type": "not_found", "reply": f"No team matching '{plan.entity_value}'.", "data": None}
        return {"type": "team", "reply": format_team_reply(summary), "data": summary}

    if plan.action == "summary" and plan.level == "company":
        try:
            summary = company_service.get_company_summary(db, plan.entity_value)
        except NotFoundError:
            return {"type": "not_found", "reply": f"No company matching '{plan.entity_value}'.", "data": None}
        return {"type": "company", "reply": format_company_reply(summary), "data": summary}

    if plan.action == "attendance_filter":
        rows = attendance_service.get_attendance_by_status(
            db=db,
            team=plan.entity_value,
            status=plan.reason,
        )
        return {"type": "attendance", "reply": format_attendance_reply(rows), "data": rows}

    return {"type": "unknown", "reply": "I'm not sure how to answer that yet.", "data": None}


def _dispatch_ir(db: Session, resolution: Resolution, session_id: str | None = None) -> dict:
    """New path (Part 4/5.5): any query the generic compiler can answer —
    leaderboards, comparisons, and filtered/thresholded/boolean-combined
    queries — regardless of whether the QueryIR came from the rule-based
    fast path or the LLM Semantic Parser."""
    ir = resolution.ir
    true_total = count_ir(db, ir)

    if true_total is None:
        metric_label = ir.sort.metric or (ir.metric.key if ir.metric else "that metric")
        return {
            "type": "unknown",
            "reply": f"I don't have a way to answer that for {metric_label} at the {ir.subject_level} level yet.",
            "data": None,
        }

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

    reply = format_ir_reply(ir, rows, total_count=capped_total, paginated=has_more)
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
        if explanation:
            reply = f"{explanation}\n\n{reply}"
        # Part 8/11: only attach insights when the response planner judges
        # the result set large enough for "outlier"/"trend" to mean
        # anything (2-row comparisons don't have a meaningful peer group).
        # Anomaly detection and trend deltas are independent evidence
        # sources — both capped, combined cap keeps the reply concise.
        if plan_response(ir, rows).show_insights:
            insights = (narrative.compute_insights(ir, rows) + narrative.compute_trends(ir, rows, db))[:3]
    return {
        "type": ir.intent,
        "reply": reply,
        "data": rows,
        "insights": insights,
        "confidence": confidence_breakdown(ir),
        "total_count": capped_total,
        "shown_count": len(rows),
        "has_more": has_more,
    }


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
    return {
        "type": "multi",
        "reply": "\n\n".join(parts),
        "data": all_data,
    }


def _dispatch_shortcut(db: Session, intent: str, entities: dict) -> dict:
    if intent == "greeting":
        return {
            "type": "text",
            "reply": "Hi — I can look up an advisor, a team, a company, or answer leaderboard and attendance questions. What would you like to know?",
            "data": None,
        }

    if intent == "thanks":
        return {
            "type": "text",
            "reply": "You're welcome! Anything else you'd like to look up?",
            "data": None,
        }

    if intent == "help":
        return {
            "type": "text",
            "reply": "Try: 'tell me about <advisor>', 'how is <team> doing', 'top 5 by revenue', 'give me target achievement', 'who was late today', 'compare <team> with <team>', 'advisors from <company> who were late but still hit 80% of target'.",
            "data": None,
        }

    if intent == "attendance_check":
        rows = attendance_service.get_attendance_issues(db, entities.get("team"))
        return {"type": "attendance", "reply": format_attendance_reply(rows), "data": rows}

    return {"type": "unknown", "reply": "I'm not sure how to answer that yet.", "data": None}


def _log_interaction(
    db: Session,
    session_id: str | None,
    message: str,
    resolution: Resolution,
    response: dict,
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
            )
        )
        db.commit()

    except Exception:
        db.rollback()
