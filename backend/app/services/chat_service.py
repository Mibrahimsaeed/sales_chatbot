from sqlalchemy.orm import Session

from app.llm import narrative
from app.llm.nlu_pipeline import resolve, Resolution
from app.llm.query_compiler import compile_and_run
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


def handle_chat_message(
    db: Session,
    message: str,
    session_id: str | None = None,
) -> dict:
    resolution = resolve(message, db, session_id=session_id)
    log.debug(f"resolved '{message}' -> kind={resolution.kind}")

    response = _dispatch(db, resolution)
    _log_interaction(db, session_id, message, resolution, response)
    return response


def _dispatch(db: Session, resolution: Resolution) -> dict:
    if resolution.kind == "shortcut":
        return _dispatch_shortcut(db, resolution.shortcut_intent, resolution.entities)

    if resolution.kind == "clarify":
        return {
            "type": "clarification",
            "reply": resolution.clarify_message,
            "data": None,
        }

    if resolution.kind == "ir":
        return _dispatch_ir(db, resolution)

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


def _dispatch_ir(db: Session, resolution: Resolution) -> dict:
    """New path (Part 4/5.5): any query the generic compiler can answer —
    leaderboards, comparisons, and filtered/thresholded/boolean-combined
    queries — regardless of whether the QueryIR came from the rule-based
    fast path or the LLM Semantic Parser."""
    ir = resolution.ir
    rows = compile_and_run(db, ir)

    if rows is None:
        metric_label = ir.sort.metric or (ir.metric.key if ir.metric else "that metric")
        return {
            "type": "unknown",
            "reply": f"I don't have a way to answer that for {metric_label} at the {ir.subject_level} level yet.",
            "data": None,
        }

    reply = format_ir_reply(ir, rows)
    if rows:
        # narrative polish (P7): deterministic facts + LLM phrasing only,
        # fail-soft back to the templated reply (see narrative.py)
        reply = narrative.polish_reply(narrative.compute_facts(ir, rows), reply)
    return {"type": ir.intent, "reply": reply, "data": rows}


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
        else:
            detected_intent = "clarify"
            confidence = 0.0

        # Part 6 / Phase 6 observability: persist the full QueryIR whenever
        # one was produced (kind == "ir", or kind == "clarify" that still
        # carries a partially-resolved IR from the validator) so a
        # production miss is debuggable from the filters/subjects/metric
        # the parser actually produced, not just the final label.
        resolved_ir_json = resolution.ir.model_dump_json() if resolution.ir is not None else None

        db.add(
            ChatLog(
                session_id=session_id,
                user_message=message,
                detected_intent=detected_intent,
                confidence=confidence,
                used_llm_fallback=resolution.used_llm_fallback,
                response_type=response["type"],
                resolved_ir=resolved_ir_json,
            )
        )
        db.commit()

    except Exception:
        db.rollback()
