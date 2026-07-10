# from sqlalchemy.orm import Session

# from app.llm.nlu_pipeline import resolve_intent
# from app.llm.intent_detector import find_missing_slots
# from app.llm.response_formatter import (
#     format_response,
#     format_clarification_reply
# )

# from app.services.query_planner import build_query_plan
# from app.services.query_executor import execute_query

# from app.database.models import ChatLog


# def handle_chat_message(
#     db: Session,
#     message: str,
#     session_id: str | None = None
# ):

#     # ----------------------------
#     # 1. NLU Processing
#     # ----------------------------
#     intent_result = resolve_intent(
#         text=message,
#         db=db
#     )


#     # ----------------------------
#     # 2. Check Missing Information
#     # ----------------------------
#     missing_slots = find_missing_slots(
#         intent_result
#     )


#     if missing_slots:

#         response = format_clarification_reply(
#             intent=intent_result.intent,
#             missing_slots=missing_slots
#         )


#     else:

#         # ----------------------------
#         # 3. Build Query Plan
#         # ----------------------------
#         plan = build_query_plan(
#             intent_result
#         )


#         # ----------------------------
#         # 4. Execute Query
#         # ----------------------------
#         result = execute_query(
#             db=db,
#             plan=plan
#         )


#         # Add metadata needed by formatter
#         result["intent"] = intent_result.intent
#         result["metric"] = (
#             intent_result.entities.get("metric")
#             if intent_result.entities
#             else None
#         )


#         # ----------------------------
#         # 5. Format Final Response
#         # ----------------------------
#         response = format_response(
#             result
#         )


#     # ----------------------------
#     # 6. Save Conversation Log
#     # ----------------------------
#     chat_log = ChatLog(
#         session_id=session_id,
#         user_message=message,
#         detected_intent=intent_result.intent,
#         confidence=intent_result.confidence,
#         used_llm_fallback=getattr(
#             intent_result,
#             "used_llm_fallback",
#             False
#         ),
#         response_type="text"
#     )

#     db.add(chat_log)
#     db.commit()


#     # ----------------------------
#     # 7. Return API Response
#     # ----------------------------
#     return {
#         "session_id": session_id,
#         "message": response,
#         "intent": intent_result.intent,
#         "confidence": intent_result.confidence
#     }



from sqlalchemy.orm import Session
from app.llm.nlu_pipeline import resolve, Resolution
from app.llm.sql_generator import run_leaderboard
from app.llm.response_formatter import (
    format_advisor_reply, format_team_reply, format_company_reply,
    format_leaderboard_reply, format_attendance_reply,
)
from app.services import advisor_service, team_service, company_service, attendance_service
from app.core.exception import NotFoundError
from app.database.models import ChatLog


# def handle_chat_message(db: Session, message: str, session_id: str | None = None) -> dict:
#     resolution = resolve(message, db)
    
#     response = _dispatch(db, resolution)
#     _log_interaction(db, session_id, message, resolution, response)
#     return response
def handle_chat_message(db: Session, message: str, session_id: str | None = None) -> dict:
    resolution = resolve(message, db)

    print("\n===== DEBUG RESOLUTION =====")
    print(resolution)
    print("KIND:", resolution.kind)

    if resolution.plan:
        print("PLAN:", resolution.plan)

    print("============================\n")

    response = _dispatch(db, resolution)
    _log_interaction(db, session_id, message, resolution, response)
    return response


def _dispatch(db: Session, resolution: Resolution) -> dict:
    if resolution.kind == "shortcut":
        return _dispatch_shortcut(db, resolution.shortcut_intent, resolution.entities)

    if resolution.kind == "clarify":
        return {"type": "clarification", "reply": resolution.clarify_message, "data": None}

    plan = resolution.plan

    if plan.action == "lookup":
        advisor = advisor_service.find_advisor_by_name(db, plan.entity_value)
        if not advisor:
            return {"type": "not_found", "reply": f"I couldn't find an advisor matching '{plan.entity_value}'.", "data": None}
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

    if plan.action == "leaderboard":
        rows = run_leaderboard(db, plan)
        if rows is None:
            return {
                "type": "unknown",
                "reply": f"I don't have a way to rank by that metric at the {plan.level} level yet.",
                "data": None,
            }
        return {"type": "leaderboard", "reply": format_leaderboard_reply(plan.metric, rows), "data": rows}

    return {"type": "unknown", "reply": "I'm not sure how to answer that yet.", "data": None}


def _dispatch_shortcut(db: Session, intent: str, entities: dict) -> dict:
    if intent == "greeting":
        return {
            "type": "text",
            "reply": "Hi \u2014 I can look up an advisor, a team, a company, or answer leaderboard and attendance questions. What would you like to know?",
            "data": None,
        }
    if intent == "help":
        return {
            "type": "text",
            "reply": "Try: 'tell me about <advisor>', 'how is <team> doing', 'top 5 by revenue', 'give me target achievement', 'who was late today'.",
            "data": None,
        }
    if intent == "attendance_check":
        rows = attendance_service.get_attendance_issues(db, entities.get("team"))
        return {"type": "attendance", "reply": format_attendance_reply(rows), "data": rows}
    return {"type": "unknown", "reply": "I'm not sure how to answer that yet.", "data": None}


def _log_interaction(db: Session, session_id: str | None, message: str, resolution: Resolution, response: dict):
    """Best-effort logging — a logging failure never breaks the chat response itself."""
    try:
        plan = resolution.plan
        if resolution.kind == "shortcut":
            detected_intent = resolution.shortcut_intent
            confidence = 1.0
        elif resolution.kind == "plan":
            detected_intent = plan.action
            confidence = 0.6 if resolution.used_llm_fallback else 0.9
        else:
            detected_intent = "clarify"
            confidence = 0.0

        db.add(ChatLog(
            session_id=session_id,
            user_message=message,
            detected_intent=detected_intent,
            confidence=confidence,
            used_llm_fallback=resolution.used_llm_fallback,
            response_type=response["type"],
        ))
        db.commit()
    except Exception:
        db.rollback()