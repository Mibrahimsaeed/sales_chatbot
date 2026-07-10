from sqlalchemy.orm import Session

from app.llm.nlu_pipeline import resolve_intent
from app.llm.intent_detector import find_missing_slots
from app.llm.response_formatter import (
    format_response,
    format_clarification_reply
)

from app.services.query_planner import build_query_plan
from app.services.query_executor import execute_query

from app.database.models import ChatLog


def handle_chat_message(
    db: Session,
    message: str,
    session_id: str | None = None
):

    # ----------------------------
    # 1. NLU Processing
    # ----------------------------
    intent_result = resolve_intent(
        text=message,
        db=db
    )


    # ----------------------------
    # 2. Check Missing Information
    # ----------------------------
    missing_slots = find_missing_slots(
        intent_result
    )


    if missing_slots:

        response = format_clarification_reply(
            intent=intent_result.intent,
            missing_slots=missing_slots
        )


    else:

        # ----------------------------
        # 3. Build Query Plan
        # ----------------------------
        plan = build_query_plan(
            intent_result
        )


        # ----------------------------
        # 4. Execute Query
        # ----------------------------
        result = execute_query(
            db=db,
            plan=plan
        )


        # Add metadata needed by formatter
        result["intent"] = intent_result.intent
        result["metric"] = (
            intent_result.entities.get("metric")
            if intent_result.entities
            else None
        )


        # ----------------------------
        # 5. Format Final Response
        # ----------------------------
        response = format_response(
            result
        )


    # ----------------------------
    # 6. Save Conversation Log
    # ----------------------------
    chat_log = ChatLog(
        session_id=session_id,
        user_message=message,
        detected_intent=intent_result.intent,
        confidence=intent_result.confidence,
        used_llm_fallback=getattr(
            intent_result,
            "used_llm_fallback",
            False
        ),
        response_type="text"
    )

    db.add(chat_log)
    db.commit()


    # ----------------------------
    # 7. Return API Response
    # ----------------------------
    return {
        "session_id": session_id,
        "message": response,
        "intent": intent_result.intent,
        "confidence": intent_result.confidence
    }