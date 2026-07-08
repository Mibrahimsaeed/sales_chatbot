from sqlalchemy.orm import Session
from app.llm.intent_detector import classify_intent
from app.services import advisor_service, team_service, leaderboard_service
from app.core.exception import NotFoundError, UnsupportedMetricError


def handle_chat_message(db: Session, message: str) -> dict:
    intent = classify_intent(message)

    if intent["type"] == "advisor_lookup":
        advisor = advisor_service.find_advisor_by_name(db, intent["name"])
        if not advisor:
            return {"type": "not_found", "data": None}
        return {"type": "advisor", "data": advisor}

    if intent["type"] == "team_summary":
        try:
            summary = team_service.get_team_summary(db, intent["team"])
        except NotFoundError:
            return {"type": "not_found", "data": None}
        return {"type": "team", "data": summary}

    if intent["type"] == "leaderboard":
        try:
            rows = leaderboard_service.get_leaderboard(db, intent["metric"])
        except UnsupportedMetricError:
            return {"type": "unknown", "data": None}
        return {"type": "leaderboard", "data": rows}

    return {"type": "unknown", "data": None}