from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.core.dependencies import get_db, get_current_user
from app.services.chat_service import handle_chat_message

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Defined here for now — move to database/schemas.py alongside
    ChatResponse once that file is rewritten for the star schema.

    `session_id` is required for Conversation Memory (query_ir follow-ups
    like "what about last month" / "same for Downtown") to work at all —
    without it, every request is treated as a fresh conversation. The
    frontend should generate and persist one client-side id per chat
    session and send it on every message."""
    message: str
    session_id: str | None = None


@router.post("")
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    return handle_chat_message(db, payload.message, session_id=payload.session_id)
