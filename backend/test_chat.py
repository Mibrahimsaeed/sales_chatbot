from app.database.session import SessionLocal
from app.services.chat_service import handle_chat_message


db = SessionLocal()


try:
    result = handle_chat_message(
        db=db,
        message="who has highest sales",
        session_id="test-session-1"
    )

    print(result)

finally:
    db.close()
    