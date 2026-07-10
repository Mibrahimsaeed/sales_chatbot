from app.database.session import SessionLocal
from app.services.chat_service import handle_chat_message


def main():

    db = SessionLocal()

    print("Sales Chatbot CLI")
    print("Type 'exit' to quit\n")

    session_id = "terminal-session"


    try:
        while True:

            message = input("You: ")

            if message.lower() in ["exit", "quit"]:
                break


            result = handle_chat_message(
                db=db,
                message=message,
                session_id=session_id
            )


            print("\nBot:")
            print(result["message"])
            print()


    finally:
        db.close()


if __name__ == "__main__":
    main()
    