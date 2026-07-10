# from app.database.session import SessionLocal
# from app.services.chat_service import handle_chat_message


# def main():

#     db = SessionLocal()

#     print("Sales Chatbot CLI")
#     print("Type 'exit' to quit\n")

#     session_id = "terminal-session"


#     try:
#         while True:

#             message = input("You: ")

#             if message.lower() in ["exit", "quit"]:
#                 break


#             result = handle_chat_message(
#                 db=db,
#                 message=message,
#                 session_id=session_id
#             )


#             print("\nBot:")
#             print(result.get("reply"))
#             print()


#     finally:
#         db.close()


# if __name__ == "__main__":
#     main()
    


#!/usr/bin/env python
"""
Local REPL for testing the chat pipeline end-to-end without HTTP, auth, or
uvicorn in the loop. Talks directly to chat_service against a real DB
session — the exact same code path production runs, minus the API wrapper.

Usage:
    python chat_cli.py
"""

from app.database.session import SessionLocal
from app.services.chat_service import handle_chat_message


def main():
    print("CCMC Sales Chatbot — local test REPL. Type 'exit' or Ctrl+C to quit.\n")
    db = SessionLocal()
    try:
        while True:
            try:
                message = input("You: ").strip()
            except EOFError:
                break
            if not message or message.lower() in ("exit", "quit"):
                break

            response = handle_chat_message(db, message, session_id="cli-test")
            print(f"Bot [{response['type']}]: {response['reply']}")
            if response.get("data") is not None:
                print(f"  data: {response['data']}")
            print()
    except KeyboardInterrupt:
        print()
    finally:
        db.close()


if __name__ == "__main__":
    main()