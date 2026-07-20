"""Read-only export of real user queries from ChatLog to JSONL, for
labeling into benchmark cases (grows the synthetic set toward the
500-1000 real-query target).

Usage:  python -m scripts.export_chatlog_queries [output.jsonl]
"""

import json
import sys

from app.database.models import ChatLog
from app.database.session import SessionLocal


def export(path: str = "chatlog_queries.jsonl") -> int:
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatLog)
            .order_by(ChatLog.created_at.desc())
            .all()
        )
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps({
                    "message": row.user_message,
                    "session_id": row.session_id,
                    "detected_intent": row.detected_intent,
                    "confidence": row.confidence,
                    "used_llm_fallback": row.used_llm_fallback,
                    "response_type": row.response_type,
                    "resolved_ir": json.loads(row.resolved_ir) if row.resolved_ir else None,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }) + "\n")
        return len(rows)
    finally:
        db.close()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "chatlog_queries.jsonl"
    count = export(out)
    print(f"exported {count} chat log rows to {out}")
