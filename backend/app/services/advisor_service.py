from sqlalchemy import text
from sqlalchemy.orm import Session


def find_advisor_by_name(db: Session, query: str) -> dict | None:
    """Reads from the advisor_profile VIEW — one query across all star-schema
    tables instead of six separate joins, since this is the chatbot's most
    common lookup pattern."""
    row = db.execute(
        text("SELECT * FROM advisor_profile WHERE name ILIKE :q ORDER BY wid LIMIT 1"),
        {"q": f"%{query}%"},
    ).mappings().first()
    return dict(row) if row else None