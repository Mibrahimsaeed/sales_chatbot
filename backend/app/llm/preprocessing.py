import re


def normalize(text: str) -> str:
    """Lowercase-safe cleanup that preserves original casing for entity
    matching later (advisor names are case-sensitive in the DB) — trims
    whitespace and collapses repeated spaces only."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text