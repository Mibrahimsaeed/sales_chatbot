import re
import time
from difflib import SequenceMatcher
from sqlalchemy.orm import Session
from sqlalchemy import distinct
from app.database.models import Advisor

_CACHE_TTL_SECONDS = 300
_cache = {"teams": [], "companies": [], "advisor_names": [], "loaded_at": 0}

METRIC_KEYWORDS = {
    "revenue": "mtd_cleared", "cleared": "mtd_cleared", "sales": "mtd_cleared", "closed": "mtd_cleared",
    "connect": "mtd_new_connect", "connects": "mtd_new_connect",
    "overdue": "overdue", "pipeline": "pipeline",
}
PERIOD_KEYWORDS = {
    "mtd": "MTD", "this month": "MTD", "month": "MTD",
    "ytd": "YTD", "this year": "YTD", "year": "YTD",
    "3m": "3M", "quarter": "3M", "three month": "3M",
}


def _refresh_cache(db: Session):
    now = time.time()
    if _cache["loaded_at"] and now - _cache["loaded_at"] < _CACHE_TTL_SECONDS:
        return
    _cache["teams"] = [t for (t,) in db.query(distinct(Advisor.team)).filter(Advisor.team.isnot(None)).all()]
    _cache["companies"] = [c for (c,) in db.query(distinct(Advisor.company)).filter(Advisor.company.isnot(None)).all()]
    _cache["advisor_names"] = [n for (n,) in db.query(Advisor.name).all()]
    _cache["loaded_at"] = now


def get_known_teams(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["teams"]


def get_known_companies(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["companies"]


def _best_name_match(text: str, candidates: list[str], cutoff: float = 0.55) -> tuple[str, float] | None:
    best_name, best_score = None, 0.0
    for name in candidates:
        score = SequenceMatcher(None, text.lower(), name.lower()).ratio()
        if score > best_score:
            best_name, best_score = name, score
    if best_name and best_score >= cutoff:
        return best_name, best_score
    return None


def extract_entities(text: str, db: Session) -> dict:
    _refresh_cache(db)
    q = text.lower()
    entities: dict = {}

    for kw, metric in METRIC_KEYWORDS.items():
        if kw in q:
            entities["metric"] = metric
            break

    for kw, period in PERIOD_KEYWORDS.items():
        if kw in q:
            entities["period"] = period
            break

    limit_match = re.search(r"top\s+(\d+)", q)
    if limit_match:
        entities["limit"] = int(limit_match.group(1))

    for c in _cache["companies"]:
        if c and c.lower() in q:
            entities["company"] = c
            break

    for t in sorted((t for t in _cache["teams"] if t), key=len, reverse=True):
        if t.lower() in q:
            entities["team"] = t
            break

    matched_name = None
    for n in _cache["advisor_names"]:
        if n and n.lower() in q:
            matched_name = n
            break

    if not matched_name:
        cleaned = re.sub(r"tell me about|how is|show me|what about|performance of", "", text, flags=re.I).strip()
        if len(cleaned) > 2:
            result = _best_name_match(cleaned, _cache["advisor_names"])
            if result:
                matched_name, score = result
                entities["advisor_match_score"] = round(score, 2)

    if matched_name:
        entities["advisor_name"] = matched_name

    return entities