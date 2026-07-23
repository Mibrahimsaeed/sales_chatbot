"""
Extracts structured entities from free text: advisor names, team names,
company names, metrics, time periods, numeric limits ("top 5"), and now
— every gazetteer match per type instead of only the first one, plus
comparator/threshold tokens ("more than 80%", "at least 5").

Advisor/team/company matching is done against a real gazetteer pulled from
the DB and cached in memory — matching against actual data beats keyword
regex for a domain this specific. The cache is per-process; for horizontal
scaling with multiple workers, swap this for a shared cache (Redis) so all
workers see the same refresh. This stays 100% rule-based on purpose: an
LLM has no advantage over SequenceMatcher for "does this substring match a
real DB value", and would only add latency/cost/hallucination risk.

Root Cause #2 fix: `extract_entities()` used to `break` on the first team/
company hit, silently discarding a second one — "Compare Blue Area with
Downtown" only ever saw "Blue Area". `teams`/`companies` below are now
lists of every gazetteer match found, longest-match-first so a full team
name isn't shadowed by a shorter partial hit contained within it. `team`/
`company` (singular) are kept for backward compatibility with the existing
rule-based query_planner.py, and are simply the first list entry.

Root Cause #4 fix: `thresholds` extracts a small, closed comparator
vocabulary (>, >=, <, <=, "at least", "more than", "over", "under",
"below") paired with a number — fed to the LLM semantic parser as a hint,
and used directly by the rule-based fast path when unambiguous.
"""

import re
import time
from sqlalchemy.orm import Session
from sqlalchemy import distinct
from app.database.models import Advisor
from app.llm.fuzzy_match import best_match, find_in_text, STRONG_FLOOR
from app.llm.temporal_parser import parse_period

_CACHE_TTL_SECONDS = 300
_cache = {"teams": [], "companies": [], "advisor_names": [], "loaded_at": 0}
ATTENDANCE_STATUS_KEYWORDS = {
    "not marked": "Not Marked",
    "late": "Late",
    "present": "Present",
    "absent": "Absent",
}

# Closed, small comparator vocabulary — kept rule-based per Part 7 of the
# redesign brief ("this vocabulary genuinely is small and closed, unlike
# metric/intent vocabulary which is open-ended").
_THRESHOLD_PATTERNS: list[tuple[str, str]] = [
    (r"more than (\d+(?:\.\d+)?)\s*%?", ">"),
    (r"at least (\d+(?:\.\d+)?)\s*%?", ">="),
    (r"over (\d+(?:\.\d+)?)\s*%?", ">"),
    (r"greater than (\d+(?:\.\d+)?)\s*%?", ">"),
    (r"less than (\d+(?:\.\d+)?)\s*%?", "<"),
    (r"below (\d+(?:\.\d+)?)\s*%?", "<"),
    (r"under (\d+(?:\.\d+)?)\s*%?", "<"),
    (r"at most (\d+(?:\.\d+)?)\s*%?", "<="),
    (r">=\s*(\d+(?:\.\d+)?)", ">="),
    (r"<=\s*(\d+(?:\.\d+)?)", "<="),
    (r">\s*(\d+(?:\.\d+)?)", ">"),
    (r"<\s*(\d+(?:\.\d+)?)", "<"),
]


def _refresh_cache(db: Session):
    now = time.time()
    if _cache["loaded_at"] and now - _cache["loaded_at"] < _CACHE_TTL_SECONDS:
        return
    # in_master_sheet=True only — a team/company/advisor name that only
    # exists because of a raw-data-only WID shouldn't ground a query or
    # get fuzzy-matched against (see models.py's in_master_sheet docstring).
    master_only = Advisor.in_master_sheet.is_(True)
    _cache["teams"] = [t for (t,) in db.query(distinct(Advisor.team)).filter(Advisor.team.isnot(None), master_only).all()]
    _cache["companies"] = [c for (c,) in db.query(distinct(Advisor.company)).filter(Advisor.company.isnot(None), master_only).all()]
    _cache["advisor_names"] = [n for (n,) in db.query(Advisor.name).filter(master_only).all()]
    _cache["loaded_at"] = now


def get_known_teams(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["teams"]


def get_known_companies(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["companies"]


def _extract_thresholds(q: str) -> list[dict]:
    thresholds = []
    for pattern, operator in _THRESHOLD_PATTERNS:
        for match in re.finditer(pattern, q):
            thresholds.append({"operator": operator, "value": float(match.group(1))})
    return thresholds


def extract_entities(text: str, db: Session) -> dict:
    _refresh_cache(db)
    q = text.lower()
    entities: dict = {}

    for keyword, status in ATTENDANCE_STATUS_KEYWORDS.items():
        if keyword in q:
            entities["attendance_status"] = status
            break

    # Part 8: parse_period() replaces the old bare "month"/"year" keyword
    # map, which used to silently turn "last month" into MTD. Genuinely
    # unsupported windows (last month, yesterday, this week, past N days,
    # custom ranges) are surfaced as period_unsupported instead of a
    # silently wrong period — never set alongside "period".
    temporal = parse_period(q)
    if temporal is not None:
        if temporal.kind == "equivalent":
            entities["period"] = temporal.period
            entities["period_confidence"] = temporal.confidence
        else:
            entities["period_unsupported"] = temporal.reason

    limit_match = re.search(r"top\s+(\d+)", q)
    if limit_match:
        entities["limit"] = int(limit_match.group(1))

    entities["thresholds"] = _extract_thresholds(q)

    # companies — substring first (exact, confidence 1.0), then a fuzzy
    # scan for typos ("grana" -> Graana), which only auto-accepts at the
    # STRONG floor since a false-positive company filter silently narrows
    # results. ALL matches collected, not just the first (Root Cause #2).
    company_matches = [
        {"value": c, "score": 1.0} for c in _cache["companies"] if c and c.lower() in q
    ]
    if not company_matches:
        company_matches = [
            {"value": c, "score": s}
            for c, s in find_in_text(q, _cache["companies"], kind="company", floor=STRONG_FLOOR)
        ]
    if company_matches:
        entities["company_matches"] = company_matches
        entities["companies"] = [m["value"] for m in company_matches]
        entities["company"] = company_matches[0]["value"]     # backward-compat singular

    # teams — longest substring match first so "Blue Area" doesn't get
    # shadowed by a shorter partial hit, then the same fuzzy fallback
    team_matches = [
        {"value": t, "score": 1.0}
        for t in sorted((t for t in _cache["teams"] if t), key=len, reverse=True)
        if t.lower() in q
    ]
    if not team_matches:
        team_matches = [
            {"value": t, "score": s}
            for t, s in find_in_text(q, _cache["teams"], kind="team", floor=STRONG_FLOOR)
        ]
    if team_matches:
        entities["team_matches"] = team_matches
        entities["teams"] = [m["value"] for m in team_matches]
        entities["team"] = team_matches[0]["value"]           # backward-compat singular

    # advisor name(s) — substring first (cheap, exact), fuzzy fallback
    advisor_names_found = [n for n in _cache["advisor_names"] if n and n.lower() in q]
    matched_name = advisor_names_found[0] if advisor_names_found else None
    if advisor_names_found:
        entities["advisor_names"] = advisor_names_found
        entities["advisor_matches"] = [{"value": n, "score": 1.0} for n in advisor_names_found]
    if not matched_name:
        # strip common filler so fuzzy matching isn't thrown off by "tell me about"
        cleaned = re.sub(r"tell me about|how is|show me|what about|performance of", "", text, flags=re.I).strip()
        if len(cleaned) > 2:
            result = best_match(cleaned, _cache["advisor_names"], kind="advisor")
            if result:
                matched_name, score = result
                entities["advisor_match_score"] = round(score, 2)
                entities["advisor_matches"] = [{"value": matched_name, "score": round(score, 2)}]
    if matched_name:
        entities["advisor_name"] = matched_name

    return entities
