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

Part 12 (semantic retrieval expansion): both the comparator vocabulary and
the attendance-status keyword map now have a semantic widening step for
phrasing the closed lists above don't cover ("north of 80", "showed up
late" without the bare word "late"). Both only run after their
deterministic list finds nothing — see _extract_thresholds() and the
attendance block in extract_entities() — so neither can override an
already-matched deterministic result. The comparator floor is
deliberately high: a wrong operator silently flips a filter's direction,
which is a wrong-answer bug, not a missing-answer one.
"""

import re
import time
from sqlalchemy.orm import Session
from sqlalchemy import distinct
from app.database.models import Advisor
from app.llm import entity_linker
from app.llm.fuzzy_match import best_match, find_in_text, STRONG_FLOOR
from app.llm.temporal_parser import parse_period

_CACHE_TTL_SECONDS = 300
_cache = {
    "teams": [], "companies": [], "advisor_names": [],
    "offices": [], "portfolio_leads": [], "management_leads": [],
    "loaded_at": 0,
}
ATTENDANCE_STATUS_KEYWORDS = {
    "not marked": "Not Marked",
    "late": "Late",
    "present": "Present",
    "absent": "Absent",
}

# Part 12: paraphrases the bare keywords above don't catch. Merged with
# them as a semantic fallback, never a replacement — see extract_entities().
#  None of these literally contain one of ATTENDANCE_STATUS_KEYWORDS'
#  words ("not marked"/"late"/"present"/"absent") — an exemplar that did
#  would never actually exercise this fallback, since the keyword check
#  above would already have matched it first. Every exemplar also contains
#  at least one word _ATTENDANCE_HINT_RE below recognizes, so the cheap
#  pre-filter and the corpus stay in sync.
_ATTENDANCE_STATUS_EXEMPLARS: list[tuple[str, str]] = [
    ("didn't show up", "Absent"), ("no show", "Absent"), ("missed today", "Absent"),
    ("was a no-show", "Absent"),
    ("walked in behind schedule", "Late"), ("clocked in behind schedule", "Late"),
    ("showed up past the scheduled start time", "Late"), ("came in past the scheduled time", "Late"),
    ("showed up on schedule", "Present"), ("showed up today", "Present"), ("clocked in on time", "Present"),
    ("hasn't punched in", "Not Marked"), ("no attendance record", "Not Marked"),
    ("didn't log attendance", "Not Marked"), ("no biometric record", "Not Marked"),
]

entity_linker.register_exemplar_type("attendance_status", lambda: _ATTENDANCE_STATUS_EXEMPLARS)

# Cheap pre-filter for the semantic attendance fallback below — most
# messages have nothing to do with attendance at all and must never pay
# for an embedding call just to find that out.
_ATTENDANCE_HINT_RE = re.compile(
    r"\b(show\w*|attend\w*|present|absent|late|punch\w*|mark\w*|biometric|login|"
    r"walk\w*|clock\w*|schedule\w*|miss\w*)\b", re.I
)

# Closed, small comparator vocabulary — kept rule-based per Part 7 of the
# redesign brief ("this vocabulary genuinely is small and closed, unlike
# metric/intent vocabulary which is open-ended"). Part 12 adds a semantic
# fallback (see _extract_thresholds) for phrasing outside this list —
# still closed in SHAPE (there are only 4 possible operators), just no
# longer closed in PHRASING.
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

# Deliberately narrow, unambiguous phrasings only — see the module
# docstring on why the comparator floor is set higher than entity linking.
_COMPARATOR_EXEMPLARS: list[tuple[str, str]] = [
    ("north of", ">"), ("above", ">"), ("in excess of", ">"), ("more than", ">"),
    ("upwards of", ">="), ("no less than", ">="), ("at minimum", ">="),
    ("south of", "<"), ("shy of", "<"), ("a bit under", "<"), ("just below", "<"),
    ("no more than", "<="), ("at maximum", "<="), ("capped at", "<="),
]

entity_linker.register_exemplar_type("comparator", lambda: _COMPARATOR_EXEMPLARS)

_SEMANTIC_COMPARATOR_FLOOR = 0.85
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?")


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
    _cache["offices"] = [o for (o,) in db.query(distinct(Advisor.office)).filter(Advisor.office.isnot(None), master_only).all()]
    _cache["portfolio_leads"] = [p for (p,) in db.query(distinct(Advisor.portfolio_lead)).filter(Advisor.portfolio_lead.isnot(None), master_only).all()]
    _cache["management_leads"] = [m for (m,) in db.query(distinct(Advisor.management_lead)).filter(Advisor.management_lead.isnot(None), master_only).all()]
    _cache["loaded_at"] = now


def get_known_teams(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["teams"]


def get_known_companies(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["companies"]


def get_known_advisor_names(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["advisor_names"]


def get_known_offices(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["offices"]


def get_known_portfolio_leads(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["portfolio_leads"]


def get_known_management_leads(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["management_leads"]


# Registered once at import time — the only step required to make a new
# entity type semantically linkable is one more call like these (Part 9).
entity_linker.register_entity_type("advisor", get_known_advisor_names, kind="advisor")
entity_linker.register_entity_type("team", get_known_teams, kind="team")
entity_linker.register_entity_type("company", get_known_companies, kind="company")
entity_linker.register_entity_type("office", get_known_offices, kind="team")
entity_linker.register_entity_type("portfolio_lead", get_known_portfolio_leads, kind="advisor")
entity_linker.register_entity_type("management_lead", get_known_management_leads, kind="advisor")


def _extract_thresholds(q: str) -> list[dict]:
    thresholds = []
    consumed_spans: list[tuple[int, int]] = []
    for pattern, operator in _THRESHOLD_PATTERNS:
        for match in re.finditer(pattern, q):
            thresholds.append({"operator": operator, "value": float(match.group(1))})
            consumed_spans.append(match.span())

    # Part 12: a number the closed vocabulary above didn't already pair
    # with a comparator might still be one, phrased unusually ("north of
    # 80", "a bit under 30") — classify the words right before it against
    # the comparator exemplars at a HIGH floor (see module docstring on
    # why) before leaving it as just a bare number with no comparator.
    for match in _NUMBER_RE.finditer(q):
        if any(start <= match.start() < end for start, end in consumed_spans):
            continue
        window = q[max(0, match.start() - 25):match.start()].strip()
        if not window:
            continue
        semantic = entity_linker.semantic_classify(window, "comparator", top_k=1, floor=_SEMANTIC_COMPARATOR_FLOOR)
        if semantic:
            thresholds.append({"operator": semantic[0]["value"], "value": float(match.group(1))})

    return thresholds


def _resolve_gazetteer_field(
    text: str, q: str, values: list[str], kind: str, entity_type: str, db: Session
) -> list[dict]:
    """Exact substring -> RapidFuzz fuzzy -> embedding semantic search
    (Part 9), in that order — each tier only runs if the previous one
    found nothing. Longest-value-first on the substring pass so a full
    name isn't shadowed by a shorter partial hit contained within it
    (Root Cause #2). Returns [] rather than guessing once every tier has
    been tried and none cleared its floor — the caller's job is then to
    ask for clarification, not to assume no match exists."""
    ordered = sorted((v for v in values if v), key=len, reverse=True)
    matches = [{"value": v, "score": 1.0} for v in ordered if v.lower() in q]
    if matches:
        return matches

    matches = [
        {"value": v, "score": s}
        for v, s in find_in_text(q, values, kind=kind, floor=STRONG_FLOOR)
    ]
    if matches:
        return matches

    return entity_linker.semantic_candidates(text, entity_type, db)


def extract_entities(text: str, db: Session) -> dict:
    _refresh_cache(db)
    q = text.lower()
    entities: dict = {}

    for keyword, status in ATTENDANCE_STATUS_KEYWORDS.items():
        if keyword in q:
            entities["attendance_status"] = status
            break
    else:
        # Part 12: no exact keyword — try semantic retrieval against
        # paraphrases ("didn't show up", "clocked in late") before
        # concluding no status was mentioned. Gated on a cheap keyword
        # hint so the common case (no attendance language at all) never
        # pays for an embedding call.
        if _ATTENDANCE_HINT_RE.search(q):
            semantic = entity_linker.semantic_classify(q, "attendance_status", top_k=1)
            if semantic:
                entities["attendance_status"] = semantic[0]["value"]

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

    # companies/teams/offices/portfolio & management leads — exact substring,
    # then fuzzy, then embedding semantic search (Part 9); see
    # _resolve_gazetteer_field. ALL matches collected, not just the first
    # (Root Cause #2). New pluggable entity types (office, portfolio_lead,
    # management_lead) follow the exact same {plural}_matches/{plural}/
    # {singular} shape as the original team/company fields, so nothing
    # downstream that only knows about team/company needs to change.
    for plural, kind, entity_type in (
        ("companies", "company", "company"),
        ("teams", "team", "team"),
        ("offices", "team", "office"),
        ("portfolio_leads", "advisor", "portfolio_lead"),
        ("management_leads", "advisor", "management_lead"),
    ):
        matches = _resolve_gazetteer_field(text, q, _cache[plural], kind, entity_type, db)
        if matches:
            entities[f"{entity_type}_matches"] = matches
            entities[plural] = [m["value"] for m in matches]
            entities[entity_type] = matches[0]["value"]   # backward-compat singular

    # advisor name(s) — substring first (cheap, exact), fuzzy fallback,
    # then embedding semantic search as a last resort (Part 9) — catches
    # things like initials or a badly misspelled name RapidFuzz rejects.
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
            else:
                semantic_matches = entity_linker.semantic_candidates(cleaned, "advisor", db)
                if semantic_matches:
                    matched_name = semantic_matches[0]["value"]
                    entities["advisor_matches"] = semantic_matches
                    entities["advisor_match_score"] = semantic_matches[0]["score"]
    if matched_name:
        entities["advisor_name"] = matched_name

    return entities
