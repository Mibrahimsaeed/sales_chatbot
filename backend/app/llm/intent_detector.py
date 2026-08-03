"""
Fast, cheap, first-pass intent classification.

This layer:
- does NOT call any LLM
- detects common business intents using rules
- extracts/uses entities provided by entity_extractor
- returns confidence so nlu_pipeline decides whether LLM fallback is needed

Important:
- Generic attendance queries are handled here.
- Specific attendance filters (team + status) are passed to query_planner.
  Example:
      "show not marked people in Blue Area"
      "who was late in Downtown"

Part 8 (intent ranking): every rule below used to be a sequential
if/return chain — first match wins, so two candidate intents never get
compared. Rules are now independent scoring functions evaluated in full;
the highest-confidence candidate wins (earliest-defined rule breaks a tie,
which reproduces the old first-match behavior exactly for every existing
test case). This also means a rule's `entities.setdefault(...)` side
effect only applies when that rule actually wins, instead of leaking into
the entities dict just because it was checked first.

Caveat worth stating plainly: nlu_pipeline.resolve() calls
classify_intent(cleaned, {}) and only ever inspects `.intent` against
SHORTCUT_INTENTS = ("greeting", "thanks", "help", "attendance_check") —
the leaderboard/*_summary/advisor_lookup branches below are not consulted
for live routing (query_planner.py and semantic_parser.py own that).
Ranking them is still worth doing — this module is independently unit
tested, and removes a first-match footgun for whoever extends the
shortcut set later — but it does not change production routing today.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.llm import entity_linker, intent_catalog as cat, token_match
from app.llm.entity_extractor import ATTENDANCE_STATUS_KEYWORDS


@dataclass
class IntentResult:
    intent: str
    confidence: float
    entities: dict = field(default_factory=dict)
    missing_slots: list = field(default_factory=list)
    used_llm_fallback: bool = False


REQUIRED_SLOTS = {
    "advisor_lookup": ["advisor_name"],
    "team_summary": ["team"],
    "company_summary": ["company"],
    "leaderboard": ["metric"],
    "attendance_check": [],
    "greeting": [],
    "thanks": [],
    "help": [],
    "unknown": [],
}


@dataclass
class _Candidate:
    intent: str
    confidence: float
    entity_patch: dict = field(default_factory=dict)


def _rule_greeting(q: str, entities: dict) -> Optional[_Candidate]:
    if re.search(r"^(hi|hello|hey|salam|assalam)\b", q):
        return _Candidate("greeting", 1.0)
    return None


def _rule_thanks(q: str, entities: dict) -> Optional[_Candidate]:
    if re.search(r"^(thanks|thank you|thankyou|thx|shukriya)\b", q):
        return _Candidate("thanks", 1.0)
    return None


def _rule_help(q: str, entities: dict) -> Optional[_Candidate]:
    if re.search(r"\b(help|what can you do|commands)\b", q):
        return _Candidate("help", 1.0)
    return None


def _is_analytical(q: str) -> bool:
    """Does this ask for a RANKING or a FILTERED set rather than a sweep?

    Deliberately narrow. A resolved metric is NOT used as the signal: the
    bare word "attendance" is itself an attendance_rate synonym, so every
    generic "show attendance issues" would look analytical and the
    shortcut would never fire at all.

    A strong ranking word or an explicit threshold are unambiguous — this
    shortcut can express neither, so a question containing one belongs to
    the planner. Both are read from the TEXT, because this runs before
    entity extraction and receives an empty entities dict.
    """
    from app.llm.entity_extractor import _extract_thresholds

    if token_match.contains_any(q, cat.RANKING_STRONG):
        return True
    return bool(_extract_thresholds(q))


def _rule_attendance(q: str, entities: dict) -> Optional[_Candidate]:
    # IMPORTANT: do NOT capture specific filters here — those go through
    # query_planner ("show not marked people in Blue Area", "who was late
    # in AMD", and — just as importantly — "who was not marked today"
    # with no team at all: query_planner.build_query_plan() routes to
    # attendance_filter off attendance_status ALONE, team is optional).
    # Only a truly generic attendance question with no specific status
    # named ("show attendance issues", "any attendance problems today")
    # becomes this shortcut. A specific status word (late/not marked/
    # absent/present) must always fall through to the filtered path —
    # otherwise "who was not marked today" incorrectly returns everyone
    # with ANY issue (late arrivals included), since this shortcut's
    # get_attendance_issues() ignores which status was actually asked
    # about, it only excludes "On Time".
    # WORD-BOUNDED. This was an unanchored alternation, so "late" matched
    # inside escaLATE / reLATEd / calcuLATEd / transLATE and this shortcut
    # — which runs BEFORE entity extraction and BEFORE the planner, and
    # returns immediately — hijacked the entire query. "How is the
    # answered calls percentage calculated?" was answered with a list of
    # advisors who have attendance problems.
    #
    # This is finding F1 in a second location: Step 1 anchored the keyword
    # TABLES, and the flag below, but left this regex's own alternation
    # unanchored. It is the more damaging of the two, because nothing
    # downstream gets a chance to disagree with it.
    attendance_match = re.search(
        r"\b(late|not marked|absent|missing|missed|biometric|login|attendance)\b", q
    )
    # F1: token-aware, like the extractor's scan of the same table —
    # otherwise 'calculated'/'related' counted as naming a status here
    # too, and this flag decides whether the shortcut fires.
    has_specific_status = token_match.contains_any(q, ATTENDANCE_STATUS_KEYWORDS)
    # A comparison phrase is context too: "compare Blue Area and DHA
    # attendance" is a two-entity question, not a generic "who has
    # attendance problems" sweep. Without this the shortcut fired first —
    # it runs BEFORE entity extraction, so it never saw the two teams —
    # and answered with a site-wide list of 153 people.
    is_comparison = re.search(r"\b(compare|comparison|vs\.?|versus)\b|\bdifference between\b", q)
    # An ANALYTICAL attendance question is not a generic sweep. "Top
    # advisors by attendance rate" and "advisors with attendance below
    # 60 percent" both name a measure and want it ranked or filtered;
    # this shortcut's get_attendance_issues() can express neither, so it
    # answered a different question with a canned list. A ranking word, a
    # threshold, or a resolved attendance METRIC all mean the planner
    # should handle it.
    has_context = (
        entities.get("team")
        or re.search(r"\b(in|from|at|team|zone|region)\b", q)
        or has_specific_status
        or is_comparison
        # An analytical attendance question is a ranking or a filtered
        # set, which this shortcut cannot express — see _is_analytical.
        or _is_analytical(q)
    )
    if attendance_match and not has_context:
        return _Candidate("attendance_check", 0.9)
    return None


def _rule_leaderboard_sales(q: str, entities: dict) -> Optional[_Candidate]:
    if re.search(
        r"(top|highest|best|maximum|most|who.*(highest|most|best)|rank|ranking)"
        r".*(sales|sale|revenue|cleared|closed|performance)",
        q,
    ):
        return _Candidate("leaderboard", 0.9, {"metric": entities.get("metric", "mtd_cleared")})
    return None


def _rule_leaderboard_connects(q: str, entities: dict) -> Optional[_Candidate]:
    if re.search(
        r"(top|highest|best|most|who.*(highest|most|best))"
        r".*(connect|connections|new connect)",
        q,
    ):
        return _Candidate("leaderboard", 0.9, {"metric": entities.get("metric", "mtd_new_connect")})
    return None


def _rule_leaderboard_overdue(q: str, entities: dict) -> Optional[_Candidate]:
    if re.search(r"(worst|highest|most|maximum|top).*(overdue|overdues)", q):
        return _Candidate("leaderboard", 0.9, {"metric": entities.get("metric", "overdue")})
    return None


def _rule_company_summary(q: str, entities: dict) -> Optional[_Candidate]:
    if entities.get("company") and re.search(r"\b(company|doing|performing|performance|how is)\b", q):
        return _Candidate("company_summary", 0.75)
    return None


def _rule_team_summary(q: str, entities: dict) -> Optional[_Candidate]:
    if entities.get("team") and re.search(r"\b(team|group|department)\b", q):
        return _Candidate("team_summary", 0.75)
    return None


def _rule_advisor_strong(q: str, entities: dict) -> Optional[_Candidate]:
    if entities.get("advisor_name") and entities.get("advisor_match_score", 1.0) >= 0.65:
        return _Candidate("advisor_lookup", 0.7)
    return None


def _rule_advisor_weak(q: str, entities: dict) -> Optional[_Candidate]:
    if entities.get("advisor_name"):
        return _Candidate("advisor_lookup", 0.4)
    return None


# Same order as the original if/elif chain — preserved as the tie-break
# order (earliest-defined rule wins a confidence tie), so any input that
# used to match exactly one rule produces a byte-identical result.
_RULES: list[Callable[[str, dict], Optional[_Candidate]]] = [
    _rule_greeting,
    _rule_thanks,
    _rule_help,
    _rule_attendance,
    _rule_leaderboard_sales,
    _rule_leaderboard_connects,
    _rule_leaderboard_overdue,
    _rule_company_summary,
    _rule_team_summary,
    _rule_advisor_strong,
    _rule_advisor_weak,
]



# Part 12 (semantic retrieval expansion): a small, curated set of
# paraphrases for the ONLY intents nlu_pipeline.py actually routes on
# (SHORTCUT_INTENTS) — deliberately excludes leaderboard/*_summary/
# advisor_lookup, which the caveat above already says aren't consulted
# for live routing; adding semantic paraphrase coverage for those would
# just duplicate what the LLM semantic parser already owns.
_INTENT_EXEMPLARS: list[tuple[str, str]] = [
    ("hi", "greeting"), ("hello there", "greeting"), ("hey", "greeting"), ("good morning", "greeting"),
    ("good afternoon", "greeting"), ("yo", "greeting"), ("what's up", "greeting"), ("salam", "greeting"),
    ("thanks", "thanks"), ("thank you so much", "thanks"), ("appreciate it", "thanks"), ("cheers", "thanks"),
    ("thanks a lot", "thanks"), ("shukriya", "thanks"),
    ("help", "help"), ("what can you do", "help"), ("how does this work", "help"),
    ("give me a hand", "help"), ("show me the commands", "help"), ("what are my options", "help"),
    ("who missed attendance today", "attendance_check"), ("any attendance problems", "attendance_check"),
    ("who didn't show up today", "attendance_check"), ("attendance issues today", "attendance_check"),
    ("is everyone accounted for today", "attendance_check"),
]

entity_linker.register_exemplar_type("intent", lambda: _INTENT_EXEMPLARS)

# Shortcuts are short utterances by nature — this is a cheap pre-filter so
# the common case (a long analytical query that will never be a greeting)
# never pays for an embedding call; only a short message that ALSO failed
# every rule above reaches semantic_classify().
_MAX_SEMANTIC_INTENT_WORDS = 8
_SEMANTIC_INTENT_FLOOR = 0.75


def classify_intent(text: str, entities: dict) -> IntentResult:
    q = text.lower().strip()

    hits = [(i, c) for i, rule in enumerate(_RULES) if (c := rule(q, entities)) is not None]
    if hits:
        _, winner = max(hits, key=lambda pair: (pair[1].confidence, -pair[0]))
        result_entities = {**entities, **winner.entity_patch}
        return IntentResult(intent=winner.intent, confidence=winner.confidence, entities=result_entities)

    # Part 12: every rule missed — try semantic retrieval against the
    # shortcut-intent exemplars before giving up. Only ever reached when
    # nothing deterministic matched, so it can never override a
    # confident rule-based hit.
    if len(q.split()) <= _MAX_SEMANTIC_INTENT_WORDS:
        semantic = entity_linker.semantic_classify(q, "intent", top_k=1, floor=_SEMANTIC_INTENT_FLOOR)
        if semantic:
            return IntentResult(intent=semantic[0]["value"], confidence=semantic[0]["score"], entities=entities)

    return IntentResult(intent="unknown", confidence=0.0, entities=entities)


def find_missing_slots(result: IntentResult) -> list[str]:
    required = REQUIRED_SLOTS.get(result.intent, [])
    return [slot for slot in required if not result.entities.get(slot)]


# ----------------------------
# Unit Tests
# ----------------------------

def test_highest_sales_query():
    result = classify_intent("who has highest sales", {})
    assert result.intent == "leaderboard"
    assert result.entities["metric"] == "mtd_cleared"
    assert result.confidence >= 0.85
