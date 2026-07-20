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
"""

import re
from dataclasses import dataclass, field


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


def classify_intent(text: str, entities: dict) -> IntentResult:

    q = text.lower().strip()


    # ----------------------------
    # Greeting
    # ----------------------------
    if re.search(r"^(hi|hello|hey|salam|assalam)\b", q):
        return IntentResult(
            intent="greeting",
            confidence=1.0,
            entities=entities
        )


    # ----------------------------
    # Thanks
    # ----------------------------
    if re.search(r"^(thanks|thank you|thankyou|thx|shukriya)\b", q):
        return IntentResult(
            intent="thanks",
            confidence=1.0,
            entities=entities
        )


    # ----------------------------
    # Help
    # ----------------------------
    if re.search(r"\b(help|what can you do|commands)\b", q):
        return IntentResult(
            intent="help",
            confidence=1.0,
            entities=entities
        )


    # ----------------------------
    # Attendance Queries
    #
    # IMPORTANT:
    # Do NOT capture specific filters here.
    #
    # These should go through query_planner:
    #
    # "show not marked people in Blue Area"
    # "who was late in AMD"
    # "give absent advisors from Downtown"
    #
    # Only generic attendance questions become shortcuts:
    #
    # "who was late today"
    # "show attendance issues"
    # ----------------------------

    attendance_match = re.search(
        r"(late|not marked|absent|missing|missed|biometric|login|attendance)",
        q
    )

    # detect if query contains a specific location/team context
    has_context = (
        entities.get("team")
        or re.search(
            r"\b(in|from|at|team|zone|region)\b",
            q
        )
    )

    if attendance_match and not has_context:

        return IntentResult(
            intent="attendance_check",
            confidence=0.9,
            entities=entities
        )


    # ----------------------------
    # Leaderboard - Sales / Revenue
    #
    # Examples:
    # who has highest sales
    # top advisors by revenue
    # best performer
    # ----------------------------
    if re.search(
        r"(top|highest|best|maximum|most|who.*(highest|most|best)|rank|ranking)"
        r".*(sales|sale|revenue|cleared|closed|performance)",
        q
    ):

        entities.setdefault(
            "metric",
            "mtd_cleared"
        )

        return IntentResult(
            intent="leaderboard",
            confidence=0.9,
            entities=entities
        )


    # ----------------------------
    # Leaderboard - Connects
    # ----------------------------
    if re.search(
        r"(top|highest|best|most|who.*(highest|most|best))"
        r".*(connect|connections|new connect)",
        q
    ):

        entities.setdefault(
            "metric",
            "mtd_new_connect"
        )

        return IntentResult(
            intent="leaderboard",
            confidence=0.9,
            entities=entities
        )


    # ----------------------------
    # Leaderboard - Overdue
    # ----------------------------
    if re.search(
        r"(worst|highest|most|maximum|top)"
        r".*(overdue|overdues)",
        q
    ):

        entities.setdefault(
            "metric",
            "overdue"
        )

        return IntentResult(
            intent="leaderboard",
            confidence=0.9,
            entities=entities
        )


    # ----------------------------
    # Company Summary
    # ----------------------------
    if (
        entities.get("company")
        and re.search(
            r"\b(company|doing|performing|performance|how is)\b",
            q
        )
    ):

        return IntentResult(
            intent="company_summary",
            confidence=0.75,
            entities=entities
        )


    # ----------------------------
    # Team Summary
    # ----------------------------
    if (
        entities.get("team")
        and re.search(
            r"\b(team|group|department)\b",
            q
        )
    ):

        return IntentResult(
            intent="team_summary",
            confidence=0.75,
            entities=entities
        )


    # ----------------------------
    # Advisor Lookup
    # ----------------------------
    if (
        entities.get("advisor_name")
        and entities.get("advisor_match_score", 1.0) >= 0.65
    ):

        return IntentResult(
            intent="advisor_lookup",
            confidence=0.7,
            entities=entities
        )


    # Weak advisor match
    if entities.get("advisor_name"):

        return IntentResult(
            intent="advisor_lookup",
            confidence=0.4,
            entities=entities
        )


    # ----------------------------
    # Unknown
    # ----------------------------
    return IntentResult(
        intent="unknown",
        confidence=0.0,
        entities=entities
    )



def find_missing_slots(result: IntentResult) -> list[str]:

    required = REQUIRED_SLOTS.get(
        result.intent,
        []
    )

    return [
        slot
        for slot in required
        if not result.entities.get(slot)
    ]



# ----------------------------
# Unit Tests
# ----------------------------

def test_highest_sales_query():

    result = classify_intent(
        "who has highest sales",
        {}
    )

    assert result.intent == "leaderboard"
    assert result.entities["metric"] == "mtd_cleared"
    assert result.confidence >= 0.85


