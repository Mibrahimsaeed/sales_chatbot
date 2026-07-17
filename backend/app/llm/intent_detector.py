# # """
# # Fast, cheap, first-pass intent classification.

# # This layer:
# # - does NOT call any LLM
# # - detects common business intents using rules
# # - extracts/uses entities provided by entity_extractor
# # - returns confidence so nlu_pipeline decides whether LLM fallback is needed
# # """

# # import re
# # from dataclasses import dataclass, field


# # @dataclass
# # class IntentResult:
# #     intent: str
# #     confidence: float
# #     entities: dict = field(default_factory=dict)
# #     missing_slots: list = field(default_factory=list)
# #     used_llm_fallback: bool = False


# # REQUIRED_SLOTS = {
# #     "advisor_lookup": ["advisor_name"],
# #     "team_summary": ["team"],
# #     "company_summary": ["company"],
# #     "leaderboard": ["metric"],
# #     "attendance_check": [],
# #     "greeting": [],
# #     "help": [],
# #     "unknown": [],
# # }


# # def classify_intent(text: str, entities: dict) -> IntentResult:

# #     q = text.lower().strip()


# #     # ----------------------------
# #     # Greeting
# #     # ----------------------------
# #     if re.search(r"^(hi|hello|hey|salam|assalam)\b", q):
# #         return IntentResult(
# #             intent="greeting",
# #             confidence=1.0,
# #             entities=entities
# #         )


# #     # ----------------------------
# #     # Help
# #     # ----------------------------
# #     if re.search(r"\b(help|what can you do|commands)\b", q):
# #         return IntentResult(
# #             intent="help",
# #             confidence=1.0,
# #             entities=entities
# #         )


# #     # ----------------------------
# #     # Attendance Queries
# #     # ----------------------------
# #     if re.search(
# #         r"(late|not marked|absent|missing|missed|biometric|login|attendance)",
# #         q
# #     ):
# #         return IntentResult(
# #             intent="attendance_check",
# #             confidence=0.9,
# #             entities=entities
# #         )


# #     # ----------------------------
# #     # Leaderboard - Sales / Revenue
# #     # Examples:
# #     # who has highest sales
# #     # top advisors by revenue
# #     # best performer
# #     # ----------------------------
# #     if re.search(
# #         r"(top|highest|best|maximum|most|who.*(highest|most|best)|rank|ranking)"
# #         r".*(sales|sale|revenue|cleared|closed|performance)",
# #         q
# #     ):
# #         entities.setdefault(
# #             "metric",
# #             "mtd_cleared"
# #         )

# #         return IntentResult(
# #             intent="leaderboard",
# #             confidence=0.9,
# #             entities=entities
# #         )


# #     # ----------------------------
# #     # Leaderboard - Connects
# #     # ----------------------------
# #     if re.search(
# #         r"(top|highest|best|most|who.*(highest|most|best))"
# #         r".*(connect|connections|new connect)",
# #         q
# #     ):
# #         entities.setdefault(
# #             "metric",
# #             "mtd_new_connect"
# #         )

# #         return IntentResult(
# #             intent="leaderboard",
# #             confidence=0.9,
# #             entities=entities
# #         )


# #     # ----------------------------
# #     # Leaderboard - Overdue
# #     # ----------------------------
# #     if re.search(
# #         r"(worst|highest|most|maximum|top)"
# #         r".*(overdue|overdues)",
# #         q
# #     ):
# #         entities.setdefault(
# #             "metric",
# #             "overdue"
# #         )

# #         return IntentResult(
# #             intent="leaderboard",
# #             confidence=0.9,
# #             entities=entities
# #         )


# #     # ----------------------------
# #     # Company Summary
# #     # ----------------------------
# #     if (
# #         entities.get("company")
# #         and re.search(
# #             r"\b(company|doing|performing|performance|how is)\b",
# #             q
# #         )
# #     ):
# #         return IntentResult(
# #             intent="company_summary",
# #             confidence=0.75,
# #             entities=entities
# #         )


# #     # ----------------------------
# #     # Team Summary
# #     # ----------------------------
# #     if (
# #         entities.get("team")
# #         and re.search(
# #             r"\b(team|group|department)\b",
# #             q
# #         )
# #     ):
# #         return IntentResult(
# #             intent="team_summary",
# #             confidence=0.75,
# #             entities=entities
# #         )


# #     # ----------------------------
# #     # Advisor Lookup
# #     # ----------------------------
# #     if (
# #         entities.get("advisor_name")
# #         and entities.get("advisor_match_score", 1.0) >= 0.65
# #     ):
# #         return IntentResult(
# #             intent="advisor_lookup",
# #             confidence=0.7,
# #             entities=entities
# #         )


# #     # Weak advisor match
# #     if entities.get("advisor_name"):

# #         return IntentResult(
# #             intent="advisor_lookup",
# #             confidence=0.4,
# #             entities=entities
# #         )


# #     # ----------------------------
# #     # Unknown
# #     # ----------------------------
# #     return IntentResult(
# #         intent="unknown",
# #         confidence=0.0,
# #         entities=entities
# #     )



# # def find_missing_slots(result: IntentResult) -> list[str]:

# #     required = REQUIRED_SLOTS.get(
# #         result.intent,
# #         []
# #     )

# #     return [
# #         slot
# #         for slot in required
# #         if not result.entities.get(slot)
# #     ]



# # # ----------------------------
# # # Unit Tests (temporary)
# # # Move these later to tests/llm/
# # # ----------------------------

# # def test_highest_sales_query():

# #     result = classify_intent(
# #         "who has highest sales",
# #         {}
# #     )

# #     assert result.intent == "leaderboard"
# #     assert result.entities["metric"] == "mtd_cleared"
# #     assert result.confidence >= 0.85


# """
# Fast, cheap, first-pass intent classification.

# This layer:
# - does NOT call any LLM
# - detects common business intents using rules
# - extracts/uses entities provided by entity_extractor
# - returns confidence so nlu_pipeline decides whether LLM fallback is needed

# Important:
# - Generic attendance queries are handled here.
# - Specific attendance filters (team + status) are passed to query_planner.
#   Example:
#       "show not marked people in Blue Area"
#       "who was late in Downtown"
# """

# import re
# from dataclasses import dataclass, field


# @dataclass
# class IntentResult:
#     intent: str
#     confidence: float
#     entities: dict = field(default_factory=dict)
#     missing_slots: list = field(default_factory=list)
#     used_llm_fallback: bool = False


# REQUIRED_SLOTS = {
#     "advisor_lookup": ["advisor_name"],
#     "team_summary": ["team"],
#     "company_summary": ["company"],
#     "leaderboard": ["metric"],
#     "attendance_check": [],
#     "greeting": [],
#     "help": [],
#     "unknown": [],
# }


# def classify_intent(text: str, entities: dict) -> IntentResult:

#     q = text.lower().strip()


#     # ----------------------------
#     # Greeting
#     # ----------------------------
#     if re.search(r"^(hi|hello|hey|salam|assalam)\b", q):
#         return IntentResult(
#             intent="greeting",
#             confidence=1.0,
#             entities=entities
#         )


#     # ----------------------------
#     # Help
#     # ----------------------------
#     if re.search(r"\b(help|what can you do|commands)\b", q):
#         return IntentResult(
#             intent="help",
#             confidence=1.0,
#             entities=entities
#         )


#     # ----------------------------
#     # Attendance Queries
#     #
#     # IMPORTANT:
#     # Do NOT capture specific filters here.
#     #
#     # These should go through query_planner:
#     #
#     # "show not marked people in Blue Area"
#     # "who was late in AMD"
#     # "give absent advisors from Downtown"
#     #
#     # Only generic attendance questions become shortcuts:
#     #
#     # "who was late today"
#     # "show attendance issues"
#     # ----------------------------

#     attendance_match = re.search(
#         r"(late|not marked|absent|missing|missed|biometric|login|attendance)",
#         q
#     )

#     # detect if query contains a specific location/team context
#     has_context = (
#         entities.get("team")
#         or re.search(
#             r"\b(in|from|at|team|zone|region)\b",
#             q
#         )
#     )

#     if attendance_match and not has_context:

#         return IntentResult(
#             intent="attendance_check",
#             confidence=0.9,
#             entities=entities
#         )


#     # ----------------------------
#     # Leaderboard - Sales / Revenue
#     #
#     # Examples:
#     # who has highest sales
#     # top advisors by revenue
#     # best performer
#     # ----------------------------
#     if re.search(
#         r"(top|highest|best|maximum|most|who.*(highest|most|best)|rank|ranking)"
#         r".*(sales|sale|revenue|cleared|closed|performance)",
#         q
#     ):

#         entities.setdefault(
#             "metric",
#             "mtd_cleared"
#         )

#         return IntentResult(
#             intent="leaderboard",
#             confidence=0.9,
#             entities=entities
#         )


#     # ----------------------------
#     # Leaderboard - Connects
#     # ----------------------------
#     if re.search(
#         r"(top|highest|best|most|who.*(highest|most|best))"
#         r".*(connect|connections|new connect)",
#         q
#     ):

#         entities.setdefault(
#             "metric",
#             "mtd_new_connect"
#         )

#         return IntentResult(
#             intent="leaderboard",
#             confidence=0.9,
#             entities=entities
#         )


#     # ----------------------------
#     # Leaderboard - Overdue
#     # ----------------------------
#     if re.search(
#         r"(worst|highest|most|maximum|top)"
#         r".*(overdue|overdues)",
#         q
#     ):

#         entities.setdefault(
#             "metric",
#             "overdue"
#         )

#         return IntentResult(
#             intent="leaderboard",
#             confidence=0.9,
#             entities=entities
#         )


#     # ----------------------------
#     # Company Summary
#     # ----------------------------
#     if (
#         entities.get("company")
#         and re.search(
#             r"\b(company|doing|performing|performance|how is)\b",
#             q
#         )
#     ):

#         return IntentResult(
#             intent="company_summary",
#             confidence=0.75,
#             entities=entities
#         )


#     # ----------------------------
#     # Team Summary
#     # ----------------------------
#     if (
#         entities.get("team")
#         and re.search(
#             r"\b(team|group|department)\b",
#             q
#         )
#     ):

#         return IntentResult(
#             intent="team_summary",
#             confidence=0.75,
#             entities=entities
#         )


#     # ----------------------------
#     # Advisor Lookup
#     # ----------------------------
#     if (
#         entities.get("advisor_name")
#         and entities.get("advisor_match_score", 1.0) >= 0.65
#     ):

#         return IntentResult(
#             intent="advisor_lookup",
#             confidence=0.7,
#             entities=entities
#         )


#     # Weak advisor match
#     if entities.get("advisor_name"):

#         return IntentResult(
#             intent="advisor_lookup",
#             confidence=0.4,
#             entities=entities
#         )


#     # ----------------------------
#     # Unknown
#     # ----------------------------
#     return IntentResult(
#         intent="unknown",
#         confidence=0.0,
#         entities=entities
#     )



# def find_missing_slots(result: IntentResult) -> list[str]:

#     required = REQUIRED_SLOTS.get(
#         result.intent,
#         []
#     )

#     return [
#         slot
#         for slot in required
#         if not result.entities.get(slot)
#     ]



# # ----------------------------
# # Unit Tests
# # ----------------------------

# def test_highest_sales_query():

#     result = classify_intent(
#         "who has highest sales",
#         {}
#     )

#     assert result.intent == "leaderboard"
#     assert result.entities["metric"] == "mtd_cleared"
#     assert result.confidence >= 0.85

"""
Fast, cheap, first-pass intent classification + slot extraction.

This layer:
- does NOT call any LLM
- detects common business intents using rules
- extracts slots (direction, limit, time period, attendance status)
  in addition to intent
- returns confidence so nlu_pipeline decides whether LLM fallback is needed

The intent detector now extracts slots in parallel with intent. The planner
in query_planner.py uses these slots, but it also re-extracts them
defensively — both layers are safe to run independently.
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


# Required slots per intent, used by find_missing_slots().
# Kept here for the contract that nlu_pipeline depends on.
REQUIRED_SLOTS = {
    "advisor_lookup": ["advisor_name"],
    "team_summary": ["team"],
    "company_summary": ["company"],
    "leaderboard": ["metric"],
    "attendance_check": [],
    "attendance_filter": ["attendance_status"],
    "greeting": [],
    "help": [],
    "unknown": [],
    # Special intent — means "let the planner handle it" (e.g. for
    # 'show late in Blue Area', the intent detector catches the status
    # but does not know the team — nlu_pipeline then routes to the planner).
    "unresolved_for_planner": [],
}


# ---------------------------------------------------------------------------
# Slot extraction helpers (run on the cleaned text)
# ---------------------------------------------------------------------------

_DIRECTION_KEYWORDS = {
    "lowest":  ["worst", "lowest", "least", "bottom"],
    "highest": ["top", "best", "highest", "most", "leaderboard", "biggest"],
}

_TIME_PERIOD_KEYWORDS = {
    "today":     ["today"],
    "this_week": ["this week", "weekly"],
    "ytd":       ["ytd", "year to date", "year-to-date", "this year"],
    "mtd":       ["mtd", "month to date", "month-to-date", "this month"],
}

_ATTENDANCE_STATUS_KEYWORDS = {
    "late":       ["late"],
    "absent":     ["absent", "not present", "no show", "didn't come", "did not come"],
    "not_marked": ["not marked", "no biometric", "missing login", "no login",
                   "not logged in", "didn't mark", "did not mark"],
}


def _extract_direction(text: str) -> str:
    q = text.lower()
    for direction, kws in _DIRECTION_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return direction
    return "highest"


def _extract_time_period(text: str) -> str:
    q = text.lower()
    for period, kws in _TIME_PERIOD_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return period
    return "MTD"


def _extract_attendance_status(text: str) -> str | None:
    q = text.lower()
    for status, kws in _ATTENDANCE_STATUS_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return status
    return None


def _has_location_context(text: str) -> bool:
    """Did the user mention a specific team/area/region (or a preposition hinting at one)?"""
    q = text.lower()
    if re.search(r"\b(in|from|at)\s+[A-Za-z]", q):
        return True
    return False


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def classify_intent(text: str, entities: dict) -> IntentResult:
    q = text.lower().strip()

    # Always extract universal slots so the planner has them regardless
    # of which intent we pick.
    entities.setdefault("direction", _extract_direction(q))
    entities.setdefault("time_period", _extract_time_period(q))

    # ----------------------------
    # Greeting
    # ----------------------------
    if re.search(r"^(hi|hello|hey|salam|assalam|salamun?)\b", q):
        return IntentResult(intent="greeting", confidence=1.0, entities=entities)

    # ----------------------------
    # Help
    # ----------------------------
    if re.search(r"\b(help|what can you do|commands|how do i)\b", q):
        return IntentResult(intent="help", confidence=1.0, entities=entities)

    # ----------------------------
    # Attendance — specific filter (status + location)
    # "show late in Blue Area" / "who was absent in Downtown"
    # Hand off to the planner by emitting a special "unresolved_for_planner" intent
    # with the attendance_status slot already filled in.
    # ----------------------------
    attendance_status = _extract_attendance_status(q)
    if attendance_status and (_has_location_context(q) or entities.get("team")
                              or entities.get("location") or entities.get("region")):
        entities["attendance_status"] = attendance_status
        return IntentResult(
            intent="unresolved_for_planner",
            confidence=0.85,
            entities=entities,
        )

    # ----------------------------
    # Attendance — generic shortcut
    # "who was late today" / "show attendance issues"
    # ----------------------------
    if attendance_status and not _has_location_context(q):
        return IntentResult(intent="attendance_check", confidence=0.9, entities=entities)

    # ----------------------------
    # Leaderboard — Sales / Revenue
    # ----------------------------
    if re.search(
        r"(top|highest|best|maximum|most|who.*(highest|most|best)|rank|ranking|worst|lowest|least|bottom)"
        r".*(sales|sale|revenue|cleared|closed|performance)",
        q
    ):
        entities.setdefault("metric", "mtd_cleared")
        return IntentResult(intent="leaderboard", confidence=0.9, entities=entities)

    # ----------------------------
    # Leaderboard — Connects
    # ----------------------------
    if re.search(
        r"(top|highest|best|most|who.*(highest|most|best))"
        r".*(connect|connections|new connect)",
        q
    ):
        entities.setdefault("metric", "mtd_new_connect")
        return IntentResult(intent="leaderboard", confidence=0.9, entities=entities)

    # ----------------------------
    # Leaderboard — Overdue
    # ----------------------------
    if re.search(
        r"(worst|highest|most|maximum|top)"
        r".*(overdue|overdues)",
        q
    ):
        entities.setdefault("metric", "overdue")
        return IntentResult(intent="leaderboard", confidence=0.9, entities=entities)

    # ----------------------------
    # Leaderboard — Target Achievement
    # ----------------------------
    if re.search(
        r"(top|highest|best|most|worst|lowest|least|target|achievement|performance|on target|behind target)",
        q
    ) and re.search(r"\b(target|achievement|on target|behind target|% of target)\b", q):
        entities.setdefault("metric", "target_achievement")
        return IntentResult(intent="leaderboard", confidence=0.85, entities=entities)

    # ----------------------------
    # Company Summary
    # ----------------------------
    if (
        entities.get("company")
        and re.search(r"\b(company|doing|performing|performance|how is|how are)\b", q)
    ):
        return IntentResult(intent="company_summary", confidence=0.75, entities=entities)

    # ----------------------------
    # Team Summary
    # ----------------------------
    if (
        entities.get("team")
        and re.search(r"\b(team|group|department|doing|performing|how is|how are)\b", q)
    ):
        return IntentResult(intent="team_summary", confidence=0.75, entities=entities)

    # ----------------------------
    # Advisor Lookup
    # ----------------------------
    if (
        entities.get("advisor_name")
        and entities.get("advisor_match_score", 1.0) >= 0.65
    ):
        return IntentResult(intent="advisor_lookup", confidence=0.7, entities=entities)

    # Weak advisor match
    if entities.get("advisor_name"):
        return IntentResult(intent="advisor_lookup", confidence=0.4, entities=entities)

    # ----------------------------
    # Unknown
    # ----------------------------
    return IntentResult(intent="unknown", confidence=0.0, entities=entities)


def find_missing_slots(result: IntentResult) -> list[str]:
    required = REQUIRED_SLOTS.get(result.intent, [])
    return [slot for slot in required if not result.entities.get(slot)]


# ---------------------------------------------------------------------------
# Tests (kept inline; move to tests/llm/ later)
# ---------------------------------------------------------------------------

def test_highest_sales_query():
    result = classify_intent("who has highest sales", {})
    assert result.intent == "leaderboard"
    assert result.entities["metric"] == "mtd_cleared"
    assert result.confidence >= 0.85


def test_lowest_overdue_query():
    result = classify_intent("worst 5 by overdue", {})
    assert result.intent == "leaderboard"
    assert result.entities["metric"] == "overdue"
    assert result.entities["direction"] == "lowest"


def test_target_achievement_query():
    result = classify_intent("top 5 by target achievement", {})
    assert result.intent == "leaderboard"
    assert result.entities["metric"] == "target_achievement"


def test_specific_attendance_filter_routes_to_planner():
    result = classify_intent("show late people in Blue Area", {})
    assert result.intent == "unresolved_for_planner"
    assert result.entities["attendance_status"] == "late"


def test_generic_attendance_stays_as_shortcut():
    result = classify_intent("who was late today", {})
    assert result.intent == "attendance_check"
    assert result.entities["time_period"] == "today"
