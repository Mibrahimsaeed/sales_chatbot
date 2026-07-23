"""
Tests the rule-based layer only (no DB, no LLM) — entities are passed in
directly rather than extracted, since extraction requires a DB session.
This is intentionally the fast, no-dependency layer to test; entity
extraction and the LLM fallback need integration tests against a real
(or fixture) DB and are not covered here.
"""

from app.llm.intent_detector import classify_intent, find_missing_slots


def test_greeting():
    result = classify_intent("hello", {})
    assert result.intent == "greeting"
    assert result.confidence == 1.0


def test_leaderboard_revenue_phrasing():
    result = classify_intent("top 5 advisors by revenue", {"limit": 5})
    assert result.intent == "leaderboard"
    assert result.entities["metric"] == "mtd_cleared"
    assert result.confidence >= 0.85


def test_leaderboard_overdue_phrasing():
    result = classify_intent("worst overdue teams", {})
    assert result.intent == "leaderboard"
    assert result.entities["metric"] == "overdue"


def test_generic_attendance_phrasing_with_no_specific_status():
    result = classify_intent("show attendance issues today", {})
    assert result.intent == "attendance_check"
    assert result.confidence >= 0.8


def test_specific_status_phrasing_does_not_hit_generic_shortcut():
    # bug fix: "who was late today" (or "not marked"/"absent"/"present")
    # must NOT hit the generic attendance_check shortcut — that shortcut's
    # get_attendance_issues() only excludes "On Time" and returns EVERY
    # other status mixed together, so "who was not marked today" used to
    # come back with late arrivals in it too. A specific status word must
    # fall through to entity extraction + query_planner's attendance_filter
    # action, which calls get_attendance_by_status() and filters exactly.
    for phrase in ("who was late today", "who was not marked today", "who was absent today"):
        result = classify_intent(phrase, {})
        assert result.intent != "attendance_check", phrase


def test_advisor_lookup_with_entity():
    result = classify_intent("tell me about Waqar Haider", {"advisor_name": "Waqar Haider", "advisor_match_score": 0.9})
    assert result.intent == "advisor_lookup"
    assert result.confidence >= 0.65


def test_missing_slot_detection():
    result = classify_intent("top advisors", {})  # "top" with no metric keyword matched
    result.entities = {}
    missing = find_missing_slots(result)
    if result.intent == "leaderboard":
        assert "metric" in missing


def test_unknown_with_no_entities():
    result = classify_intent("asdkjalksjd nonsense query", {})
    assert result.intent == "unknown"
    assert result.confidence == 0.0


# ---- Part 8: intent ranking ----

def test_higher_confidence_rule_wins_when_multiple_rules_match():
    # "top advisors by revenue" fires leaderboard (0.9) AND, because
    # advisor_name happens to be present, the weak advisor-lookup rule
    # (0.4) too — the ranked winner must be the higher-confidence one,
    # not whichever rule happened to run first.
    result = classify_intent("top advisors by revenue", {"advisor_name": "Ali Raza"})
    assert result.intent == "leaderboard"
    assert result.confidence == 0.9
    assert result.entities["metric"] == "mtd_cleared"


def test_confidence_tie_breaks_to_earlier_rule():
    # company_summary and team_summary both score 0.75 here — the earlier
    # rule (company_summary) wins the tie, matching the original
    # first-match-wins behavior.
    result = classify_intent(
        "how is the team and company doing",
        {"team": "Downtown", "company": "Graana"},
    )
    assert result.intent == "company_summary"
    assert result.confidence == 0.75


    