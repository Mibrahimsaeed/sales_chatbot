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


def test_attendance_phrasing():
    result = classify_intent("who was late today", {})
    assert result.intent == "attendance_check"
    assert result.confidence >= 0.8


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