"""Multi-turn clarification slot-filling (WI-8): ask -> answer,
ask -> ignore-with-new-query, ask -> unhelpful answers -> give up."""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod
from app.llm import conversation_memory, entity_extractor, nlu_pipeline, semantic_parser


CLARIFY_IR = {
    # what the LLM emits for an ambiguous query: no metric, low confidence
    "intent": "clarify",
    "subject_level": "advisor",
    "subjects": [],
    "metric": None,
    "filters": [],
    "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None},
    "sort": {"metric": None, "direction": "desc"},
    "limit": 10,
    "group_by": None,
    "overall_confidence": 0.4,
}

VERY_LOW_CONFIDENCE_IR = {
    # genuinely unclear — Part 10's "low" tier: reject and ask to rephrase
    # instead of setting a pending clarification for a specific slot
    "intent": "clarify",
    "subject_level": "advisor",
    "subjects": [],
    "metric": None,
    "filters": [],
    "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None},
    "sort": {"metric": None, "direction": "desc"},
    "limit": 10,
    "group_by": None,
    "overall_confidence": 0.2,
}

LEADERBOARD_IR = {
    "intent": "leaderboard",
    "subject_level": "advisor",
    "subjects": [],
    "metric": {"key": "overdue", "confidence": 0.95},
    "filters": [],
    "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None},
    "sort": {"metric": "overdue", "direction": "desc"},
    "limit": 10,
    "group_by": None,
    "overall_confidence": 0.95,
}


@pytest.fixture()
def clarify_db(db_session, monkeypatch):
    db_session.add_all([
        Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana"),
        Advisor(wid=2, name="Ali Raza", team="Downtown", company="IMARAT"),
    ])
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, cleared=900))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "llm_first")
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()


def _mock_llm(monkeypatch, response):
    calls = []

    def fake_llm(prompt, schema, schema_name):
        calls.append(prompt)
        return response

    monkeypatch.setattr(semantic_parser, "call_llm_structured", fake_llm)
    return calls


def test_clarify_then_slot_answer_resolves_same_query(clarify_db, monkeypatch):
    _mock_llm(monkeypatch, CLARIFY_IR)
    session = "cl-1"

    # turn 1: ambiguous — chatbot asks which metric
    r1 = nlu_pipeline.resolve("who is struggling", clarify_db, session_id=session)
    assert r1.kind == "clarify"
    assert "metric" in r1.clarify_message
    assert conversation_memory.get_pending(session) is not None

    # turn 2: bare metric answer fills the slot — no new LLM parse needed
    r2 = nlu_pipeline.resolve("revenue", clarify_db, session_id=session)
    assert r2.kind == "ir"
    assert r2.ir.metric.key == "mtd_cleared"
    assert conversation_memory.get_pending(session) is None


def test_new_query_while_pending_clears_it_and_answers_fresh(clarify_db, monkeypatch):
    _mock_llm(monkeypatch, CLARIFY_IR)
    session = "cl-2"

    r1 = nlu_pipeline.resolve("who is struggling", clarify_db, session_id=session)
    assert r1.kind == "clarify"

    # user ignores the question and asks something self-standing
    _mock_llm(monkeypatch, LEADERBOARD_IR)
    r2 = nlu_pipeline.resolve("show top advisors by overdue amounts today", clarify_db, session_id=session)
    assert r2.kind == "ir"
    assert r2.ir.metric.key == "overdue"
    assert conversation_memory.get_pending(session) is None


def test_unhelpful_answers_reask_then_give_up(clarify_db, monkeypatch):
    _mock_llm(monkeypatch, CLARIFY_IR)
    session = "cl-3"

    r1 = nlu_pipeline.resolve("who is struggling", clarify_db, session_id=session)
    assert r1.kind == "clarify"

    # unhelpful short reply — re-asks (attempt 2)
    r2 = nlu_pipeline.resolve("hmm dunno", clarify_db, session_id=session)
    assert r2.kind == "clarify"
    assert "metric" in r2.clarify_message

    # still unhelpful — gives up gracefully instead of looping
    r3 = nlu_pipeline.resolve("whatever man", clarify_db, session_id=session)
    assert r3.kind == "clarify"
    assert "start fresh" in r3.clarify_message
    assert conversation_memory.get_pending(session) is None


def test_low_confidence_ir_is_rejected_not_asked_about(clarify_db, monkeypatch):
    _mock_llm(monkeypatch, VERY_LOW_CONFIDENCE_IR)
    session = "cl-5"

    r1 = nlu_pipeline.resolve("uhh something about numbers i guess", clarify_db, session_id=session)
    assert r1.kind == "clarify"
    assert r1.ir.confidence_level == "low"
    # the rephrase message, not a targeted "which metric" question
    assert "start fresh" in r1.clarify_message
    # never set a pending slot for a query this unclear — a follow-up
    # answer would likely just compound the confusion
    assert conversation_memory.get_pending(session) is None


def test_pending_expires_after_ttl(clarify_db, monkeypatch):
    _mock_llm(monkeypatch, CLARIFY_IR)
    session = "cl-4"

    nlu_pipeline.resolve("who is struggling", clarify_db, session_id=session)
    pending = conversation_memory.get_pending(session)
    assert pending is not None

    # age the question past the pending TTL
    pending.asked_at -= 6 * 60
    assert conversation_memory.get_pending(session) is None
