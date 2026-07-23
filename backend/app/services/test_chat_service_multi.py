"""End-to-end multi-intent test (Part 8): a compound utterance splits into
independent sub-queries at nlu_pipeline.resolve() and chat_service stitches
their replies into labeled sections. rules_first + no LLM spy needed since
both halves resolve via the rule-based fast path deterministically."""

import pytest

from app.database.models import Advisor, Attendance, Performance, PerformancePeriod
from app.llm import conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def multi_db(db_session, monkeypatch):
    db_session.add_all([
        Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana"),
        Advisor(wid=2, name="Ali Raza", team="Downtown", company="IMARAT"),
    ])
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, cleared=900))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, cleared=500))
    db_session.add(Attendance(wid=2, biometric_status="Late"))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    # no network calls in this test — anything reaching the LLM fails
    # soft to the rule-based degrade path exactly as it would with a
    # quota-exhausted key, just without the real round trip
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def test_compound_query_splits_into_two_labeled_sections(multi_db):
    response = handle_chat_message(
        multi_db, "top advisors by revenue; who was late today", session_id="multi-1"
    )
    assert response["type"] == "multi"
    assert "1. " in response["reply"]
    assert "2. " in response["reply"]
    assert isinstance(response["data"], list)
    assert len(response["data"]) == 2


def test_ordinary_compound_filter_query_is_not_split(multi_db):
    # this must stay ONE query (existing multi-filter QueryIR behavior) —
    # the splitter must not fire on bare "and"
    response = handle_chat_message(
        multi_db, "advisors with revenue above 400 and attendance issues", session_id="multi-2"
    )
    assert response["type"] != "multi"
