"""Part 10: confidence metadata (per-field breakdown, confidence_level,
ambiguity_reasons) gets persisted onto ChatLog.confidence_metadata for
every IR-bearing resolution — see ChatLog.confidence_metadata's docstring
in app/database/models.py for why this is its own column and not just
left nested inside resolved_ir."""

import json

import pytest

from app.database.models import Advisor, ChatLog, SalesFunnel
from app.llm import conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def confidence_db(db_session, monkeypatch):
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()
    # rule-based fast path only — no live LLM/Ollama dependency in this test
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    db_session.add(Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana"))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=10))
    db_session.commit()
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _last_chat_log(db):
    return db.query(ChatLog).order_by(ChatLog.id.desc()).first()


def test_ir_resolution_persists_confidence_metadata(confidence_db):
    handle_chat_message(confidence_db, "top 5 advisors by connects", session_id="conf-1")
    row = _last_chat_log(confidence_db)
    assert row.confidence_metadata is not None

    metadata = json.loads(row.confidence_metadata)
    for key in ("intent", "entities", "metric", "time", "filters", "level", "ambiguity_reasons"):
        assert key in metadata
    assert metadata["level"] == "high"
    assert metadata["ambiguity_reasons"] == []


def test_shortcut_resolution_has_no_confidence_metadata(confidence_db):
    handle_chat_message(confidence_db, "hello", session_id="conf-2")
    row = _last_chat_log(confidence_db)
    assert row.confidence_metadata is None
