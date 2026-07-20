"""Mode-inversion tests (WI-6): llm_first sends every analytical query to
the LLM; rules_first preserves the old rule-based fast path. The LLM is
monkeypatched throughout — no API key or network involved."""

import pytest

from app.llm import semantic_parser
from app.llm.query_ir import QueryIR


VALID_LLM_IR = {
    "intent": "leaderboard",
    "subject_level": "advisor",
    "subjects": [],
    "metric": {"key": "mtd_cleared", "confidence": 0.95},
    "filters": [],
    "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None},
    "sort": {"metric": "mtd_cleared", "direction": "desc"},
    "limit": 5,
    "group_by": None,
    "overall_confidence": 0.95,
}

# a query the OLD pipeline would answer rule-based without any LLM call
SIMPLE_QUERY = "top 5 advisors by revenue"


@pytest.fixture()
def llm_spy(monkeypatch):
    calls = []

    def fake_llm(prompt, schema, schema_name):
        calls.append(prompt)
        return fake_llm.response

    fake_llm.response = VALID_LLM_IR
    monkeypatch.setattr(semantic_parser, "call_llm_structured", fake_llm)
    return calls, fake_llm


def _set_mode(monkeypatch, mode):
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", mode)


def test_llm_first_calls_llm_even_for_simple_queries(db_session, llm_spy, monkeypatch):
    calls, _ = llm_spy
    _set_mode(monkeypatch, "llm_first")

    outcome = semantic_parser.parse(SIMPLE_QUERY, {"limit": 5}, db_session, session_id=None)

    assert len(calls) == 1
    assert outcome.used_llm is True
    assert outcome.ir.metric.key == "mtd_cleared"
    assert outcome.ir.nlu_mode == "llm_first"


def test_rules_first_skips_llm_for_simple_queries(db_session, llm_spy, monkeypatch):
    calls, _ = llm_spy
    _set_mode(monkeypatch, "rules_first")

    outcome = semantic_parser.parse(SIMPLE_QUERY, {"limit": 5}, db_session, session_id=None)

    assert calls == []
    assert outcome.used_llm is False
    assert outcome.ir.metric.key == "mtd_cleared"
    assert outcome.ir.nlu_mode == "rules_first"


def test_llm_first_degrades_to_rule_plan_when_llm_fails(db_session, llm_spy, monkeypatch):
    calls, fake_llm = llm_spy
    fake_llm.response = None
    _set_mode(monkeypatch, "llm_first")

    outcome = semantic_parser.parse(SIMPLE_QUERY, {"limit": 5}, db_session, session_id=None)

    assert len(calls) == 1
    assert outcome.used_llm is False              # served IR did NOT come from the LLM
    assert outcome.ir is not None
    assert outcome.ir.metric.key == "mtd_cleared"


def test_llm_first_unanswerable_query_asks_for_intent(db_session, llm_spy, monkeypatch):
    _, fake_llm = llm_spy
    fake_llm.response = None
    _set_mode(monkeypatch, "llm_first")

    outcome = semantic_parser.parse("xyzzy plugh", {}, db_session, session_id=None)

    assert outcome.ir is None
    assert outcome.missing == ["intent"]


def test_valid_ir_is_stored_in_conversation_memory(db_session, llm_spy, monkeypatch):
    from app.llm import conversation_memory

    _set_mode(monkeypatch, "llm_first")
    semantic_parser.parse(SIMPLE_QUERY, {"limit": 5}, db_session, session_id="s-1")

    stored = conversation_memory.get("s-1")
    assert isinstance(stored, QueryIR)
    assert stored.metric.key == "mtd_cleared"
