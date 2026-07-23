"""Part 8: pagination cursor lifecycle in conversation_memory.py — no
existing test file covered this module directly before."""

from app.llm import conversation_memory
from app.llm.query_ir import MetricRef, QueryIR, Sort


def _ir():
    return QueryIR(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="total_connects"),
        sort=Sort(metric="total_connects", direction="desc"),
        limit=None,
    )


def setup_function():
    conversation_memory._store.clear()


def test_set_and_get_pagination_roundtrip():
    conversation_memory.set_pagination("s1", _ir(), offset=15, capped_total=583)
    state = conversation_memory.get_pagination("s1")
    assert state.offset == 15
    assert state.capped_total == 583


def test_no_pagination_state_returns_none():
    assert conversation_memory.get_pagination("nonexistent") is None


def test_advance_pagination_updates_offset():
    conversation_memory.set_pagination("s1", _ir(), offset=15, capped_total=583)
    conversation_memory.advance_pagination("s1", 30)
    assert conversation_memory.get_pagination("s1").offset == 30


def test_clear_pagination_removes_state():
    conversation_memory.set_pagination("s1", _ir(), offset=15, capped_total=583)
    conversation_memory.clear_pagination("s1")
    assert conversation_memory.get_pagination("s1") is None


def test_pagination_expires_after_ttl(monkeypatch):
    conversation_memory.set_pagination("s1", _ir(), offset=15, capped_total=583)
    state = conversation_memory._store["s1"]
    state.pagination.saved_at -= conversation_memory._PENDING_TTL_SECONDS + 1
    assert conversation_memory.get_pagination("s1") is None


def test_a_fresh_resolved_query_wipes_previous_pagination_state():
    # set() (called whenever a new IR resolves) replaces the whole
    # SessionState — a stale "Show More" cursor from a DIFFERENT query
    # must not survive into the new one.
    conversation_memory.set_pagination("s1", _ir(), offset=15, capped_total=583)
    conversation_memory.set("s1", _ir())
    assert conversation_memory.get_pagination("s1") is None
