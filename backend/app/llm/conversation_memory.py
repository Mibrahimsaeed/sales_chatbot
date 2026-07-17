"""
Conversation Memory (Part 5.7) — stores the last resolved QueryIR per
session so follow-ups ("what about last month", "same for Downtown") can
be parsed as a patch/diff against the previous turn instead of from
scratch (Root Cause #8).

This is an in-process dict, matching entity_extractor.py's existing
per-process gazetteer cache pattern. It is NOT shared across workers —
for horizontal scaling swap this for Redis or the `chat_sessions` table
mentioned in the redesign roadmap (Phase 5), keyed the same way ChatLog
already keys session_id. Kept intentionally small: this is a reference
implementation of the *shape* (session_id -> last IR), not a production
session store.
"""

import time

from app.llm.query_ir import QueryIR

_TTL_SECONDS = 60 * 30  # a follow-up more than 30 minutes later isn't really a follow-up
_store: dict[str, tuple[float, QueryIR]] = {}


def get(session_id: str | None) -> QueryIR | None:
    if not session_id or session_id not in _store:
        return None
    saved_at, ir = _store[session_id]
    if time.time() - saved_at > _TTL_SECONDS:
        del _store[session_id]
        return None
    return ir


def set(session_id: str | None, ir: QueryIR) -> None:
    if not session_id:
        return
    _store[session_id] = (time.time(), ir)