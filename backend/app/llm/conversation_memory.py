"""
Conversation Memory (Part 5.7, extended by the NLU rework's P5/P6) —
per-session state for follow-ups and multi-turn clarification:

- last_ir: the last successfully resolved QueryIR, so "only Graana" /
  "top 5" can be parsed as a patch against the previous turn instead of
  from scratch (ir_patcher.py, or the LLM prompt's prior-IR block).
- pending: an in-flight clarification — the partial IR that failed
  validation plus which missing[] slots were asked about, so the user's
  next message ("revenue", "Blue Area") can be merged back into the SAME
  query instead of starting over. Shorter TTL than last_ir: an answer to
  a question asked 10 minutes ago isn't an answer anymore.

This is an in-process dict, matching entity_extractor.py's existing
per-process gazetteer cache pattern. It is NOT shared across workers —
for horizontal scaling swap this for Redis or a chat_sessions table,
keyed the same way ChatLog already keys session_id.
"""

import time
from dataclasses import dataclass, field

from app.llm.query_ir import QueryIR

_TTL_SECONDS = 60 * 30       # a follow-up more than 30 minutes later isn't really a follow-up
_PENDING_TTL_SECONDS = 60 * 5  # an answer to a 10-minute-old question isn't an answer
MAX_CLARIFY_ATTEMPTS = 2     # after this many re-asks, give generic help and stop nagging


@dataclass
class PendingClarification:
    partial_ir: QueryIR
    missing: list[str]
    asked_at: float = field(default_factory=time.time)
    attempts: int = 1


@dataclass
class SessionState:
    saved_at: float = field(default_factory=time.time)
    last_ir: QueryIR | None = None
    pending: PendingClarification | None = None


_store: dict[str, SessionState] = {}


def _state(session_id: str | None) -> SessionState | None:
    if not session_id or session_id not in _store:
        return None
    state = _store[session_id]
    if time.time() - state.saved_at > _TTL_SECONDS:
        del _store[session_id]
        return None
    return state


def get(session_id: str | None) -> QueryIR | None:
    state = _state(session_id)
    return state.last_ir if state else None


def set(session_id: str | None, ir: QueryIR) -> None:
    """A successfully resolved IR both becomes the new follow-up base and
    closes any clarification that was in flight — the question got
    answered by resolving the query."""
    if not session_id:
        return
    _store[session_id] = SessionState(saved_at=time.time(), last_ir=ir, pending=None)


def get_pending(session_id: str | None) -> PendingClarification | None:
    state = _state(session_id)
    if not state or not state.pending:
        return None
    if time.time() - state.pending.asked_at > _PENDING_TTL_SECONDS:
        state.pending = None
        return None
    return state.pending


def set_pending(session_id: str | None, partial_ir: QueryIR, missing: list[str]) -> None:
    """Record (or re-ask) a clarification. Re-asking bumps `attempts` so
    the pipeline can stop after MAX_CLARIFY_ATTEMPTS instead of looping."""
    if not session_id:
        return
    state = _state(session_id) or SessionState()
    attempts = state.pending.attempts + 1 if state.pending else 1
    state.pending = PendingClarification(
        partial_ir=partial_ir, missing=missing, asked_at=time.time(), attempts=attempts
    )
    state.saved_at = time.time()
    _store[session_id] = state


def clear_pending(session_id: str | None) -> None:
    state = _state(session_id)
    if state:
        state.pending = None
