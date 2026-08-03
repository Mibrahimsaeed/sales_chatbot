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
class PaginationState:
    """Part 8: the cursor for a "Show More" continuation — the IR that
    produced the current result set, how many rows have been shown so
    far (the next page's OFFSET), and the true total capped by the IR's
    own limit (if the user asked for an explicit "top N", the cursor
    stops at N even if more rows technically match)."""
    ir: QueryIR
    offset: int
    capped_total: int
    saved_at: float = field(default_factory=time.time)


@dataclass
class PendingPersonChoice:
    """Phase 5: an in-flight "which Yasir Ali did you mean?" question.

    Holds the candidates that were offered plus the ORIGINAL query, so
    once the user picks one the question they actually asked can be
    re-run against the chosen wid — rather than making them retype it."""
    candidates: list
    original_text: str
    asked_at: float = field(default_factory=time.time)
    attempts: int = 1


@dataclass
class SessionState:
    saved_at: float = field(default_factory=time.time)
    last_ir: QueryIR | None = None
    pending: PendingClarification | None = None
    pagination: PaginationState | None = None
    pending_person: PendingPersonChoice | None = None
    # Phase 5: the advisor the user settled on, carried for the rest of
    # the session. Once "which Yasir Ali?" has been answered, a later
    # "what about his overdue" must not re-ask — the identity question is
    # already settled, and re-asking is the same failure as guessing:
    # both ignore what the conversation already established.
    resolved_advisor_wid: int | None = None
    resolved_advisor_name: str | None = None


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
    answered by resolving the query.

    The resolved advisor identity (Phase 5) deliberately SURVIVES this
    reset: it answers "who are we talking about", which a new analytical
    query doesn't invalidate. Rebuilding SessionState from scratch here
    would silently drop the person the user just disambiguated and make
    the very next follow-up ask again."""
    if not session_id:
        return
    previous = _state(session_id)
    _store[session_id] = SessionState(
        saved_at=time.time(),
        last_ir=ir,
        pending=None,
        resolved_advisor_wid=previous.resolved_advisor_wid if previous else None,
        resolved_advisor_name=previous.resolved_advisor_name if previous else None,
    )


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


# ---------------------------------------------------------------------
# Phase 5 — person disambiguation state
# ---------------------------------------------------------------------

def get_pending_person(session_id: str | None) -> PendingPersonChoice | None:
    state = _state(session_id)
    if not state or not state.pending_person:
        return None
    if time.time() - state.pending_person.asked_at > _PENDING_TTL_SECONDS:
        state.pending_person = None
        return None
    return state.pending_person


def set_pending_person(session_id: str | None, candidates: list, original_text: str) -> None:
    if not session_id:
        return
    state = _state(session_id) or SessionState()
    attempts = state.pending_person.attempts + 1 if state.pending_person else 1
    state.pending_person = PendingPersonChoice(
        candidates=list(candidates), original_text=original_text,
        asked_at=time.time(), attempts=attempts,
    )
    state.saved_at = time.time()
    _store[session_id] = state


def clear_pending_person(session_id: str | None) -> None:
    state = _state(session_id)
    if state:
        state.pending_person = None


def set_resolved_advisor(session_id: str | None, wid: int, name: str | None = None) -> None:
    """Remember which person the conversation is about. Set both when the
    user answers a disambiguation AND when a query resolves to exactly
    one person on its own, so a follow-up has a subject either way."""
    if not session_id:
        return
    state = _state(session_id) or SessionState()
    state.resolved_advisor_wid = wid
    state.resolved_advisor_name = name
    state.pending_person = None      # the identity question is settled
    state.saved_at = time.time()
    _store[session_id] = state


def get_resolved_advisor(session_id: str | None) -> tuple[int, str | None] | None:
    state = _state(session_id)
    if not state or state.resolved_advisor_wid is None:
        return None
    return state.resolved_advisor_wid, state.resolved_advisor_name


def get_pagination(session_id: str | None) -> PaginationState | None:
    state = _state(session_id)
    if not state or not state.pagination:
        return None
    if time.time() - state.pagination.saved_at > _PENDING_TTL_SECONDS:
        state.pagination = None
        return None
    return state.pagination


def set_pagination(session_id: str | None, ir: QueryIR, offset: int, capped_total: int) -> None:
    """Records a fresh "Show More" cursor for the query that just
    resolved. Called by chat_service._dispatch_ir right after `set()`
    stores the new last_ir — set() already wiped any previous pagination
    state (it replaces the whole SessionState), so this is establishing
    the cursor for the CURRENT query, not overwriting a stale one."""
    if not session_id:
        return
    state = _state(session_id) or SessionState()
    state.pagination = PaginationState(ir=ir, offset=offset, capped_total=capped_total)
    state.saved_at = time.time()
    _store[session_id] = state


def advance_pagination(session_id: str | None, new_offset: int) -> None:
    state = _state(session_id)
    if state and state.pagination:
        state.pagination.offset = new_offset
        state.pagination.saved_at = time.time()


def clear_pagination(session_id: str | None) -> None:
    state = _state(session_id)
    if state:
        state.pagination = None
