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
from collections import deque
from dataclasses import dataclass, field

from app.core.config import settings
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
class PendingLevelChoice:
    """An in-flight "did you mean the BCM or the Advisor?" question.

    A name that grounds at more than one hierarchy level is ambiguous,
    and entity_extractor._detect_ambiguous_entity surfaces it so the
    pipeline can ask rather than guess. Asking was already right; what
    was missing is the record that a question is OUTSTANDING.

    Sibling of PendingPersonChoice, holding the same three things for the
    same reason: what was offered, and the ORIGINAL query, so once the
    user picks a level the question they actually asked is re-run instead
    of retyped.

    Distinct from PendingPersonChoice because the answers differ in kind
    — that one picks among PEOPLE ("which Yasir Ali"), this one picks
    among LEVELS ("the BCM or the advisor of that name"). Collapsing them
    would mean one matcher trying to read a wid and a level word out of
    the same reply.
    """
    value: str
    levels: list[str]
    original_text: str
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
    pending_level: PendingLevelChoice | None = None
    # Phase 5: the advisor the user settled on, carried for the rest of
    # the session. Once "which Yasir Ali?" has been answered, a later
    # "what about his overdue" must not re-ask — the identity question is
    # already settled, and re-asking is the same failure as guessing:
    # both ignore what the conversation already established.
    resolved_advisor_wid: int | None = None
    resolved_advisor_name: str | None = None
    # The recent conversation as MESSAGES, for the LLM prompt.
    #
    # Distinct from `last_ir` above, which is the structured record the
    # deterministic layer patches and merges. That layer resolves almost
    # every follow-up on its own and keeps doing so; this exists for the
    # references it cannot see — a reply's own wording, or a turn whose
    # subject never became a grounded entity.
    #
    # A bounded deque rather than a list: history that grows without a
    # ceiling is the failure mode of every naive context window, and the
    # ceiling belongs where the writes happen rather than at each reader.
    turns: deque = field(default_factory=lambda: deque(
        maxlen=max(1, settings.conversation_window_turns) * 2))

    def __post_init__(self):
        # `turns=None` from a caller carrying state forward means "start
        # fresh" — accepted so set() can pass `previous.turns if previous
        # else None` in the same shape as every other carried field.
        if self.turns is None:
            self.turns = deque(maxlen=max(1, settings.conversation_window_turns) * 2)


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
        # The conversation window survives this reset for the same reason
        # the resolved advisor does: it records WHAT HAS BEEN SAID, which
        # answering a new query does not invalidate. Rebuilding it empty
        # here wiped the window on every answered turn, so it only ever
        # held the turn in progress — the exact failure the docstring
        # above describes, in a second field.
        turns=previous.turns if previous else None,
        # Carried for the same reason as the turns and the resolved
        # advisor: an outstanding question is not answered by some OTHER
        # query resolving. Rebuilding this as None would drop the
        # question between the turn that asked it and the turn that
        # answers it — which is the whole failure being fixed.
        pending_level=previous.pending_level if previous else None,
    )


def clear_last_ir(session_id: str | None) -> None:
    """Forget the structured follow-up base, keeping everything else.

    For a turn that was ANSWERED but whose answer no QueryIR describes —
    an advisor profile card, a roster, a hierarchy chain. Those are served
    off the rule-based plan path, which never wrote `last_ir`, so the IR
    from some earlier turn stayed the follow-up base and the conversation
    silently forked: "Blue Area revenue" -> "what about Downtown?" ->
    "and Graana?" inherited Blue Area on turn 3, because turn 2 left no
    record of having moved.

    Writing a made-up IR for those answers would be worse — it would put
    a shape the user never asked for in front of the next follow-up. The
    honest state is "the conversation moved somewhere this structure
    cannot hold", and that is what this records.

    The identity (`resolved_advisor_*`), the message window and any
    outstanding question all survive, for the same reason they survive
    `set()`: none of them is invalidated by the topic moving.
    """
    state = _state(session_id)
    if state:
        state.last_ir = None
        state.pagination = None
        state.saved_at = time.time()


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


def get_pending_level(session_id: str | None) -> PendingLevelChoice | None:
    """The in-flight level question, or None once it has expired. Same
    TTL as every other pending: an answer to a question asked five
    minutes ago is not an answer."""
    state = _state(session_id)
    if not state or not state.pending_level:
        return None
    if time.time() - state.pending_level.asked_at > _PENDING_TTL_SECONDS:
        state.pending_level = None
        return None
    return state.pending_level


def set_pending_level(session_id: str | None, value: str, levels: list,
                      original_text: str) -> None:
    if not session_id:
        return
    state = _state(session_id) or SessionState()
    attempts = state.pending_level.attempts + 1 if state.pending_level else 1
    state.pending_level = PendingLevelChoice(
        value=value, levels=list(levels), original_text=original_text,
        asked_at=time.time(), attempts=attempts,
    )
    state.saved_at = time.time()
    _store[session_id] = state


def clear_pending_level(session_id: str | None) -> None:
    state = _state(session_id)
    if state:
        state.pending_level = None


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


# ---------------------------------------------------------------------
# The conversation window
# ---------------------------------------------------------------------


def record_turn(session_id: str | None, role: str, text: str) -> None:
    """Append one message. Silently ignored without a session — an
    anonymous request has no conversation to append to."""
    if not session_id or not text:
        return
    # _state() rather than _store.get(): an EXPIRED session is dropped
    # here exactly as every other reader drops it, so the turns that
    # follow start a genuinely new conversation.
    #
    # `saved_at` is deliberately NOT refreshed. The TTL means "a follow-up
    # this long later isn't really a follow-up", and appending a message
    # is not evidence against that — refreshing it let a new message
    # revive a conversation that had already expired, so "his team" bound
    # to a person named half an hour earlier. set() still refreshes on
    # every answered turn, which is the signal that the conversation is
    # actually live.
    state = _state(session_id)
    if state is None:
        state = SessionState()
        _store[session_id] = state
    state.turns.append((role, text.strip()))


def recent_turns(session_id: str | None) -> list[tuple[str, str]]:
    """The recent conversation, oldest first, within both budgets.

    Returns [] for an unknown or EXPIRED session, so a new conversation
    can never inherit another's context and a stale one is not revived —
    _state() applies the same TTL every other reader here uses.

    Trimmed from the END backwards: when the character budget binds, the
    most recent turns are the ones that survive, because a reference
    almost always points at the turn immediately before it.
    """
    state = _state(session_id)
    if state is None or not state.turns:
        return []

    budget = max(0, settings.conversation_window_chars)
    kept: list[tuple[str, str]] = []
    used = 0
    for role, text in reversed(state.turns):
        remaining = budget - used
        cost = len(text) + len(role) + 2
        if cost > remaining:
            # A single reply can exceed the whole budget on its own (a
            # 15-row leaderboard). Dropping it loses the turn a follow-up
            # most likely refers to; passing it whole crowds out the
            # schema the prompt needs. Truncated, the reference survives
            # and the budget holds.
            if kept or remaining < 40:
                break
            text = text[:remaining - len(role) - 5] + "…"
            cost = remaining
        kept.append((role, text))
        used += cost
    kept.reverse()
    return kept
