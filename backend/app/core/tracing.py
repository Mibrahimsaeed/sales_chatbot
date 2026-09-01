"""
Per-request tracing (Phase 7).

WHY THIS EXISTS — from the pipeline audit: every wrong-person and wrong-
hierarchy bug found had to be diagnosed by hand-instrumenting the pipeline
in a REPL, because the only per-request record was
`log.debug("resolved '…' -> kind=…")` plus a QueryIR. In production those
failures were invisible: the bot returned a confident answer about the
wrong person and nothing after the fact distinguished it from a correct
one.

A trace captures the full decision chain for one message:

    user query
      -> extracted entities
      -> resolved WID + the candidates that were considered
      -> planner decision
      -> QueryIR
      -> generated SQL (statement, params, rowcount, duration)
      -> returned row count
      -> final response

That is deliberately enough to REPRODUCE a failure: the candidates say
what identity resolution was choosing between, the plan says what the
router decided the question was, and the SQL is the literal statement the
database ran.

Design notes:

- Request-scoped via contextvars, not a global — a trace belongs to one
  message, and FastAPI serves requests on a threadpool where a module
  global would interleave two users' traces.
- Every entry point is FAIL-SOFT. Tracing is diagnostics; it must never
  be the reason a chat response fails. All public functions swallow their
  own errors, and `traced()` re-raises only what the wrapped work raised.
- SQL is captured by a SQLAlchemy event listener (install_sql_capture),
  so it records what was ACTUALLY executed rather than what a call site
  believed it was building — which is the distinction that matters when
  the complaint is "the answer doesn't match the question".
"""

from __future__ import annotations

import json
from functools import lru_cache
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.logger import get_logger

log = get_logger("core.tracing")

# Bounds so one pathological request can't produce an unbounded log line
# or an unbounded DB row.
MAX_SQL_STATEMENTS = 25
MAX_SQL_CHARS = 2000
MAX_CANDIDATES = 10
MAX_REPLY_CHARS = 500


@dataclass
class SqlEvent:
    statement: str
    params: Any = None
    row_count: int | None = None
    duration_ms: float | None = None


@dataclass
class RequestTrace:
    trace_id: str
    query: str
    session_id: str | None = None
    started_at: float = field(default_factory=time.time)

    entities: dict = field(default_factory=dict)
    resolved_wid: int | None = None
    resolved_name: str | None = None
    matched_span: str | None = None
    candidates: list[dict] = field(default_factory=list)
    resolution_status: str | None = None

    plan: dict | None = None
    ir: dict | None = None
    sql: list[SqlEvent] = field(default_factory=list)

    planner: dict | None = None
    # LLM-first parse attempt: whether the model was asked, whether it
    # answered, which model and prompt produced it, and — when it did not
    # — what served the query instead. Written by
    # semantic_parser.parse() via record_llm_parse().
    #
    # This is the one record that distinguishes "the LLM understood this"
    # from "the LLM was never asked" from "the LLM failed and the rule
    # plan answered". All three previously looked identical in the trace,
    # which made an LLM-path regression indistinguishable from an outage.
    llm: dict | None = None
    row_count: int | None = None
    response_type: str | None = None
    response_preview: str | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "session_id": self.session_id,
            "entities": self.entities,
            "identity": {
                "resolved_wid": self.resolved_wid,
                "resolved_name": self.resolved_name,
                "matched_span": self.matched_span,
                "status": self.resolution_status,
                "candidates": self.candidates,
            },
            "planner": self.planner,
            "llm": self.llm,
            "plan": self.plan,
            "ir": self.ir,
            "sql": [asdict(e) for e in self.sql],
            "row_count": self.row_count,
            "response_type": self.response_type,
            "response_preview": self.response_preview,
            "duration_ms": self.duration_ms,
        }


_current: ContextVar[RequestTrace | None] = ContextVar("request_trace", default=None)


def current() -> RequestTrace | None:
    return _current.get()


@contextmanager
def traced(query: str, session_id: str | None = None):
    """Wrap one chat message. Yields the trace so the caller can populate
    it; emits the structured log line on exit, including when the wrapped
    work raised — a trace for a request that BLEW UP is the most valuable
    one there is, so the emit happens in `finally`."""
    trace = RequestTrace(trace_id=uuid.uuid4().hex[:12], query=query, session_id=session_id)
    token = _current.set(trace)
    try:
        yield trace
    finally:
        try:
            trace.duration_ms = round((time.time() - trace.started_at) * 1000, 1)
            _emit(trace)
        except Exception:
            log.exception("Trace emit failed — the request itself is unaffected")
        _current.reset(token)


def _emit(trace: RequestTrace) -> None:
    log.info(
        "chat_trace %s",
        json.dumps(
            {
                "trace_id": trace.trace_id,
                "query": trace.query,
                "session": trace.session_id,
                "wid": trace.resolved_wid,
                "identity": trace.resolution_status,
                "candidates": len(trace.candidates),
                "plan": (trace.plan or {}).get("action"),
                "level": (trace.plan or {}).get("level"),
                "score": (trace.plan or {}).get("intent_score"),
                "runner_up": (trace.plan or {}).get("runner_up"),
                "intent": (trace.ir or {}).get("intent"),
                "sql_count": len(trace.sql),
                "rows": trace.row_count,
                "response": trace.response_type,
                "ms": trace.duration_ms,
            },
            default=str,
        ),
    )


# ---------------------------------------------------------------------
# Population helpers — each is a no-op when there is no active trace, so
# call sites never need an `if tracing.current()` guard.
# ---------------------------------------------------------------------

def _safe(fn):
    def wrapper(*args, **kwargs):
        trace = _current.get()
        if trace is None:
            return
        try:
            fn(trace, *args, **kwargs)
        except Exception:
            log.debug("trace population failed", exc_info=True)
    return wrapper


# Entity keys worth recording. The full dict carries embedding candidate
# lists and resolution objects that are large and not JSON-friendly; these
# are the ones that explain a routing decision.
# The level keys are DERIVED, so a level added to the registry is traced
# without anyone remembering to add it here. The hand-written list had
# gone stale: it named the retired `business_center` and omitted bcm,
# office and region — meaning a misroute involving any of those three was
# invisible in the very trace built to explain misroutes.
_FIXED_TRACE_KEYS = (
    "advisor_name", "advisor_wid", "advisor_wids", "advisor_match_score",
    "advisor_ambiguous", "ambiguous_entity",
    "metric", "limit", "period", "attendance_status", "thresholds",
)


@lru_cache(maxsize=1)
def _traced_entity_keys() -> frozenset[str]:
    """Which entity keys appear in the trace.

    Computed LAZILY and cached: app.database.session imports this module
    during initialisation, so importing hierarchy at module level here
    would cycle (tracing -> hierarchy -> models -> session -> tracing).

    The level keys are derived, so a level added to the registry is
    traced without anyone remembering to add it. The hand-written list
    had gone stale — it named the retired `business_center` and omitted
    bcm, office and region, meaning a misroute involving any of those
    three was invisible in the very trace built to explain misroutes.
    """
    from app.llm import hierarchy

    keys = set(_FIXED_TRACE_KEYS)
    for level in hierarchy.GROUP_LEVELS:
        keys.add(level)
        plural = hierarchy.LEVEL_ENTITY_KEYS.get(level)
        if plural:
            keys.add(plural)
    return frozenset(keys)


@_safe
def record_entities(trace: RequestTrace, entities: dict) -> None:
    trace.entities = {
        k: v for k, v in (entities or {}).items()
        if k in _traced_entity_keys() and v not in (None, [], {})
    }


@_safe
def record_identity(trace: RequestTrace, resolution) -> None:
    """The advisor_resolver result — status, chosen wid, and the
    candidates it was choosing between. The candidate list is the piece
    that makes a wrong-person report diagnosable: it shows what the
    alternatives were and why one was (or wasn't) picked."""
    trace.resolution_status = getattr(resolution, "status", None)
    trace.resolved_wid = getattr(resolution, "wid", None)
    trace.resolved_name = getattr(resolution, "name", None)
    trace.matched_span = getattr(resolution, "matched_text", None)
    trace.candidates = [
        {"wid": c.wid, "name": c.name, "team": c.team, "score": round(c.score, 2)}
        for c in (getattr(resolution, "candidates", None) or [])[:MAX_CANDIDATES]
    ]


@_safe
def record_resolved_wid(trace: RequestTrace, wid: int | None, name: str | None = None) -> None:
    if wid is not None:
        trace.resolved_wid = wid
        if name:
            trace.resolved_name = name


@_safe
def record_plan(trace: RequestTrace, plan) -> None:
    if plan is None:
        return
    trace.plan = {
        "action": getattr(plan, "action", None),
        "level": getattr(plan, "level", None),
        "entity_value": getattr(plan, "entity_value", None),
        "entity_wid": getattr(plan, "entity_wid", None),
        "metric": getattr(plan, "metric", None),
        "limit": getattr(plan, "limit", None),
        "ascending": getattr(plan, "ascending", None),
        "flat": getattr(plan, "flat", None),
        "reason": getattr(plan, "reason", "") or None,
        "candidate_count": len(getattr(plan, "person_candidates", None) or []),
        # Why this intent won, and what came second. Intent selection is
        # scored rather than ordered, so a misroute is a score comparison
        # to inspect rather than control flow to re-read.
        "intent_score": getattr(plan, "intent_score", None),
        "intent_evidence": getattr(plan, "intent_evidence", None) or [],
        "runner_up": getattr(plan, "runner_up", None),
    }


@_safe
def record_ir(trace: RequestTrace, ir) -> None:
    if ir is None:
        return
    trace.ir = json.loads(ir.model_dump_json())


@_safe
def record_sql(trace: RequestTrace, statement: str, params=None,
               row_count: int | None = None, duration_ms: float | None = None) -> None:
    if len(trace.sql) >= MAX_SQL_STATEMENTS:
        return
    trace.sql.append(SqlEvent(
        statement=" ".join(str(statement).split())[:MAX_SQL_CHARS],
        params=_shrink_params(params),
        row_count=row_count,
        duration_ms=duration_ms,
    ))


def _shrink_params(params):
    """Bind params, bounded. Kept because "which SQL ran" is only half of
    reproducing a failure — the other half is what it ran WITH."""
    try:
        if params is None:
            return None
        text = json.dumps(params, default=str)
        return json.loads(text) if len(text) <= 500 else text[:500] + "…"
    except Exception:
        return str(params)[:500]


@_safe
def record_planner(trace: RequestTrace, prompt: str | None, raw, plan,
                   rejected: str | None, elapsed_ms: float | None) -> None:
    """The LLM planner's full exchange. The PROMPT is recorded because a
    planner misroute is usually a prompt problem, and reconstructing which
    prompt produced a given plan after the fact is otherwise guesswork —
    the ontology and gazetteer it embeds change as the data changes."""
    trace.planner = {
        "prompt_chars": len(prompt) if prompt else 0,
        "prompt": (prompt[-1200:] if prompt else None),
        "raw_response": raw,
        "validated_plan": plan,
        "rejected": rejected,
        "elapsed_ms": elapsed_ms,
    }


@_safe
def record_llm_parse(trace: RequestTrace, *, attempted: bool, succeeded: bool,
                     model: str | None = None, prompt_hash: str | None = None,
                     prompt_tokens: int | None = None,
                     validation_missing: list | None = None,
                     fallback_used: bool = False,
                     fallback_reason: str | None = None) -> None:
    """The LLM-first parse attempt, in the shape a debugger needs.

    Deliberately records the prompt HASH rather than the prompt: the
    prompt embeds the live gazetteer and can run to 28k characters, and
    the question a trace has to answer is "was this the same prompt as
    the run that worked?", which a hash answers exactly. The full text is
    still available under CHAT_AUDIT_DEBUG (app/core/audit.py) when a
    specific prompt genuinely needs reading.

    `attempted=False` is as important as a failure: it means a routing
    gate served the query before the model was consulted, which is the
    state that used to be invisible.
    """
    trace.llm = {
        "attempted": attempted,
        "succeeded": succeeded,
        "model": model,
        "prompt_hash": prompt_hash,
        "prompt_tokens": prompt_tokens,
        "validation_missing": list(validation_missing or []),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }


@_safe
def record_response(trace: RequestTrace, response: dict) -> None:
    trace.response_type = response.get("type")
    reply = response.get("reply")
    if isinstance(reply, str):
        trace.response_preview = reply[:MAX_REPLY_CHARS]
    data = response.get("data")
    if isinstance(data, list):
        trace.row_count = len(data)
    elif isinstance(data, dict):
        trace.row_count = 1
    elif data is None:
        trace.row_count = 0


# ---------------------------------------------------------------------
# SQL capture
# ---------------------------------------------------------------------

_sql_capture_installed = False


def install_sql_capture(engine=None) -> None:
    """Records every statement executed while a trace is active.

    A listener rather than per-call-site logging: it captures what the
    database ACTUALLY ran, including SQL from services that never touch
    query_compiler, and it cannot drift out of sync with the code the way
    hand-placed log lines do. Statements executed with no active trace
    (startup, ETL, the ChatLog insert itself) are ignored, so this costs
    nothing outside a chat request.

    Registered on the SQLAlchemy Engine CLASS, not one instance: a
    diagnostic that silently captures nothing depending on which engine
    happened to run the query is worse than no diagnostic at all, and
    binding to a single instance did exactly that — the test suite builds
    its own in-memory engine, so SQL capture was invisible there while
    appearing to work in production. `engine` is accepted and ignored for
    call-site readability at the point where the app engine is created.
    Idempotent."""
    global _sql_capture_installed
    if _sql_capture_installed:
        return

    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    @event.listens_for(Engine, "before_cursor_execute")
    def _before(conn, cursor, statement, parameters, context, executemany):
        if _current.get() is not None:
            conn.info["_trace_started"] = time.time()

    @event.listens_for(Engine, "after_cursor_execute")
    def _after(conn, cursor, statement, parameters, context, executemany):
        if _current.get() is None:
            return
        started = conn.info.pop("_trace_started", None)
        duration = round((time.time() - started) * 1000, 1) if started else None
        row_count = getattr(cursor, "rowcount", None)
        record_sql(
            statement,
            params=parameters,
            row_count=row_count if row_count is not None and row_count >= 0 else None,
            duration_ms=duration,
        )

    _sql_capture_installed = True
