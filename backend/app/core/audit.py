"""
Human-readable chat audit log (diagnostics only).

WHY THIS EXISTS — the complaint is "responses are inconsistent", and the
existing observability can't settle that question on its own:
`core/tracing.py` emits ONE structured JSON line per request (great for
grepping, unreadable when you want to eyeball twenty answers in a row),
and the ChatLog table stores the decision chain but not the literal
prompt text that was sent to the model. Inconsistency between two runs of
the same question almost always comes down to the prompt differing —
the gazetteer, the ontology, and the conversation-memory context are all
interpolated at call time and change as the data changes. So the prompt
is the artifact worth keeping verbatim.

This module writes a flat, readable block per user query:

    ====================
    CHAT AUDIT
    ====================
    User Query:   ...
    Timestamp:    ...
    ... every LLM prompt sent while answering it, complete and unabridged
    Final Response: ...
    ====================

Design rules, all of them deliberate:

- OFF BY DEFAULT. Gated on settings.chat_audit_debug (CHAT_AUDIT_DEBUG).
  With the flag off, every public function here returns immediately, no
  file is opened and no log directory is created.
- NO BUSINESS LOGIC. Nothing here inspects or alters a response; call
  sites only hand over what they already computed.
- FAIL-SOFT, like tracing.py. An audit log must never be the reason a
  chat response fails, so every public function swallows its own errors.
- Request-scoped via contextvars, not a module global: FastAPI serves
  requests on a threadpool, where a global would interleave two users'
  prompts into one block.
- Prompts are recorded BEFORE inference, so a call that times out or is
  refused still leaves its prompt behind — those are exactly the runs
  that produce a surprising answer via the fail-soft degrade path.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("core.audit")

BANNER = "=" * 20
SEPARATOR = "-" * 20

# One audit file, rotated, so an enabled flag on a busy instance can't
# fill the disk. 5 MB x 5 keeps a few thousand full-prompt blocks.
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5

# Result rows are summarised by count rather than dumped: this file is for
# reading the QUESTION and the ANSWER TEXT, and the rows themselves are
# already persisted on ChatLog / in the trace.
MAX_REPLY_CHARS = 4000


class _Entry:
    """The in-flight audit record for one user query."""

    def __init__(self, query: str, session_id: str | None):
        self.query = query
        self.session_id = session_id
        self.timestamp = datetime.now().astimezone()
        self.started = self.timestamp.timestamp()
        self.prompts: list[dict] = []
        self.response: dict | None = None

        # --- rule-based execution trace ---
        # `decisions` is an ordered log of every branch the pipeline took
        # WITH the reason, appended as it happens; the rest are the
        # artifacts each stage produced. All stay None/empty for a query
        # that never touches the rule-based path, and the RULE-BASED
        # AUDIT block is then omitted rather than printed empty.
        self.decisions: list[dict] = []
        self.entities: dict | None = None
        self.plan = None
        self.ir_json: str | None = None
        self.advisor: dict | None = None
        self.formatter: str | None = None
        self.rule_path: bool = False
        # SQL is snapshotted at record_response() time, NOT read at format
        # time: the audit context wraps the tracing context in
        # chat_service, so by the time this entry is formatted the trace's
        # contextvar has already been reset and the statements would read
        # as zero — which is indistinguishable from a query that genuinely
        # ran no SQL, the worst possible ambiguity for this log.
        self.sql: list = []


_current: ContextVar[_Entry | None] = ContextVar("audit_entry", default=None)

_file_logger: logging.Logger | None = None
_file_logger_path: Path | None = None


def enabled() -> bool:
    """The single gate. Read live (not cached at import) so a test or a
    REPL can flip settings.chat_audit_debug without reimporting."""
    try:
        return bool(settings.chat_audit_debug)
    except Exception:
        return False


def log_path() -> Path:
    """Absolute path of the audit file. A relative CHAT_AUDIT_DIR is
    resolved against the backend/ root rather than the process's working
    directory, so uvicorn started from the repo root and a script started
    from backend/ write to the same place instead of two."""
    configured = Path(settings.chat_audit_dir).expanduser()
    if not configured.is_absolute():
        configured = Path(__file__).resolve().parents[2] / configured
    return configured / "chat_audit.log"


def _writer() -> logging.Logger | None:
    """Lazily builds the file logger on FIRST write, never at import — so
    a disabled flag leaves no directory and no file behind.

    Keyed on the RESOLVED PATH, not merely "have I built one yet": the
    underlying logging.Logger is a process-wide singleton, so a cached
    handler would keep writing to the first path this process ever
    resolved even after CHAT_AUDIT_DIR changed. In production the path is
    fixed and this never fires; it is what makes the module honest under
    a changed setting, where the alternative is writes silently landing
    in a file nobody is reading.

    propagate=False keeps these multi-line blocks out of the root handler
    that core/logger.py installs; otherwise every audit block would also
    land in the app's normal stdout log with a timestamp prefix per line.
    """
    global _file_logger, _file_logger_path
    try:
        path = log_path()
        if _file_logger is not None and _file_logger_path == path:
            return _file_logger

        os.makedirs(path.parent, exist_ok=True)
        writer = logging.getLogger("chat_audit_file")
        writer.setLevel(logging.INFO)
        writer.propagate = False
        for stale in list(writer.handlers):
            writer.removeHandler(stale)
            try:
                stale.close()
            except Exception:
                pass
        handler = RotatingFileHandler(
            path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        # Bare message: the block carries its own timestamp and framing,
        # and a per-line prefix would wreck the prompt text.
        handler.setFormatter(logging.Formatter("%(message)s"))
        writer.addHandler(handler)
        _file_logger = writer
        _file_logger_path = path
        return _file_logger
    except Exception:
        log.exception("Chat audit log could not be opened — auditing is skipped")
        return None


@contextmanager
def audit_query(query: str, session_id: str | None = None):
    """Wrap one user query. Prompts recorded inside this block, plus the
    final response, are emitted as a single audit block on exit —
    including when the wrapped work raised, since a request that BLEW UP
    is the one most worth reading.

    Re-entrant: a nested call (handle_show_more() reached through
    _dispatch()) keeps the OUTER entry rather than starting a second
    block, so one user message is always exactly one block.
    """
    if not enabled() or _current.get() is not None:
        yield None
        return

    entry = _Entry(query, session_id)
    token = _current.set(entry)
    try:
        yield entry
    finally:
        try:
            _emit(entry)
        except Exception:
            log.exception("Chat audit emit failed — the request itself is unaffected")
        _current.reset(token)


def record_prompt(
    prompt: str,
    purpose: str,
    model: str | None = None,
    messages: list[dict] | None = None,
) -> None:
    """The COMPLETE prompt text, stored verbatim and unabridged — the
    whole point of this module. Called before inference, so a timed-out
    or refused call still leaves its prompt in the audit.

    `messages` is the literal payload handed to the provider, so the log
    shows which ROLES were transmitted rather than what the prompt text
    reads as. That distinction matters here: the role instruction and the
    output contract are inlined into a single user message, so a reader
    who assumed a real system role would be wrong."""
    entry = _current.get()
    if entry is None or not enabled():
        return
    try:
        entry.prompts.append({
            "purpose": purpose,
            "model": model,
            "at": datetime.now().astimezone(),
            "prompt": prompt,
            "roles": [m.get("role") for m in (messages or [])],
            "raw_response": None,
        })
    except Exception:
        log.debug("audit prompt capture failed", exc_info=True)


def record_llm_response(raw) -> None:
    """Attaches the model's raw output to the prompt recorded most
    recently. Same prompt + different output across two runs is the exact
    signature of the inconsistency being investigated, so the pair has to
    be readable side by side."""
    entry = _current.get()
    if entry is None or not enabled() or not entry.prompts:
        return
    try:
        entry.prompts[-1]["raw_response"] = raw
    except Exception:
        log.debug("audit response capture failed", exc_info=True)


def decision(stage: str, chose: str, why: str) -> None:
    """One branch point, with the reason it went that way.

    Every call site states WHY in the terms the code actually branched on
    (the predicate, the score, the missing keyword) rather than a
    restatement of what was chosen — a decision log that only records the
    outcome cannot tell you where a requirement was dropped."""
    entry = _current.get()
    if entry is None or not enabled():
        return
    try:
        entry.decisions.append({
            "stage": stage,
            "chose": chose,
            "why": why,
            "ms": round((datetime.now().timestamp() - entry.started) * 1000, 1),
        })
    except Exception:
        log.debug("audit decision capture failed", exc_info=True)


def record_entities(entities: dict) -> None:
    entry = _current.get()
    if entry is None or not enabled():
        return
    try:
        entry.entities = dict(entities or {})
    except Exception:
        log.debug("audit entity capture failed", exc_info=True)


def record_plan(plan) -> None:
    """The QueryPlan object itself. Held by reference and read only at
    format time — the audit never mutates it."""
    entry = _current.get()
    if entry is None or not enabled():
        return
    entry.plan = plan


def record_ir(ir) -> None:
    entry = _current.get()
    if entry is None or not enabled() or ir is None:
        return
    try:
        entry.ir_json = ir.model_dump_json()
    except Exception:
        log.debug("audit IR capture failed", exc_info=True)


def record_advisor(wid: int | None, name: str | None, status: str | None = None,
                   candidates: list | None = None) -> None:
    """Identity resolution outcome. `candidates` matters as much as the
    winner: "resolved to the wrong person" and "resolved to the only
    person" look identical without it."""
    entry = _current.get()
    if entry is None or not enabled():
        return
    try:
        entry.advisor = {
            "wid": wid,
            "name": name,
            "status": status,
            "candidates": [
                {"wid": getattr(c, "wid", None), "name": getattr(c, "name", None),
                 "team": getattr(c, "team", None), "score": round(getattr(c, "score", 0.0), 2)}
                for c in (candidates or [])[:10]
            ],
        }
    except Exception:
        log.debug("audit advisor capture failed", exc_info=True)


def record_formatter(name: str, why: str = "") -> None:
    """Which reply formatter produced the text. Recorded at the branch
    that calls it, not inferred from the response type — several branches
    share a response type, so inference would be a guess."""
    entry = _current.get()
    if entry is None or not enabled():
        return
    entry.formatter = name
    if why:
        decision("formatter", name, why)


def mark_rule_path(why: str, chose: str = "rule_based_path") -> None:
    """Flags this request as answered WITHOUT an LLM call, which is what
    causes the RULE-BASED AUDIT block to print.

    Every deterministic exit qualifies, not just the planner one: a
    shortcut, a "show more", and a deterministic IR patch are equally
    LLM-free, and a trace that only covered the planner exit would leave
    the most common queries of all with no explanation at all."""
    entry = _current.get()
    if entry is None or not enabled():
        return
    entry.rule_path = True
    decision("routing", chose, why)


def record_response(response: dict) -> None:
    """The final response dict as returned to the caller — recorded, not
    inspected or modified."""
    entry = _current.get()
    if entry is None or not enabled():
        return
    try:
        entry.response = response
        # Snapshot the SQL WHILE the request trace is still open — see
        # _Entry.sql. Called here because record_response() is the last
        # thing to happen inside both contexts.
        entry.sql = _sql_events()
    except Exception:
        log.debug("audit response capture failed", exc_info=True)


def _row_summary(data) -> str:
    if data is None:
        return "none"
    if isinstance(data, list):
        return f"{len(data)} row(s)"
    if isinstance(data, dict):
        return "1 record"
    return type(data).__name__


def _format_prompt_breakdown(record: dict) -> list[str]:
    """Renders one prompt as the five requested views: System, Developer,
    Retrieved Context, Conversation History, and the final prompt as
    actually transmitted.

    The breakdown is a partition of the prompt, not a copy of parts of it
    — so printing every section IS printing the complete prompt, once.
    `reconstructs exactly` states whether that's still true; on a `no`,
    the raw prompt is dumped verbatim underneath so the log never shows a
    breakdown that has silently drifted from the text that was sent.
    """
    prompt = record.get("prompt") or ""
    try:
        # Imported lazily: app.llm.llm_client imports THIS module, and a
        # core -> llm import at module scope would invert the layering and
        # invite a cycle. Diagnostics can afford the deferred lookup.
        from app.llm import prompt_inspector

        breakdown = prompt_inspector.segment(prompt, roles_sent=record.get("roles") or [])
    except Exception:
        log.debug("prompt segmentation failed", exc_info=True)
        return ["[prompt breakdown unavailable — raw prompt follows]", prompt, ""]

    out: list[str] = []
    for category in prompt_inspector.CATEGORY_ORDER:
        sections = [s for s in breakdown.by_category(category) if s.lines]
        if not sections:
            # Stated explicitly rather than omitted: "this app sends no
            # developer prompt" is itself a finding worth reading.
            out.append(f"### {category}: (none)")
            out.append("")
            continue
        out.append(f"### {category} ({breakdown.category_chars(category)} chars)")
        for section in sections:
            out += [f"--- [{section.label}] ---", section.text]
        out.append("")

    roles = breakdown.roles_sent or ["(not captured)"]
    out += [
        "### Final Prompt sent to the LLM",
        f"roles transmitted: {roles} — "
        + (
            "no system/developer role message; every section above is inlined "
            "into one user message"
            if roles == ["user"]
            else "see roles above"
        ),
        f"total chars: {len(prompt)}   sha256[:16]: {breakdown.sha256}",
        f"sections above reconstruct it exactly: {'yes' if breakdown.is_lossless else 'NO'}",
        "",
    ]
    if not breakdown.is_lossless:
        out += ["[breakdown drifted — raw prompt verbatim]", prompt, ""]
    return out


def _sql_events() -> list:
    """SQL is read off the ACTIVE REQUEST TRACE rather than captured
    again here. core/tracing.py already installs a SQLAlchemy engine-level
    listener that records what the database actually ran; a second
    listener would double-instrument every query to produce the same
    list. Empty when tracing isn't active (e.g. a unit test calling the
    audit directly)."""
    try:
        from app.core import tracing

        trace = tracing.current()
        return list(trace.sql) if trace is not None else []
    except Exception:
        return []


def _format_rule_audit(entry: _Entry) -> list[str]:
    """The rule-based execution trace: every stage, in execution order,
    with the reason each branch was taken.

    Read top to bottom it answers one question — at which stage did the
    information the query needed stop being carried forward?
    """
    plan = entry.plan
    response = entry.response or {}
    out = [
        "",
        BANNER,
        "RULE-BASED AUDIT",
        BANNER,
        "",
        f"User Query:  {entry.query}",
        f"Timestamp:   {entry.timestamp.isoformat(timespec='seconds')}",
        "",
        "Routing Decision:",
    ]

    if entry.decisions:
        for d in entry.decisions:
            out.append(f"  [{d['ms']:>7} ms] {d['stage']}: {d['chose']}")
            out.append(f"              WHY: {d['why']}")
    else:
        out.append("  (no decision points recorded)")

    action = getattr(plan, "action", None)
    out += [
        "",
        f"Detected Intent: {action or '-'}",
        f"Confidence:      {getattr(plan, 'intent_score', None)}"
        f"   (runner-up: {getattr(plan, 'runner_up', None) or 'none'})",
        f"Plan Reason:     {getattr(plan, 'reason', '') or '-'}",
        "",
        "Extracted Entities:",
    ]
    entities = entry.entities or {}
    if entities:
        for key in sorted(entities):
            value = entities[key]
            if value in (None, [], {}, ""):
                continue
            out.append(f"  {key} = {_shrink(value)}")
    else:
        out.append("  (none extracted)")

    # ---- keyword signals + full intent scoring, re-derived ----
    out += ["", "Extracted Keywords:"]
    try:
        from app.llm import rule_inspector

        signal_list = rule_inspector.signals(entry.query)
        for signal in signal_list:
            out.append("  " + signal.line())

        decision_view = rule_inspector.planner_decision(entry.query, entities)
        out += ["", "Intent Scoring (every scorer, not just the winner):"]
        if decision_view.error:
            out.append(f"  unavailable: {decision_view.error}")
        else:
            for cand in decision_view.candidates:
                out.append(
                    f"  scored  {cand['intent']:<20} {cand['score']:<7} "
                    f"evidence={cand['evidence']}"
                )
            for name in decision_view.declined:
                out.append(f"  declined {name}")
            if decision_view.winner and action:
                out.append(
                    f"  reconstructed winner '{decision_view.winner['intent']}' "
                    f"-> plan.action '{action}'"
                )
        out += ["", "Why this intent:"]
        out += ["  " + line for line in rule_inspector.why_lines(decision_view, signal_list)]
    except Exception:
        log.debug("rule inspection failed", exc_info=True)
        out.append("  (keyword/intent reconstruction unavailable)")

    # ---- identity ----
    out += ["", "Resolved Advisor:"]
    advisor = entry.advisor
    if advisor and (advisor.get("wid") is not None or advisor.get("name")):
        out.append(f"  name={advisor.get('name')!r}  wid={advisor.get('wid')}  "
                   f"status={advisor.get('status')}")
        for cand in advisor.get("candidates") or []:
            out.append(f"    candidate: {cand}")
    elif advisor:
        out.append(f"  UNRESOLVED (status={advisor.get('status')})")
        for cand in advisor.get("candidates") or []:
            out.append(f"    candidate: {cand}")
    else:
        out.append("  (no person resolution attempted for this query)")

    # ---- IR ----
    out += ["", "Generated QueryIR:"]
    out.append(f"  {entry.ir_json}" if entry.ir_json else
               "  none — the rule-based path answers from the QueryPlan directly, "
               "no QueryIR is built")

    # ---- planner functions ----
    out += ["", "Selected Planner Function(s):"]
    if action:
        out.append(f"  query_planner.build_query_plan -> action={action!r}"
                   f"  level={getattr(plan, 'level', None)!r}"
                   f"  entity={getattr(plan, 'entity_value', None)!r}"
                   f"  metric={getattr(plan, 'metric', None)!r}")
    else:
        out.append("  (no QueryPlan recorded — this query did not reach the planner)")

    # ---- SQL ----
    sql = entry.sql
    out += ["", f"Generated SQL/API Calls ({len(sql)}):"]
    for i, event in enumerate(sql, start=1):
        out.append(f"  [{i}] rows={event.row_count} {event.duration_ms}ms")
        out.append(f"      {event.statement}")
        if event.params is not None:
            out.append(f"      params={_shrink(event.params)}")
    if not sql:
        out.append("  (none — this query ran no SQL, or tracing was inactive)")

    # ---- data + formatter + response ----
    out += [
        "",
        f"Retrieved Data Summary: {_row_summary(response.get('data'))}",
        "",
        f"Formatter Selected: {entry.formatter or '(not recorded)'}",
        "",
        "Final Response:",
        (response.get("reply") if isinstance(response.get("reply"), str) else repr(response.get("reply"))),
        "",
        f"Response Type: {response.get('type', '-')}",
        "",
        BANNER,
    ]
    return out


def _shrink(value, limit: int = 300) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _format(entry: _Entry) -> str:
    """Builds the readable block. Layout is fixed on purpose — these get
    skimmed twenty at a time, and a stable shape is what makes a diff
    between two runs of the same question obvious."""
    response = entry.response or {}
    reply = response.get("reply")
    reply = reply if isinstance(reply, str) else repr(reply)
    if len(reply) > MAX_REPLY_CHARS:
        reply = reply[:MAX_REPLY_CHARS] + f"\n… [truncated, {len(reply)} chars total]"

    elapsed_ms = round((datetime.now().timestamp() - entry.started) * 1000, 1)

    lines = [
        "",
        BANNER,
        "CHAT AUDIT",
        BANNER,
        f"User Query:  {entry.query}",
        f"Timestamp:   {entry.timestamp.isoformat(timespec='seconds')}",
        f"Session:     {entry.session_id or '-'}",
        "",
    ]

    if entry.prompts:
        lines.append(f"LLM Prompts ({len(entry.prompts)} sent before inference):")
        for i, record in enumerate(entry.prompts, start=1):
            lines += [
                SEPARATOR,
                f"[{i}/{len(entry.prompts)}] purpose={record['purpose']} "
                f"model={record['model'] or '-'} "
                f"at={record['at'].isoformat(timespec='seconds')} "
                f"chars={len(record['prompt'] or '')}",
                SEPARATOR,
            ]
            lines += _format_prompt_breakdown(record)
            lines += [f"LLM Raw Output: {record['raw_response']!r}", ""]
    else:
        lines += ["LLM Prompts: none (answered without an LLM call)", ""]

    lines += [
        "Final Response:",
        reply,
        "",
        f"Response Type: {response.get('type', '-')}",
        f"Result Data:   {_row_summary(response.get('data'))}",
        f"Duration:      {elapsed_ms} ms",
        "",
        BANNER,
    ]

    # The rule-based trace follows the chat block for the queries that
    # took that path, so one query is still one contiguous record.
    if entry.rule_path:
        lines += _format_rule_audit(entry)

    return "\n".join(lines)


def _emit(entry: _Entry) -> None:
    block = _format(entry)
    writer = _writer()
    if writer is not None:
        writer.info(block)
    # Console echo is separate from the file so a terminal session can
    # watch queries live; the file is the durable record either way.
    if settings.chat_audit_console:
        print(block, flush=True)
