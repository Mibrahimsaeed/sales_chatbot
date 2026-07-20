"""
Resolution order, cheapest and most deterministic first:

1. Shortcuts — greeting / help / attendance_check. Fixed patterns, zero
   cost, handled by intent_detector.py. Unchanged from the previous
   design.
2. Entity grounding — advisor/team/company gazetteer match (ALL matches,
   not just the first — see entity_extractor.py), thresholds, period,
   limit.
3. Rule-based query planner (query_planner.py) — still handles "lookup"
   (a specific advisor by name), "summary" (team/company overview), and
   "attendance_filter" directly, exactly as before. These action types
   were never the compound-query problem area the redesign targeted, so
   they're intentionally left on the simpler, already-working path.
4. Metric-shaped queries ("leaderboard" action, or anything the rule-based
   planner couldn't resolve) go through semantic_parser.parse(), which:
     - takes the rule-based fast path (skip the LLM) when the query
       doesn't look compound,
     - widens the metric match via fuzzy synonym comparison next,
     - and only then calls the LLM Semantic Parser to produce a full
       QueryIR — capable of expressing multi-filter, multi-subject,
       thresholded, boolean-combined queries the old flat QueryPlan
       could not represent at all (Root Cause #1 in the review).
   The LLM call is genuinely last resort, and optional: if the provider
   is unavailable, rate-limited, or misconfigured, this degrades to a
   schema-grounded clarifying question — never a dead end, never a
   blocking retry loop.
"""

from dataclasses import dataclass

from sqlalchemy.orm import Session

import re

from app.llm import conversation_memory, semantic_parser
from app.llm.conversation_memory import MAX_CLARIFY_ATTEMPTS, PendingClarification
from app.llm.entity_extractor import extract_entities
from app.llm.fallback_reasoning import fuzzy_resolve_metric
from app.llm.intent_detector import classify_intent as classify_shortcut
from app.llm.ir_patcher import try_patch
from app.llm.ir_validator import build_targeted_clarification, validate_ir
from app.llm.preprocessing import normalize
from app.llm.query_ir import MetricRef, QueryIR, Subject
from app.llm.query_planner import build_query_plan, QueryPlan
from app.llm.metric_ontology import describe_available_metrics
from app.core.logger import get_logger

log = get_logger("llm.nlu_pipeline")

SHORTCUT_INTENTS = ("greeting", "thanks", "help", "attendance_check")
_RULE_BASED_ACTIONS = ("lookup", "summary", "attendance_filter")


@dataclass
class Resolution:
    kind: str                          # "shortcut" | "plan" | "ir" | "clarify"
    shortcut_intent: str | None = None
    plan: QueryPlan | None = None
    ir: QueryIR | None = None
    entities: dict | None = None
    used_llm_fallback: bool = False
    clarify_message: str | None = None


def _fill_pending_slot(
    pending: PendingClarification, text: str, entities: dict
) -> QueryIR | None:
    """Merge a short answer ("revenue", "Blue Area") into the pending
    partial IR's asked-about slots. Returns the filled copy if anything
    was actually filled, else None (the message didn't answer the
    question)."""
    ir = pending.partial_ir.model_copy(deep=True)
    filled = False

    if any(m == "metric" or m.startswith("metric") for m in pending.missing):
        metric = fuzzy_resolve_metric(text)
        if metric:
            ir.metric = MetricRef(key=metric, confidence=1.0)
            ir.sort.metric = metric
            filled = True

    if any(m.startswith("subject") for m in pending.missing):
        existing = {s.value for s in ir.subjects}
        for team in entities.get("teams", []):
            if team not in existing:
                ir.subjects.append(Subject(type="team", value=team, match_confidence=1.0))
                filled = True
        for company in entities.get("companies", []):
            if company not in existing:
                ir.subjects.append(Subject(type="company", value=company, match_confidence=1.0))
                filled = True

    if not filled:
        return None

    # the gap that made the parser punt to "clarify" is now filled —
    # promote to an executable intent so revalidation can pass
    if ir.intent == "clarify":
        ir.intent = "comparison" if len(ir.subjects) >= 2 else "leaderboard"
    return ir


_GIVE_UP_MESSAGE = (
    "Let's start over — try a full question like 'top 5 advisors by revenue' "
    "or 'compare Blue Area with Downtown on achievement %'."
)


def _handle_pending(
    pending: PendingClarification,
    cleaned: str,
    entities: dict,
    plan: QueryPlan,
    db: Session,
    session_id: str | None,
) -> Resolution | None:
    """Multi-turn clarification (P6): try to read this message as the
    answer to the question we asked last turn. Returns a Resolution to
    serve, or None to fall through to normal processing (the message is
    its own new query, or we've given up on this clarification)."""
    is_short = len(re.findall(r"\S+", cleaned)) <= 4
    filled = _fill_pending_slot(pending, cleaned, entities) if is_short else None

    if filled is not None:
        result = validate_ir(filled, db)
        if result.is_valid:
            conversation_memory.set(session_id, result.ir)  # also closes the pending
            return Resolution(kind="ir", ir=result.ir, entities=entities)
        if pending.attempts >= MAX_CLARIFY_ATTEMPTS:
            conversation_memory.clear_pending(session_id)
            return Resolution(kind="clarify", entities=entities, clarify_message=_GIVE_UP_MESSAGE)
        conversation_memory.set_pending(session_id, result.ir, result.missing)
        return Resolution(
            kind="clarify", ir=result.ir, entities=entities,
            clarify_message=build_targeted_clarification(result.missing),
        )

    # nothing filled: a self-standing query means the user moved on —
    # drop the pending and answer the new question instead
    if plan.action != "unresolved" or semantic_parser.looks_compound(cleaned, entities):
        conversation_memory.clear_pending(session_id)
        return None

    # short but unhelpful answer — re-ask once, then give up gracefully
    if pending.attempts >= MAX_CLARIFY_ATTEMPTS:
        conversation_memory.clear_pending(session_id)
        return Resolution(kind="clarify", entities=entities, clarify_message=_GIVE_UP_MESSAGE)
    conversation_memory.set_pending(session_id, pending.partial_ir, pending.missing)
    return Resolution(
        kind="clarify", ir=pending.partial_ir, entities=entities,
        clarify_message=build_targeted_clarification(pending.missing),
    )


def resolve(text: str, db: Session, session_id: str | None = None) -> Resolution:
    cleaned = normalize(text)

    shortcut = classify_shortcut(cleaned, {})
    if shortcut.intent in SHORTCUT_INTENTS:
        return Resolution(kind="shortcut", shortcut_intent=shortcut.intent, entities={})

    entities = extract_entities(cleaned, db)
    plan = build_query_plan(cleaned, entities)

    # an in-flight clarification takes precedence: a bare "revenue" or
    # "Blue Area" only means something as the answer to last turn's question
    pending = conversation_memory.get_pending(session_id)
    if pending is not None:
        served = _handle_pending(pending, cleaned, entities, plan, db, session_id)
        if served is not None:
            return served

    # short follow-up modifiers ("only Graana", "top 5", "sort ascending")
    # patch the previous turn's IR deterministically — no LLM round trip.
    # try_patch declines anything that stands alone as its own query.
    prior_ir = conversation_memory.get(session_id)
    if prior_ir is not None:
        patched = try_patch(prior_ir, cleaned, entities, plan.action)
        if patched is not None:
            result = validate_ir(patched, db)
            if result.is_valid:
                conversation_memory.set(session_id, result.ir)
                return Resolution(kind="ir", ir=result.ir, entities=entities)
            # an invalid patch falls through to the normal parse path

    # lookup/summary/attendance_filter stay on the simple rule-based path,
    # but only when the query doesn't look compound — "summary for Graana
    # AND Downtown" belongs to the semantic parser, not a single-entity plan.
    if plan.action in _RULE_BASED_ACTIONS and not semantic_parser.looks_compound(cleaned, entities):
        return Resolution(kind="plan", plan=plan, entities=entities)

    outcome = semantic_parser.parse(cleaned, entities, db, session_id)

    if outcome.ir and not outcome.missing:
        return Resolution(
            kind="ir",
            ir=outcome.ir,
            entities=entities,
            used_llm_fallback=outcome.used_llm,
        )

    if outcome.ir and outcome.missing:
        # remember what we asked, so the next message can answer it
        # (slot-filling, P6) instead of starting from scratch
        conversation_memory.set_pending(session_id, outcome.ir, outcome.missing)
        return Resolution(
            kind="clarify",
            ir=outcome.ir,
            entities=entities,
            used_llm_fallback=outcome.used_llm,
            clarify_message=build_targeted_clarification(outcome.missing),
        )

    return Resolution(
        kind="clarify",
        entities=entities,
        clarify_message=(
            f"I couldn't match that to something I track. I can answer about: "
            f"{describe_available_metrics()} — or look up an advisor, team, or company by name."
        ),
    )
