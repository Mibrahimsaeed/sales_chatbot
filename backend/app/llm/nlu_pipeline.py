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

from app.llm import conversation_memory, multi_intent, semantic_parser
from app.llm.conversation_memory import MAX_CLARIFY_ATTEMPTS, PendingClarification
from app.llm.entity_extractor import extract_entities
from app.llm.fallback_reasoning import fuzzy_resolve_metric
from app.llm.intent_detector import classify_intent as classify_shortcut
from app.llm.ir_patcher import try_patch
from app.llm.ir_validator import (
    build_targeted_clarification,
    clarification_options,
    pick_clarification_slot,
    validate_ir,
)
from app.llm.preprocessing import normalize
from app.llm.query_ir import MetricRef, QueryIR, Subject
from app.llm.query_planner import build_query_plan, QueryPlan
from app.llm.metric_ontology import describe_available_metrics
from app.core.logger import get_logger

log = get_logger("llm.nlu_pipeline")

SHORTCUT_INTENTS = ("greeting", "thanks", "help", "attendance_check")
_RULE_BASED_ACTIONS = ("lookup", "summary", "attendance_filter")

# Part 8: typed "show more" — the alternative to clicking the button
# (POST /chat/more, see app/api/chat.py). Only recognized when there's an
# active pagination cursor for this session (conversation_memory); with
# no cursor, "more" et al. fall through to the normal pipeline unchanged
# (verified against intent_detector's rules — none of them match these
# phrases, so this can't misfire and steal a real query).
_SHOW_MORE_RE = re.compile(r"^(show more|more|next|next page|load more)$", re.I)


@dataclass
class Resolution:
    kind: str                          # "shortcut" | "plan" | "ir" | "clarify"
    shortcut_intent: str | None = None
    plan: QueryPlan | None = None
    ir: QueryIR | None = None
    entities: dict | None = None
    used_llm_fallback: bool = False
    clarify_message: str | None = None
    clarify_options: list[str] | None = None
    sections: list[tuple[str, "Resolution"]] | None = None  # kind == "multi" only


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

    # Part 10: overall_confidence/intent_confidence reflected doubt about
    # WHAT the user wanted before they answered directly — a slot the user
    # just answered in response to our own question is as trustworthy as a
    # rule-based match, so a stale low holistic score from the original
    # ambiguous parse must not keep tripping the "low confidence" gate in
    # ir_validator.classify_confidence() after the ambiguity is resolved.
    ir.overall_confidence = max(ir.overall_confidence, 0.9)
    ir.intent_confidence = max(ir.intent_confidence, 0.9)
    return ir


def _clarify(missing: list[str], db: Session) -> tuple[str, list[str]]:
    """Part 8: the targeted question plus suggested options for it (real
    metric labels / real team or company names), computed from the same
    highest-priority slot build_targeted_clarification() already picks."""
    slot = pick_clarification_slot(missing)
    return build_targeted_clarification(missing), clarification_options(slot, db)


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
        if pending.attempts >= MAX_CLARIFY_ATTEMPTS or result.confidence_level == "low":
            conversation_memory.clear_pending(session_id)
            return Resolution(kind="clarify", entities=entities, clarify_message=_GIVE_UP_MESSAGE)
        conversation_memory.set_pending(session_id, result.ir, result.missing)
        message, options = _clarify(result.missing, db)
        return Resolution(
            kind="clarify", ir=result.ir, entities=entities,
            clarify_message=message, clarify_options=options,
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
    message, options = _clarify(pending.missing, db)
    return Resolution(
        kind="clarify", ir=pending.partial_ir, entities=entities,
        clarify_message=message, clarify_options=options,
    )


def resolve(text: str, db: Session, session_id: str | None = None, _depth: int = 0) -> Resolution:
    cleaned = normalize(text)

    if _SHOW_MORE_RE.match(cleaned.strip()) and conversation_memory.get_pagination(session_id) is not None:
        return Resolution(kind="paginate", entities={})

    # Part 8 (light multi-intent): checked BEFORE shortcut classification
    # on purpose — classify_intent scores the whole string, so a compound
    # message ("top advisors by revenue; who was late today") can contain
    # a shortcut-triggering word (here "late") that would otherwise hijack
    # the ENTIRE message into a single shortcut instead of ever reaching
    # the splitter. A genuinely compound utterance is split into
    # independent sub-queries, each resolved through this same pipeline
    # (including its own shortcut check), and stitched into labeled
    # sections by chat_service. Depth-capped at 1 — a split segment is
    # never re-split, and only the LAST segment's IR ends up persisted to
    # conversation_memory (each resolve() call below writes it in turn) —
    # a known, documented limitation of the light version.
    if _depth == 0:
        segments = multi_intent.split_subqueries(cleaned)
        if segments is not None:
            sections = [(seg, resolve(seg, db, session_id, _depth=1)) for seg in segments]
            return Resolution(kind="multi", entities={}, sections=sections)

    shortcut = classify_shortcut(cleaned, {})
    if shortcut.intent in SHORTCUT_INTENTS:
        return Resolution(kind="shortcut", shortcut_intent=shortcut.intent, entities={})

    entities = extract_entities(cleaned, db)

    # Part 8: a genuinely unsupported time window (last month, yesterday,
    # this week, past N days, a custom date range) must never silently
    # fall back to MTD — say so plainly instead of guessing wrong.
    if entities.get("period_unsupported"):
        return Resolution(
            kind="clarify",
            entities=entities,
            clarify_message=entities["period_unsupported"] + ". Try 'this month', 'year to date', or 'last 3 months' instead.",
        )

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
        # Part 10: "low" confidence means the parse itself is too shaky to
        # trust `missing` enough to ask about one specific slot — reject
        # outright and ask the user to rephrase, instead of setting a
        # pending clarification that's likely to keep missing the mark.
        # Never executed (this never returns kind="ir").
        if outcome.ir.confidence_level == "low":
            return Resolution(
                kind="clarify",
                ir=outcome.ir,
                entities=entities,
                used_llm_fallback=outcome.used_llm,
                clarify_message=_GIVE_UP_MESSAGE,
            )
        # "medium": remember what we asked, so the next message can answer
        # it (slot-filling, P6) instead of starting from scratch
        conversation_memory.set_pending(session_id, outcome.ir, outcome.missing)
        message, options = _clarify(outcome.missing, db)
        return Resolution(
            kind="clarify",
            ir=outcome.ir,
            entities=entities,
            used_llm_fallback=outcome.used_llm,
            clarify_message=message,
            clarify_options=options,
        )

    return Resolution(
        kind="clarify",
        entities=entities,
        clarify_message=(
            f"I couldn't match that to something I track. I can answer about: "
            f"{describe_available_metrics()} — or look up an advisor, team, or company by name."
        ),
    )
