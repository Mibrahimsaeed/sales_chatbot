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

from app.llm import semantic_parser
from app.llm.entity_extractor import extract_entities
from app.llm.intent_detector import classify_intent as classify_shortcut
from app.llm.ir_validator import build_targeted_clarification
from app.llm.preprocessing import normalize
from app.llm.query_ir import QueryIR
from app.llm.query_planner import build_query_plan, QueryPlan
from app.llm.metric_ontology import describe_available_metrics
from app.core.logger import get_logger

log = get_logger("llm.nlu_pipeline")

SHORTCUT_INTENTS = ("greeting", "help", "attendance_check")
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


def resolve(text: str, db: Session, session_id: str | None = None) -> Resolution:
    cleaned = normalize(text)

    shortcut = classify_shortcut(cleaned, {})
    if shortcut.intent in SHORTCUT_INTENTS:
        return Resolution(kind="shortcut", shortcut_intent=shortcut.intent, entities={})

    entities = extract_entities(cleaned, db)
    plan = build_query_plan(cleaned, entities)

    if plan.action in _RULE_BASED_ACTIONS:
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
