"""
LLM planner — natural language to a structured QueryPlan.

Replaces ONLY the planning step. Everything downstream is untouched and
still fully deterministic: entity resolution to WIDs, SQL compilation,
execution, and formatting all run exactly as they do under the
rule-based planner. The LLM never sees the database, never produces SQL,
and never phrases an answer.

THE PIPELINE, and where each guarantee is enforced:

    text
      -> build_planner_prompt()        prompt is generated from the live
                                       ontology, so it cannot advertise a
                                       metric the compiler can't execute
      -> call_llm_structured()         provider-side JSON schema; a
                                       malformed shape never reaches us
      -> LLMQueryPlan.model_validate() structural validation (pydantic)
      -> validate_plan()               SEMANTIC validation: metric exists,
                                       entity counts fit the intent,
                                       filter fields are real
      -> to_query_plan()               adapt to the existing QueryPlan
                                       dataclass, so dispatch/compiler
                                       need no changes at all
      -> existing WID resolver, compiler, formatter

FAIL-SOFT AT EVERY STEP. Any failure — provider down, no quota, invalid
JSON, hallucinated metric, low confidence — returns None, and
nlu_pipeline falls back to the rule-based planner. That fallback is what
makes the feature flag safe to flip: the worst case is the behaviour we
have today, never an error.
"""

from __future__ import annotations

import json
import time

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core import tracing
from app.core.config import settings
from app.core.logger import get_logger
from app.llm import hierarchy
from app.llm.llm_client import call_llm_structured
from app.llm.metric_ontology import METRICS
from app.llm.planner_prompt import build_planner_prompt
from app.llm.planner_schema import LLMQueryPlan, QUERY_PLAN_JSON_SCHEMA
from app.llm.query_planner import QueryPlan

log = get_logger("llm.planner")

# Below this the plan is treated as a clarification rather than executed.
# Matches the QueryIR path's confidence gate so both planners apply the
# same standard for "sure enough to act on".
MIN_CONFIDENCE = 0.5

# Non-metric filter fields the compiler understands. Anything else in
# `filters` is a hallucination and the plan is rejected.
_VALID_FILTER_FIELDS = set(hierarchy.GROUP_LEVELS) | {"advisor", "attendance_status"}

# Intents that are meaningless without at least one named entity.
_REQUIRE_ENTITY = {
    "advisor_profile", "hierarchy", "reverse_hierarchy", "roster", "entity_summary",
}

# Planner intent -> the action the existing dispatch already handles.
# This mapping is why nothing downstream changes: the LLM planner is a
# new way to CHOOSE, not a new set of capabilities.
_INTENT_TO_ACTION = {
    "advisor_profile": "lookup",
    "hierarchy": "breakdown",
    "reverse_hierarchy": "reverse_hierarchy",
    "roster": "roster",
    "leaderboard": "leaderboard",
    "comparison": "comparison",
    "attendance_filter": "attendance_filter",
    "entity_summary": "summary",
    "clarification": "unresolved",
    "greeting": "unresolved",
    "help": "unresolved",
    "fallback": "unresolved",
}


class PlannerRejection(Exception):
    """The plan was structurally fine but semantically unusable."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def validate_plan(plan: LLMQueryPlan) -> LLMQueryPlan:
    """SEMANTIC validation, past what pydantic can express.

    Every rejection here is a case where executing the plan would produce
    a confidently wrong answer, so rejecting into a clarification is
    strictly better than proceeding."""
    if plan.confidence < MIN_CONFIDENCE:
        raise PlannerRejection(f"low_confidence:{plan.confidence:.2f}")

    if plan.metric is not None and plan.metric not in METRICS:
        # A hallucinated metric would otherwise reach the compiler and
        # either crash or silently resolve to something else.
        raise PlannerRejection(f"unknown_metric:{plan.metric}")

    if plan.sort and plan.sort.metric and plan.sort.metric not in METRICS:
        raise PlannerRejection(f"unknown_sort_metric:{plan.sort.metric}")

    for f in plan.filters:
        if f.field not in _VALID_FILTER_FIELDS and f.field not in METRICS:
            raise PlannerRejection(f"unknown_filter_field:{f.field}")

    if plan.intent in _REQUIRE_ENTITY and not plan.entities:
        raise PlannerRejection(f"missing_entities_for:{plan.intent}")

    if plan.intent == "comparison" and len(plan.entities) < 2:
        raise PlannerRejection("comparison_needs_two_entities")

    if plan.intent == "leaderboard" and not (plan.metric or (plan.sort and plan.sort.metric)):
        raise PlannerRejection("leaderboard_needs_a_metric")

    if plan.intent == "attendance_filter":
        if not any(f.field == "attendance_status" for f in plan.filters):
            raise PlannerRejection("attendance_filter_needs_a_status")

    return plan


def to_query_plan(plan: LLMQueryPlan, entities: dict) -> QueryPlan:
    """Adapt the planner's output to the existing QueryPlan dataclass.

    `entities` is the RULE-BASED extraction for the same message. The LLM
    supplies intent and entity TEXT; the grounded values, WIDs and
    ambiguity flags still come from the deterministic resolver — the LLM
    is never trusted to say which real record a name refers to. Where the
    resolver already grounded a value, that value wins over the LLM's raw
    text."""
    action = _INTENT_TO_ACTION.get(plan.intent, "unresolved")
    evidence = [f"llm_intent:{plan.intent}", f"llm_confidence:{plan.confidence:.2f}"]

    def _grounded(entity_type: str, raw: str) -> str:
        """Prefer the resolver's grounded spelling when it found this
        level, since it matched against real database values."""
        if entity_type == "advisor":
            return entities.get("advisor_name") or raw
        return entities.get(entity_type) or raw

    typed = [(e.type, _grounded(e.type, e.value)) for e in plan.entities]
    primary = typed[0] if typed else (None, None)

    # An ambiguous person must ask, regardless of which planner chose the
    # intent — the LLM has no way to know a name maps to 8 real people.
    if plan.intent in ("advisor_profile", "reverse_hierarchy") and entities.get("advisor_ambiguous"):
        resolution = entities.get("advisor_resolution")
        return QueryPlan(
            action="clarify_person",
            level="advisor",
            entity_value=entities.get("advisor_name"),
            person_candidates=list(resolution.candidates) if resolution else [],
            intent_evidence=evidence + ["ambiguous_person"],
        )

    built = QueryPlan(
        action=action,
        level=primary[0],
        entity_value=primary[1],
        metric=plan.metric or (plan.sort.metric if plan.sort else None),
        limit=plan.limit or 10,
        ascending=bool(plan.sort and plan.sort.direction == "asc"),
        intent_score=plan.confidence,
        intent_evidence=evidence,
    )

    if plan.intent == "advisor_profile":
        built.entity_wid = entities.get("advisor_wid")
    elif plan.intent == "reverse_hierarchy":
        built.entity_wid = entities.get("advisor_wid")
        built.entity_value = entities.get("advisor_name") or primary[1]
        # which manager level was asked for is a hierarchy question the
        # rule-based detector already answers well from the wording
        built.level = _reverse_level(plan)
    elif plan.intent == "comparison":
        built.comparison_targets = typed
    elif plan.intent == "attendance_filter":
        status = next((f.value for f in plan.filters if f.field == "attendance_status"), None)
        built.reason = str(status) if status is not None else ""
        built.level = "advisor"
        built.entity_value = entities.get("team")
    elif plan.intent == "leaderboard":
        built.level = _leaderboard_level(plan, entities)
    elif plan.intent in ("clarification", "greeting", "help", "fallback"):
        built.reason = plan.clarification or plan.intent

    return built


def _reverse_level(plan: LLMQueryPlan) -> str:
    from app.llm import intent_catalog as cat

    # DERIVED. This was ("unit_head", "zonal_head", "business_center") —
    # stale on both ends: it named the retired `business_center` and
    # omitted `bcm`, so "who is X's BCM" fell through to the default
    # level and answered about the wrong manager. A reverse lookup asks
    # about a level ABOVE the advisor, which is exactly the chain minus
    # its leaf and its root.
    from app.llm import hierarchy

    manager_levels = [
        level for level in hierarchy.CHAIN
        if level != "advisor" and hierarchy.parent_of(level) is not None
    ]
    for entity in plan.entities:
        if hierarchy.canonical_level(entity.type) in manager_levels:
            return hierarchy.canonical_level(entity.type)
    return cat.DEFAULT_REVERSE_LEVEL


def _leaderboard_level(plan: LLMQueryPlan, entities: dict) -> str:
    """The level a ranking groups BY.

    Phase 2: delegates to subject_level.decide(), the single owner. This
    used to return `metric.primary_level` outright on the reasoning that
    "an entity in the plan is a scope filter, not the grouping level".
    That is true for a RANKING ("top advisors in Graana") and false for
    everything else — "Downtown's pipeline value" names Downtown as the
    subject, and answering it with a list of advisors answered a
    different question. decide() draws exactly that distinction from the
    ranking signal, so both planners now reach the same level for the
    same query instead of holding two copies of the rule.
    """
    from app.llm import subject_level

    metric_key = plan.metric or (plan.sort.metric if plan.sort else None)
    metric = METRICS.get(metric_key) if metric_key else None
    entity_level, entity_value = subject_level.entity_level_from(entities)
    return subject_level.decide(
        level_word=None,   # the LLM plan carries no level word of its own
        entity_level=entity_level,
        entity_value=entity_value,
        metric_default=metric.primary_level if metric else "advisor",
        has_ranking=bool(plan.limit) or bool(plan.sort),
    ).level


def plan_query(
    text: str, entities: dict, db: Session, session_id: str | None = None
) -> QueryPlan | None:
    """Plan `text` with the LLM. None means "could not plan" — the caller
    falls back to the rule-based planner, which is why every failure path
    here is silent-and-safe rather than an exception."""
    started = time.monotonic()

    from app.llm.entity_extractor import get_known_companies, get_known_teams

    try:
        prompt = build_planner_prompt(
            text,
            known_teams=get_known_teams(db),
            known_companies=get_known_companies(db),
        )
    except Exception:
        log.exception("Planner prompt build failed")
        return None

    raw = call_llm_structured(prompt, QUERY_PLAN_JSON_SCHEMA, schema_name="query_plan")
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)

    if raw is None:
        # provider unavailable — already logged by llm_client
        tracing.record_planner(prompt=prompt, raw=None, plan=None,
                              rejected="provider_unavailable", elapsed_ms=elapsed_ms)
        return None

    try:
        parsed = LLMQueryPlan.model_validate(raw)
    except ValidationError as e:
        log.warning("LLM planner returned a structurally invalid plan: %s", e)
        tracing.record_planner(prompt=prompt, raw=raw, plan=None,
                              rejected="schema_invalid", elapsed_ms=elapsed_ms)
        return None

    try:
        validated = validate_plan(parsed)
    except PlannerRejection as rejection:
        # A rejected plan becomes a CLARIFICATION, not a fallback to the
        # rule-based planner: the model understood the question well
        # enough to answer structurally but produced something
        # unexecutable, and quietly re-planning with weaker machinery
        # would hide that.
        log.info("LLM planner output rejected (%s) for %r", rejection.reason, text)
        tracing.record_planner(prompt=prompt, raw=raw, plan=None,
                              rejected=rejection.reason, elapsed_ms=elapsed_ms)
        return QueryPlan(
            action="unresolved",
            reason=parsed.clarification or f"planner_rejected:{rejection.reason}",
            intent_evidence=[f"llm_rejected:{rejection.reason}"],
        )

    built = to_query_plan(validated, entities)
    tracing.record_planner(
        prompt=prompt, raw=raw, plan=json.loads(validated.model_dump_json()),
        rejected=None, elapsed_ms=elapsed_ms,
    )
    log.debug("LLM planner: %r -> %s (%.0fms)", text, built.action, elapsed_ms)
    return built


def is_enabled() -> bool:
    return bool(getattr(settings, "use_llm_planner", False))
