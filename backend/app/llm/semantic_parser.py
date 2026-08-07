"""
Orchestrates the semantic-parse step of the pipeline (see nlu_pipeline.py's
module docstring for the full order). Two modes, switched by
settings.nlu_mode (NLU_MODE env var):

  "llm_first" (default): the LLM parses EVERY analytical query reaching
      this module into a QueryIR — the rule-based plan is computed only
      as the fail-soft degrade target for LLM failure/invalid output.
      This is the P1 inversion of the NLU rework: regex/rule parsing
      systematically under-expressed real business queries, so rules are
      now the safety net, not the front door.

  "rules_first": the pre-inversion behavior, preserved verbatim as the
      rollback path — rule-based fast path for simple leaderboards, fuzzy
      metric widening, LLM only for compound-looking queries.

Both modes end at the same place: IR Validator/Grounder, and never a
dead end — LLM unavailability degrades to the rule-based plan.
"""

from __future__ import annotations

import re

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.llm import conversation_memory, hierarchy, semantic_retrieval
from app.llm.entity_extractor import (
    get_known_teams, get_known_companies,
    get_known_unit_heads, get_known_zonal_heads, get_known_bcms,
)
from app.llm.fallback_reasoning import fuzzy_resolve_metric
from app.llm.ir_validator import validate_ir
from app.llm.llm_client import call_llm_structured, QUERY_IR_JSON_SCHEMA
from app.llm.prompt_builder import build_ir_prompt
from app.llm.query_ir import QueryIR, plan_to_ir
from app.llm.query_planner import QueryPlan, build_query_plan
from app.core.logger import get_logger

log = get_logger("llm.semantic_parser")

_COMPOUND_HINTS = re.compile(
    r"\b(but|compare|comparison|vs\.?|versus|still|and also|except|excluding)\b", re.I
)


def looks_compound(text: str, entities: dict) -> bool:
    """Cheap heuristic gate: does this query look like it needs more than
    query_planner.py's single-metric/single-filter QueryPlan can express?
    Multiple matches at ANY hierarchy level (team/company/unit_head/
    zonal_head/business_center), a threshold, or boolean-language keywords
    — any of these is a signal to route to the LLM semantic parser instead
    of trusting the rule-based fast path."""
    if any(len(entities.get(entity_key, [])) > 1 for entity_key in hierarchy.LEVEL_ENTITY_KEYS.values()):
        return True
    if entities.get("thresholds"):
        return True
    if _COMPOUND_HINTS.search(text):
        return True
    return False


class ParseOutcome:
    def __init__(self, ir: QueryIR | None, missing: list[str], used_llm: bool):
        self.ir = ir
        self.missing = missing
        self.used_llm = used_llm


def _call_llm_for_ir(text: str, entities: dict, db: Session, session_id: str | None) -> QueryIR | None:
    """One LLM round trip -> QueryIR, or None on any failure (no key,
    provider error, schema-invalid output)."""
    prior_ir = conversation_memory.get(session_id)
    prompt = build_ir_prompt(
        text,
        get_known_teams(db),
        get_known_companies(db),
        grounded_entities=entities,
        prior_ir_json=prior_ir.model_dump_json() if prior_ir else None,
        recent_turns=conversation_memory.recent_turns(session_id),
        known_unit_heads=get_known_unit_heads(db),
        known_zonal_heads=get_known_zonal_heads(db),
        known_bcms=get_known_bcms(db),
    )
    raw = call_llm_structured(prompt, QUERY_IR_JSON_SCHEMA, schema_name="query_ir")
    if not raw:
        return None
    try:
        return QueryIR.model_validate(raw)
    except ValidationError as e:
        log.warning(f"LLM returned a QueryIR that failed validation: {e}")
        return None


# Plan actions the rule planner can turn into a complete QueryIR.
# Phase 5B added "comparison": it used to execute on the plan path
# through comparison_service, and once that path was removed the fast
# path below had to recognise it or every comparison degraded to
# missing=["intent"]. Named once so the four checks that used to spell
# out "leaderboard" cannot drift apart.
# Phase 7 added "group_metric": a first-class classification for "one
# group, one measure" that compiles to the same scoped QueryIR a
# leaderboard would. Listed here so the fast path recognises it — without
# this every group-metric query degraded to missing=["intent"].
_IR_ACTIONS = ("leaderboard", "comparison", "group_metric")


def _rule_based_ir(text: str, entities: dict, plan: QueryPlan) -> QueryIR | None:
    """The deterministic degrade target: the rule plan if it resolved to a
    leaderboard, else a widening attempt — fuzzy synonym match first
    (typos, "revnue"), then embedding-based semantic retrieval (Part 8)
    for genuine paraphrases with no lexical overlap ("who's crushing it"
    for achievement_pct). None if nothing produces something answerable.
    """
    if plan.action in _IR_ACTIONS:
        return plan_to_ir(plan, entities)
    if plan.action == "unresolved":
        widened = fuzzy_resolve_metric(text) or semantic_retrieval.retrieve_metric(text)
        if widened:
            widened_entities = {**entities, "metric": widened}
            widened_plan = build_query_plan(text, widened_entities)
            if widened_plan.action in _IR_ACTIONS:
                log.info(f"Fallback reasoning resolved '{text}' via widened metric match: {widened}")
                return plan_to_ir(widened_plan, widened_entities)
    return None


def _finish(ir: QueryIR, db: Session, session_id: str | None, used_llm: bool) -> ParseOutcome:
    """Validate and return. Deliberately does NOT store the IR.

    Phase 4: this used to write conversation_memory itself, making four
    writers of last_ir across two modules. Worse, it wrote the PRE-MERGE
    IR — the one built from this turn's words alone — so the turn that
    actually answered and the turn the next message inherited from could
    differ. nlu_pipeline.resolve() owns conversation state now and stores
    the merged IR once, after conversation_context has run.
    """
    result = validate_ir(ir, db)
    result.ir.nlu_mode = settings.nlu_mode
    return ParseOutcome(ir=result.ir, missing=result.missing, used_llm=used_llm)


def parse(text: str, entities: dict, db: Session, session_id: str | None,
          plan: QueryPlan | None = None) -> ParseOutcome:
    """Phase 4: `plan` is supplied by the caller.

    nlu_pipeline.resolve() has already planned this message — it routes on
    plan.action — and this function then planned it AGAIN from the same
    text and the same entities, so build_query_plan ran twice per request
    for an identical result. Planning is deterministic and pure, so the
    duplicate was invisible in behaviour and paid for twice in work.

    The parameter is optional so the module stays independently callable
    (its own tests do), but the pipeline always passes the plan it
    already made. One message, one planner run.
    """
    if plan is None:
        plan = build_query_plan(text, entities)

    # ---- llm_first (P1 inversion): LLM parses everything reaching here ----
    if settings.nlu_mode == "llm_first":
        ir = _call_llm_for_ir(text, entities, db, session_id)
        if ir is not None:
            return _finish(ir, db, session_id, used_llm=True)
        degraded = _rule_based_ir(text, entities, plan)
        if degraded is not None:
            return _finish(degraded, db, session_id, used_llm=False)
        return ParseOutcome(ir=None, missing=["intent"], used_llm=False)

    # ---- rules_first: pre-inversion behavior, kept as the rollback path ----
    compound = looks_compound(text, entities)

    # 1. rule-based fast path (skips the LLM for the common case).
    #
    # A comparison is exempt from the compound gate, as it was on the
    # plan path (nlu_pipeline._COMPOUND_EXEMPT_ACTIONS). looks_compound()
    # fires on the word "compare" itself AND on two entities at one
    # level, so EVERY comparison looks compound — the gate exists because
    # the rule planner once could not express multi-entity queries, which
    # stopped being true when comparison_targets() started grounding both
    # sides. Without the exemption every comparison would take an LLM
    # round trip to arrive at the IR the rule planner already had.
    if plan.action in _IR_ACTIONS and (plan.action == "comparison" or not compound):
        return _finish(plan_to_ir(plan, entities), db, session_id, used_llm=False)

    # 2. widen via fuzzy metric match before reaching for the LLM
    if plan.action == "unresolved" and not compound:
        degraded = _rule_based_ir(text, entities, plan)
        if degraded is not None:
            return _finish(degraded, db, session_id, used_llm=False)

    # 3. LLM semantic parser — compound query, or nothing else worked
    ir = _call_llm_for_ir(text, entities, db, session_id)

    # 4. fail-soft degrade
    if ir is None:
        if plan.action not in _IR_ACTIONS:
            return ParseOutcome(ir=None, missing=["intent"], used_llm=False)
        ir = plan_to_ir(plan, entities)

    return _finish(ir, db, session_id, used_llm=True)
