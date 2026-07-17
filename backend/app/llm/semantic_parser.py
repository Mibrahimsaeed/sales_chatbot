"""
Orchestrates steps 2-4 of the new pipeline (see nlu_pipeline.py's
module docstring for the full order):

  rule-based QueryPlan (cheap, exact) for simple queries
      -> LLM Semantic Parser (QueryIR) only for compound/ambiguous ones
      -> IR Validator/Grounder
      -> fail-soft degrade to the rule-based plan if the LLM is
         unavailable or returns something invalid

This keeps the property the original pipeline had ("no LLM involved for
the large majority of real queries") while fixing the actual bottleneck
the review identified: when a query genuinely needs a compound plan, the
LLM is now asked to produce a QueryIR that CAN express it, instead of
being constrained to the same flat single-metric schema the regex layer
used.
"""

from __future__ import annotations

import re

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.llm import conversation_memory
from app.llm.entity_extractor import get_known_teams, get_known_companies
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
    Multiple teams/companies, a threshold, or boolean-language keywords —
    any of these is a signal to route to the LLM semantic parser instead
    of trusting the rule-based fast path."""
    if len(entities.get("teams", [])) > 1:
        return True
    if len(entities.get("companies", [])) > 1:
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


def parse(text: str, entities: dict, db: Session, session_id: str | None) -> ParseOutcome:
    plan: QueryPlan = build_query_plan(text, entities)
    compound = looks_compound(text, entities)

    # ---- 1. rule-based fast path (skips the LLM for the common case) ----
    if plan.action == "leaderboard" and not compound:
        ir = plan_to_ir(plan, entities)
        result = validate_ir(ir, db)
        return ParseOutcome(ir=result.ir, missing=result.missing, used_llm=False)

    # ---- 2. widen via fuzzy metric match before reaching for the LLM ----
    if plan.action == "unresolved" and not compound:
        widened = fuzzy_resolve_metric(text)
        if widened:
            entities = {**entities, "metric": widened}
            widened_plan = build_query_plan(text, entities)
            if widened_plan.action == "leaderboard":
                log.info(f"Fallback reasoning resolved '{text}' via widened metric match: {widened}")
                ir = plan_to_ir(widened_plan, entities)
                result = validate_ir(ir, db)
                return ParseOutcome(ir=result.ir, missing=result.missing, used_llm=False)

    # ---- 3. LLM semantic parser — compound query, or nothing else worked ----
    prior_ir = conversation_memory.get(session_id)
    prompt = build_ir_prompt(
        text,
        get_known_teams(db),
        get_known_companies(db),
        grounded_entities=entities,
        prior_ir_json=prior_ir.model_dump_json() if prior_ir else None,
    )
    raw = call_llm_structured(prompt, QUERY_IR_JSON_SCHEMA, schema_name="query_ir")

    ir: QueryIR | None = None
    if raw:
        try:
            ir = QueryIR.model_validate(raw)
        except ValidationError as e:
            log.warning(f"LLM returned a QueryIR that failed validation: {e}")
            ir = None

    # ---- 4. fail-soft degrade ----
    if ir is None:
        if plan.action == "leaderboard":
            ir = plan_to_ir(plan, entities)
        else:
            return ParseOutcome(ir=None, missing=["intent"], used_llm=False)

    result = validate_ir(ir, db)
    if result.is_valid:
        conversation_memory.set(session_id, result.ir)
    return ParseOutcome(ir=result.ir, missing=result.missing, used_llm=True)
