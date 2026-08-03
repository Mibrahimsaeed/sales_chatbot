"""
The contract between the LLM planner and everything downstream.

DESIGN RULE, and the reason this file is small and strict: the LLM's ONLY
job is turning natural language into this structure. It never writes SQL,
never resolves an identity, never phrases an answer. Everything past this
boundary — entity resolution to WIDs, SQL compilation, execution,
formatting — stays deterministic and auditable exactly as it is today.

So this schema is deliberately a DESCRIPTION OF INTENT, not a query:

  - `entities` carry TYPE + TEXT only. No wids, no ids. Resolution stays
    with advisor_resolver/entity_extractor, which know about the 238
    duplicate-name groups the LLM cannot see.
  - `metric` is a key that must exist in metric_ontology. A hallucinated
    metric is rejected here rather than reaching the compiler.
  - there is no field in which SQL could be expressed.

Anything the model returns that doesn't satisfy this is rejected by
validate_plan() and becomes a clarification — never a guess, and never a
partially-applied plan.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from app.llm import hierarchy
from app.llm.periods import PERIODS

# Every intent the planner may emit. Mirrors the actions the existing
# dispatch already handles — the LLM planner is a new way to CHOOSE an
# intent, not a new set of capabilities.
PlannedIntent = Literal[
    "advisor_profile",
    "hierarchy",
    "reverse_hierarchy",
    "roster",
    "leaderboard",
    "comparison",
    "attendance_filter",
    "entity_summary",
    "clarification",
    "greeting",
    "help",
    "fallback",
]

# DERIVED from the hierarchy registry, plus the legacy aliases so a
# stored plan still validates.
#
# F2 on the PLANNER path. This was a hand-written list that had already
# drifted: it omitted `bcm` — a level of the verified chain — and
# `office`/`region`, while still naming `business_center`, which Phase 3
# retired in favour of `office`. The JSON enum below carried the same
# list, and it is sent with strict decoding, so the planner LLM could not
# emit a BCM or a region at all and was offered a deprecated name
# instead. Exactly the defect Phase 5.2 fixes on the IR path, one module
# over.
ENTITY_TYPES: tuple[str, ...] = tuple(
    sorted(set(hierarchy.HIERARCHY_LEVELS) | set(hierarchy.LEVEL_ALIASES))
)
EntityType = Literal[ENTITY_TYPES]

Operator = Literal["=", "!=", ">", ">=", "<", "<=", "in"]
# DERIVED from temporal_parser.PERIODS — see llm_client for what a
# hardcoded copy costs. Literal[tuple] is the runtime spelling of
# Literal[a, b, c].
Period = Literal[PERIODS]


class PlannedEntity(BaseModel):
    """An entity as the USER NAMED IT. `value` is raw text; the existing
    WID resolver grounds it afterwards. The planner deliberately cannot
    supply an id — it has no way to know which of several people sharing
    a name was meant, and inventing one is the exact failure mode the
    identity refactor removed."""
    type: EntityType
    value: str

    @field_validator("value")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("entity value must be non-empty")
        return v.strip()


class PlannedFilter(BaseModel):
    field: str
    operator: Operator = "="
    value: Optional[Union[str, float, int, list]] = None


class PlannedSort(BaseModel):
    metric: Optional[str] = None
    direction: Literal["asc", "desc"] = "desc"


class LLMQueryPlan(BaseModel):
    """What the planner returns. Validated structurally by pydantic here,
    then SEMANTICALLY by planner_validator (metric exists, entity counts
    match the intent, filter fields are real)."""
    intent: PlannedIntent
    entities: list[PlannedEntity] = Field(default_factory=list)
    metric: Optional[str] = None
    period: Optional[Period] = None
    filters: list[PlannedFilter] = Field(default_factory=list)
    sort: Optional[PlannedSort] = None
    limit: Optional[int] = None
    # The planner's own confidence in this reading. Low confidence is
    # routed to a clarification rather than executed — same policy the
    # QueryIR path already applies.
    confidence: float = 1.0
    # Free text the planner may use to explain an ambiguity. Never shown
    # verbatim to the user as an ANSWER; only used when intent is
    # "clarification", so a model that tries to answer the question here
    # cannot reach the user as though it were data.
    clarification: Optional[str] = None

    @field_validator("limit")
    @classmethod
    def _sane_limit(cls, v: int | None) -> int | None:
        if v is None:
            return None
        return max(1, min(int(v), 100))

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(float(v), 1.0))


# Strict JSON schema handed to the provider's structured-output mode, so
# a malformed shape is a provider-level error rather than something the
# validator has to catch. Hand-written to mirror the models above —
# same reasoning as llm_client.QUERY_IR_JSON_SCHEMA.
QUERY_PLAN_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent", "entities", "metric", "period", "filters",
        "sort", "limit", "confidence", "clarification",
    ],
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "advisor_profile", "hierarchy", "reverse_hierarchy", "roster",
                "leaderboard", "comparison", "attendance_filter",
                "entity_summary", "clarification", "greeting", "help", "fallback",
            ],
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "value"],
                "properties": {
                    "type": {
                        "type": "string",
                        # The CANONICAL levels only. Aliases validate
                        # (see EntityType) but are never offered — the
                        # model should not be taught a retired name.
                        "enum": list(hierarchy.HIERARCHY_LEVELS),
                    },
                    "value": {"type": "string"},
                },
            },
        },
        "metric": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "period": {"anyOf": [{"type": "string", "enum": list(PERIODS)}, {"type": "null"}]},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "operator", "value"],
                "properties": {
                    "field": {"type": "string"},
                    "operator": {"type": "string", "enum": ["=", "!=", ">", ">=", "<", "<=", "in"]},
                    "value": {"anyOf": [
                        {"type": "string"}, {"type": "number"},
                        {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}]}},
                        {"type": "null"},
                    ]},
                },
            },
        },
        "sort": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["metric", "direction"],
                    "properties": {
                        "metric": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "direction": {"type": "string", "enum": ["asc", "desc"]},
                    },
                },
            ]
        },
        "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "confidence": {"type": "number"},
        "clarification": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
}
