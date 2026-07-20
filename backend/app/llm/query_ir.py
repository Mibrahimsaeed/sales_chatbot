"""
QueryIR — replaces the single-metric, single-filter QueryPlan (still in
query_planner.py, kept only as the rule-based fast path / fail-soft
degrade target — see plan_to_ir() below).

This is the one structure both the LLM semantic parser (semantic_parser.py)
and the deterministic query compiler (query_compiler.py) agree on. It is
able to express every compound example in the redesign brief without a new
field per query shape:

  - multiple named subjects (comparisons)              -> subjects[]
  - multiple simultaneous filters, AND-combined         -> filters[]
  - thresholds/comparators ("more than 80%")            -> filters[].operator
  - one sort metric + independent filter metrics        -> sort vs filters[]
  - per-field confidence instead of one whole-query one -> *.confidence
  - unresolved pieces for targeted clarification        -> missing[]

Nothing here talks to the database. Grounding real gazetteer/ontology
values into `resolved_id` / validity happens in ir_validator.py; turning a
valid IR into SQL happens in query_compiler.py.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

Level = Literal["advisor", "team", "company"]
Operator = Literal["=", "!=", ">", ">=", "<", "<=", "in"]
Intent = Literal["leaderboard", "comparison", "lookup", "trend", "filtered_list", "clarify"]


class Subject(BaseModel):
    type: Level
    value: str
    resolved_id: Optional[str] = None
    match_confidence: float = 1.0


class MetricRef(BaseModel):
    key: str
    confidence: float = 1.0


class Filter(BaseModel):
    field: str                                   # a metric key, or "team" | "company" | "advisor" | "attendance_status"
    operator: Operator = "="
    value: Optional[Union[str, float, int, list]] = None
    confidence: float = 1.0


class TimeRange(BaseModel):
    mode: Literal["snapshot", "compare"] = "snapshot"
    period: Literal["MTD", "YTD", "3M"] = "MTD"
    compare_to: Optional[str] = None             # e.g. previous period key — Phase 4, not compiled yet


class Sort(BaseModel):
    metric: Optional[str] = None
    direction: Literal["asc", "desc"] = "desc"


class QueryIR(BaseModel):
    intent: Intent
    subject_level: Level = "advisor"
    subjects: list[Subject] = Field(default_factory=list)
    metric: Optional[MetricRef] = None
    filters: list[Filter] = Field(default_factory=list)      # AND-combined
    time_range: TimeRange = Field(default_factory=TimeRange)
    sort: Sort = Field(default_factory=Sort)
    limit: Optional[int] = 10
    group_by: Optional[Level] = None
    overall_confidence: float = 1.0
    missing: list[str] = Field(default_factory=list)
    # observability only (persisted in ChatLog.resolved_ir): which NLU mode
    # served this IR — not part of the LLM output schema, never validated
    nlu_mode: Optional[str] = None


def plan_to_ir(plan, entities: dict) -> QueryIR:
    """Fail-soft degrade path (Part 5.1 error handling): wraps the existing
    rule-based query_planner.QueryPlan into a minimal, single-metric,
    single-filter QueryIR. Used both as the normal fast path for simple
    leaderboard queries (skip the LLM call entirely when the rule-based
    planner already resolved it and the text doesn't look compound) and as
    the degrade target when the LLM call fails or returns invalid JSON.
    """
    filters: list[Filter] = []
    if entities.get("team"):
        filters.append(Filter(field="team", operator="=", value=entities["team"]))
    if entities.get("company"):
        filters.append(Filter(field="company", operator="=", value=entities["company"]))
    if entities.get("attendance_status"):
        filters.append(Filter(field="attendance_status", operator="=", value=entities["attendance_status"]))

    return QueryIR(
        intent="leaderboard",
        subject_level=plan.level or "advisor",
        metric=MetricRef(key=plan.metric) if plan.metric else None,
        filters=filters,
        sort=Sort(metric=plan.metric, direction="asc" if plan.ascending else "desc"),
        limit=plan.limit or 10,
        overall_confidence=0.75,
    )
