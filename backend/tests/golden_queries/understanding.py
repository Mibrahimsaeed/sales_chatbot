"""
The normalised record a golden case asserts against.

WHY A NORMALISED RECORD. "Understanding" is not stored in one place. A
metric ranking becomes a QueryIR; "who is X's unit head" becomes a
QueryPlan and never builds an IR at all; an unanswerable measure becomes
a clarification with no IR or plan. Asserting directly against QueryIR
would therefore cover only the leaderboard family and silently skip every
hierarchy, roster and refusal case — which is most of what users ask.

So each resolution is flattened into ONE shape with the same field names
regardless of which path produced it. A golden case then reads the same
whether the answer came from the rule-based planner or the semantic
parser, which is also the property that makes the file useful as a
contract: a change that moves a query between paths is only a regression
if the UNDERSTANDING changes.

Nothing here executes SQL or formats a reply. This is semantic
understanding only, as specified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Understanding:
    """What the pipeline decided the user meant."""

    # What KIND of question this is. Normalised across paths: a QueryIR
    # with intent="leaderboard" and a QueryPlan with action="leaderboard"
    # both report "leaderboard".
    intent: str
    # The measure, as an ontology key. None when the question is not
    # about a measure (a roster, a manager lookup) or when none resolved.
    metric: Optional[str] = None
    # "MTD" | "YTD" | "3M" | "DAILY", or None when the question named no
    # window and none was implied.
    period: Optional[str] = None
    # The hierarchy level the answer is about.
    level: Optional[str] = None
    # The named subject, for questions about one entity.
    entity: Optional[str] = None
    # (field, operator, value) triples, sorted so declaration order in a
    # case never matters.
    filters: tuple[tuple[str, str, Any], ...] = ()
    # The THRESHOLD operators, sorted. Derived from filters on a metric
    # only — an entity filter is always "=" and including it would bury
    # the operator a case actually cares about. Called out separately
    # because comparator polarity is its own regression class: "no more
    # than 50" once parsed as "> 50".
    comparators: tuple[str, ...] = ()
    # "asc" | "desc" | None. None means the user named no direction, so
    # the metric's polarity decides downstream.
    ranking: Optional[str] = None
    limit: Optional[int] = None
    # Populated for a refusal, so a case can assert that the reply
    # explains itself rather than just that it refused.
    reason: Optional[str] = None

    def compared_on(self, keys: tuple[str, ...]) -> dict:
        return {key: getattr(self, key) for key in keys}


# Plan actions whose name IS the intent. Listed rather than inferred so a
# new action shows up here as a deliberate decision instead of leaking
# through with whatever name it happens to have.
_PLAN_INTENTS = {
    "leaderboard": "leaderboard",
    "comparison": "comparison",
    "comparison_incomplete": "comparison_incomplete",
    "roster": "roster",
    "breakdown": "breakdown",
    "summary": "summary",
    "lookup": "advisor_profile",
    "advisor_metric": "advisor_metric",
    "reverse_hierarchy": "reverse_hierarchy",
    "attendance_filter": "attendance_filter",
    "clarify_person": "clarify_person",
    "clarify_ambiguous": "clarify_ambiguous",
    "clarify_metric": "clarify_metric",
    "unresolved": "unresolved",
}


def _filters_from_ir(ir) -> tuple[tuple[str, str, Any], ...]:
    return tuple(sorted(
        (f.field, f.operator, f.value) for f in ir.filters
    ))


def _comparators(filters: tuple[tuple[str, str, Any], ...]) -> tuple[str, ...]:
    """Threshold operators only — the ones on a METRIC field.

    An entity filter ("team", "=", "Blue Area") contributes an "=" that
    says nothing about how the user compared a number, and mixing it in
    makes every entity-scoped threshold case read as ("=", ">").
    """
    from app.llm.metric_ontology import METRICS

    return tuple(sorted(op for f, op, _v in filters if f in METRICS))


def _filters_from_plan(plan) -> tuple[tuple[str, str, Any], ...]:
    """A plan's constraints in the same shape an IR's filters take.

    The entity scope is NOT included: it is reported as `entity`/`level`,
    because "the team this is about" and "a condition rows must satisfy"
    are different things and conflating them would make a roster case
    look like a filtered ranking.
    """
    metric = plan.metric or "?"
    return tuple(sorted(
        (metric, t["operator"], t["value"]) for t in (plan.thresholds or [])
    ))


def from_resolution(resolution) -> Understanding:
    """Flatten a nlu_pipeline.Resolution into the golden shape."""
    kind = resolution.kind

    if kind == "shortcut":
        return Understanding(intent=f"shortcut:{resolution.shortcut_intent}")

    if kind == "multi":
        # A compound message. Reported as its own intent with the number
        # of segments, so a case can assert the split without this module
        # having to invent a merged understanding.
        return Understanding(
            intent="multi",
            limit=len(resolution.sections or ()),
        )

    if kind == "paginate":
        return Understanding(intent="paginate")

    if kind == "ir" and resolution.ir is not None:
        ir = resolution.ir
        filters = _filters_from_ir(ir)
        return Understanding(
            intent=ir.intent,
            metric=(ir.sort.metric or (ir.metric.key if ir.metric else None)),
            period=ir.time_range.period,
            level=ir.subject_level,
            entity=(ir.subjects[0].value if ir.subjects else None),
            filters=filters,
            comparators=_comparators(filters),
            ranking=ir.sort.direction,
            limit=ir.limit,
        )

    if kind == "plan" and resolution.plan is not None:
        plan = resolution.plan
        filters = _filters_from_plan(plan)
        return Understanding(
            intent=_PLAN_INTENTS.get(plan.action, plan.action),
            metric=plan.metric,
            period=plan.period,
            level=plan.level,
            entity=plan.entity_value,
            filters=filters,
            comparators=_comparators(filters),
            ranking=(None if plan.ascending is None else ("asc" if plan.ascending else "desc")),
            limit=plan.limit,
        )

    if kind == "clarify":
        # A clarification carries either a plan (rule-based refusals) or
        # a partial IR (slot-filling). Prefer whichever is present so the
        # case can still assert the level/metric that WAS understood.
        plan = resolution.plan
        ir = resolution.ir
        intent = "clarify"
        if plan is not None:
            intent = _PLAN_INTENTS.get(plan.action, plan.action)
        return Understanding(
            intent=intent,
            metric=(plan.metric if plan is not None else
                    (ir.sort.metric if ir is not None else None)),
            period=(plan.period if plan is not None else
                    (ir.time_range.period if ir is not None else None)),
            level=(plan.level if plan is not None else
                   (ir.subject_level if ir is not None else None)),
            entity=(plan.entity_value if plan is not None else None),
            reason=resolution.clarify_message,
        )

    return Understanding(intent=f"unhandled:{kind}")


def observe(text: str, db, session_id: Optional[str] = None) -> Understanding:
    """Run one question through the real pipeline and flatten the result.

    Deliberately calls nlu_pipeline.resolve, not the planner directly:
    routing IS part of understanding. A query that the planner scores
    correctly but the pipeline sends somewhere else is still wrong, and
    that is exactly the class of bug this framework exists to catch.
    """
    from app.llm.nlu_pipeline import resolve

    return from_resolution(resolve(text, db, session_id=session_id))
