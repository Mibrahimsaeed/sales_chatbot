"""
Side-by-side comparison of two or more entities.

Comparison existed downstream (QueryIR.intent=="comparison", the
compiler's subject filter, format_ir_comparison_reply) but was reachable
ONLY through the LLM semantic parser. With the LLM unavailable — or
simply not consulted — "Compare Graana and Agency21" fell through to the
metric-help message, and "…by revenue" degraded to a plain leaderboard
that silently dropped both named entities. This module is the
deterministic path that was missing.

Two shapes, per the same function:

  - no metric named  -> the DEFAULT KPI SET (advisors, connects,
                        meetings, revenue, bookings, pipeline,
                        attendance), because "compare A and B" is a
                        request for an overview, not for one number.
  - a metric named   -> that metric alone.

KPIs are computed from metric_ontology bindings rather than hand-written
SQL, so a comparison automatically honours each metric's own aggregation
rule — rate-like metrics (attendance, conversion) AVERAGE across an
entity while count/currency metrics SUM. Hard-coding sum() here would
have reported an entity's attendance rate as the sum of its advisors'
percentages.

Entities keep their own types: a comparison can legitimately span levels
("compare Blue Area with Graana" — a team against a company), so each
target carries its own (level, value) and is resolved through
hierarchy.column_for().
"""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.exception import NotFoundError
from app.database.models import Advisor
from app.llm import aggregation, hierarchy
from app.llm.metric_ontology import METRICS

# The overview shown when no metric was named. Ordered as a human would
# read a scorecard: how many people, then activity, then outcomes.
# "advisors" is a headcount, not an ontology metric, so it is computed
# separately; the rest are ontology keys and inherit their own bindings.
DEFAULT_KPIS: tuple[str, ...] = (
    "total_connects",
    "total_meetings",
    "mtd_cleared",
    "bookings",
    "pipeline_value",
    "attendance_rate",
)

# Display labels for the headcount row, which has no MetricDef to read
# a label from.
_HEADCOUNT_KEY = "advisors"
_HEADCOUNT_LABEL = "Advisors"


def _headcount(db: Session, level: str, value: str) -> int:
    """Scope comes from the engine, which delegates to
    hierarchy.scope_filter — so a comparison counts exactly the advisors
    a breakdown or a leaderboard would."""
    return aggregation.headcount(db, level, value)


def _metric_value(db: Session, level: str, value: str, metric_key: str) -> float | None:
    """PHASE 4: delegates to the aggregation engine.

    This used to re-derive the roll-up rule here — read the advisor
    binding, then look up the TEAM binding to find out whether the metric
    averages or sums. That reasoning was correct and duplicated, so a
    comparison and a leaderboard over the same entity could disagree.
    Kept as a thin named function because the comparison code reads
    better for it, and because the engine's argument order matches.
    """
    return aggregation.metric_value(db, level, value, metric_key)


def get_comparison(
    db: Session, targets: list[tuple[str, str]], metric=None
) -> dict:
    """Compare `targets` — a list of (level, value) — side by side.

    `metric` is one key, several keys, or None for the default KPI set.
    Several is not a new capability: this function has walked a TUPLE of
    keys since it was written, because the no-metric case renders the
    whole default set. Accepting a list here is what finally lets a user
    choose that set — "compare X and Y on connects and answered calls"
    was reaching this function with one of its two measures already
    discarded upstream.

    Raises NotFoundError when a named entity has no advisors at all: a
    comparison against an entity that doesn't exist would render as a
    column of zeros, which reads as a real (and very poor) result rather
    than as "I couldn't find that"."""
    if len(targets) < 2:
        raise ValueError("a comparison needs at least two targets")

    if metric is None:
        kpi_keys = DEFAULT_KPIS
    elif isinstance(metric, str):
        kpi_keys = (metric,)
    else:
        # Order preserved (it is the order the user named them in) and
        # duplicates dropped, so a measure named twice renders one column.
        kpi_keys = tuple(dict.fromkeys(metric))
    entities: list[dict] = []

    for level, value in targets:
        count = _headcount(db, level, value)
        if not count:
            raise NotFoundError(f"No {hierarchy.label_for(level)} matching '{value}'")

        values: dict[str, float | None] = {_HEADCOUNT_KEY: float(count)}
        for key in kpi_keys:
            values[key] = _metric_value(db, level, value, key)

        entities.append({
            "level": level,
            "level_label": hierarchy.label_for(level),
            "value": value,
            "advisors": count,
            "metrics": values,
        })

    # Row order drives the reply. Headcount is only meaningful for the
    # overview — when one metric was asked for, showing headcount beside
    # it invites the reader to treat it as part of the answer.
    rows: list[dict] = []
    if not metric:
        rows.append({"key": _HEADCOUNT_KEY, "label": _HEADCOUNT_LABEL, "is_percentage": False})
    for key in kpi_keys:
        definition = METRICS.get(key)
        rows.append({
            "key": key,
            "label": definition.label if definition else key,
            "is_percentage": bool(definition and "%" in definition.label),
        })

    return {
        "metric": metric,
        "entities": entities,
        "rows": rows,
        "winners": _winners(rows, entities),
    }


def _winners(rows: list[dict], entities: list[dict]) -> dict[str, str | None]:
    """Which entity leads on each row. Computed here rather than in the
    formatter so the same judgement is available to any consumer (API,
    frontend) without re-deriving it — and so "higher is better" is
    stated once.

    Every KPI in the default set is higher-is-better, including pipeline
    and overdue-free measures; a metric where that is false would need
    its own direction flag on MetricDef before being added here."""
    winners: dict[str, str | None] = {}
    for row in rows:
        key = row["key"]
        best_value: float | None = None
        best_entity: str | None = None
        tied = False
        for entity in entities:
            value = entity["metrics"].get(key)
            if value is None:
                continue
            if best_value is None or value > best_value:
                best_value, best_entity, tied = value, entity["value"], False
            elif value == best_value:
                tied = True
        winners[key] = None if (tied or best_entity is None) else best_entity
    return winners
