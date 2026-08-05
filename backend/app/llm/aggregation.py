"""
THE aggregation engine. Every metric value in the product comes from here.

WHY IT EXISTS. Five places computed metric values independently:
team_service, company_service and hierarchy_service each hand-rolled the
same `func.sum(...)` summary; comparison_service read the ontology
binding and re-derived a roll-up rule; query_compiler applied
`binding.agg`. They agreed on additive metrics and disagreed on
percentages — a team's achievement was 50% through one path and 82.7%
through another for the same two advisors, because one averaged the
advisors' percentages and the other divided the sums.

Three responsibilities, and deliberately no more:

1. ROLL-UP RULE. How per-advisor values become a group value, read from
   `MetricDef.rollup` — SUM, AVG or RATIO. The rule belongs to the
   metric, not to whichever service happens to be asking.
2. SCOPE. Who is in the group, always via `hierarchy.scope_filter`, so a
   leaderboard, a comparison and a breakdown cannot disagree about
   membership.
3. ONE VALUE, ONE SUMMARY. `metric_value()` for a single number and
   `summary()` for the shared summary shape the services return.

What this module deliberately does NOT do: sorting, limiting, filtering
by other metrics, pagination, or response shaping. `query_compiler` still
owns query construction — it asks this module only what the value
EXPRESSION is, so that ranking and aggregating cannot drift apart either.
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import func, literal
from sqlalchemy.orm import Session

from app.database.models import (
    Advisor, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import hierarchy
from app.llm.metric_ontology import METRICS, ColumnBinding, Rollup

# The advisor leaf is never rolled up — one row IS the value.
LEAF = "advisor"


def rollup_for(metric_key: str) -> Rollup:
    """The declared roll-up rule, defaulting to SUM for anything the
    ontology hasn't spoken about."""
    metric = METRICS.get(metric_key)
    return metric.rollup if metric else Rollup.SUM


def needs_rollup(level: str, binding: ColumnBinding) -> bool:
    """A team-named binding IS the group's truth (the row is the team),
    so it is never aggregated; the leaf never is either."""
    return level != LEAF and not binding.team_named


def _working_day_denominator(metric_key: str, binding) -> Optional[float]:
    """`perAdvisorPerDay x workingDays` for one advisor row, or None.

    Both halves are read from their owners — the per-day target from the
    metric ontology, the working days from working_days.for_period — so
    this function holds neither number itself. The period comes from the
    BINDING when it declares one and the metric otherwise, which is the
    same precedence the rest of the compiler uses for period-specific
    bindings.
    """
    from app.llm import working_days
    from app.llm.metric_ontology import METRICS, daily_target_rate

    rate = daily_target_rate(metric_key)
    if not rate:
        # RATIO plus working_day_scaled but no declared target is a
        # declaration error, not a runtime state — the caller falls back
        # to an additive reading rather than emitting a bad denominator.
        return None
    metric = METRICS.get(metric_key)
    period = binding.period or (metric.period if metric else None)
    return rate * working_days.for_period(period)


def value_expression(
    binding: ColumnBinding,
    metric_key: str,
    level: str,
    numerator=None,
    denominator=None,
    expr=None,
):
    """The SQL expression for this metric at this level.

    `expr` / `numerator` / `denominator` may be passed already rebound
    onto a join alias by the caller — `query_compiler` does this when the
    same model is joined twice under different periods. Everything else
    about the rule lives here.

    RATIO is the case the old two-way `agg` flag could not express, and
    the reason a percentage had two different answers depending on who
    asked. NULLIF guards a group whose denominator sums to zero: the
    value compiles to NULL, which the callers already render as "no data"
    rather than as 0%.
    """
    expr = binding.expr if expr is None else expr

    # Working-day scaled rates (CR %, Connect %, Meeting %). The spec
    # measures these against a target rather than a stored denominator:
    #
    #     value / (teamSize x perAdvisorPerDay x workingDays) x 100
    #
    # Expressed here rather than in the ontology because `workingDays`
    # depends on the PERIOD, which is a query-time fact, while a
    # ColumnBinding is a static declaration. The two static halves stay
    # declared: the numerator is the binding's, and the per-day figure is
    # MetricDef.daily_target_rate.
    #
    # The denominator is a per-ADVISOR-ROW constant, so SUM over the
    # scope's rows yields teamSize x rate x workingDays exactly — the
    # spec's formula, using the RATIO machinery already here rather than
    # a second calculation path. teamSize is therefore never computed
    # separately; it is the row count the SUM already walks.
    if getattr(binding, "working_day_scaled", False):
        per_row = _working_day_denominator(metric_key, binding)
        if per_row is None:
            return func.sum(expr)
        numerator = binding.ratio_numerator if numerator is None else numerator
        if not needs_rollup(level, binding):
            # One advisor: teamSize is 1, so the denominator is the
            # per-row constant itself.
            return numerator / func.nullif(literal(per_row), 0)
        return func.sum(numerator) / func.nullif(func.sum(literal(per_row)), 0)

    if not needs_rollup(level, binding):
        return expr

    rule = rollup_for(metric_key)
    if rule is Rollup.RATIO:
        numerator = binding.ratio_numerator if numerator is None else numerator
        denominator = binding.ratio_denominator if denominator is None else denominator
        if numerator is None or denominator is None:
            # A metric declaring RATIO without components is a
            # declaration error, not a runtime condition — fall back to
            # the safest additive reading rather than emitting bad SQL.
            return func.sum(expr)
        return func.sum(numerator) / func.nullif(func.sum(denominator), 0)
    if rule is Rollup.AVG:
        return func.avg(expr)
    return func.sum(expr)


def binding_for(metric_key: str, level: str) -> Optional[ColumnBinding]:
    """The binding this engine aggregates from.

    For any GROUP level it deliberately uses the ADVISOR binding and
    rolls it up, rather than the level's own binding. Two reasons:

    - Uniformity. Every level then computes the same way, which is what
      makes a BCM, a zonal head, a unit head and a team comparable at
      all. A per-level implementation is exactly what produced five
      answers to one question.
    - Team-named sources are a DIFFERENT question. `TeamTarget` carries
      the sheet's own team figure; it is not this engine's roll-up of
      advisor rows, and whether it or the computed value is authoritative
      is an open business decision (Phase 1, Q3). Services that want the
      sheet figure read it explicitly and label it as such, instead of it
      arriving here disguised as an aggregate.
    """
    metric = METRICS.get(metric_key)
    if metric is None:
        return None
    if level == LEAF:
        return metric.bindings.get(LEAF)

    advisor_binding = metric.bindings.get(LEAF)
    if advisor_binding is not None and not advisor_binding.team_named:
        return advisor_binding

    # No advisor-rooted source to roll up (or the only one is team-named):
    # fall back to whatever the level declares for itself, which may be
    # nothing. Read straight from the ontology — `query_compiler` now
    # delegates its binding choice HERE, so calling back into it would
    # recurse.
    return metric.bindings.get(level)


def metric_value(db: Session, level: str, value: str, metric_key: str) -> Optional[float]:
    """ONE metric for ONE entity at any level — advisor, BCM, zonal head,
    unit head, team, or any groupable attribute.

    The single implementation behind comparisons, summaries and any
    service that needs a scalar. Returns None when the metric has no
    binding for that level or the group has no contributing rows, which
    callers render as "no data" rather than zero.
    """
    binding = binding_for(metric_key, level)
    if binding is None:
        return None

    expression = value_expression(binding, metric_key, level)

    if binding.team_named:
        query = db.query(expression).filter(
            binding.model.team.ilike(value)
        )
    else:
        query = db.query(expression).select_from(Advisor)
        # Advisor is the root — see query_compiler._join_fact_table.
        if binding.model is not Advisor:
            query = query.join(binding.model, binding.model.wid == Advisor.wid)
        query = query.filter(scope(level, value), Advisor.in_master_sheet.is_(True))
        # FIX 3: a ratio whose denominator lives on another fact table
        # declares it via ColumnBinding.join_models. Without the join,
        # SQLAlchemy cross-joins the table instead of failing, so a
        # comparison would quietly disagree with the leaderboard.
        for extra in getattr(binding, "join_models", ()) or ():
            query = query.join(extra, extra.wid == Advisor.wid)
        if binding.period is not None:
            query = query.filter(binding.model.period == binding.period)

    result = query.scalar()
    return float(result) if result is not None else None


def scope(level: str, value: str):
    """Membership, delegated to the hierarchy so aggregation cannot
    invent its own traversal (Phase 3 made this the one definition)."""
    return hierarchy.scope_filter(level, value)


def headcount(db: Session, level: str, value: str) -> int:
    """Advisors in scope. Master-sheet only, like every other read."""
    return (
        db.query(func.count(Advisor.wid))
        .filter(scope(level, value), Advisor.in_master_sheet.is_(True))
        .scalar()
    ) or 0


# ---------------------------------------------------------------------
# The shared summary
# ---------------------------------------------------------------------

# Fields every level's summary carries, and the metric each is computed
# from. Declared once so team/company/unit-head summaries cannot drift
# into reporting different things — they were three near-identical
# hand-written queries before.
SUMMARY_METRICS: dict[str, str] = {
    "connects": "total_connects",
    "overdue": "overdue",
    "pipeline": "pipeline_value",
    "mtd_target": "mtd_target",
    "mtd_cleared": "mtd_cleared",
    "ytd_cleared": "ytd_cleared",
}


def summary(db: Session, level: str, value: str) -> dict:
    """The aggregate shape every level's summary shares.

    One implementation replacing three near-copies. `ytd_target` has no
    metric of its own in the ontology, so it is read directly from the
    same Performance table the ytd_cleared binding uses — the one field
    here that is not yet expressible as a metric.
    """
    result: dict[str, Any] = {
        "level": level,
        "level_label": hierarchy.label_for(level),
        "value": value,
        "advisors": headcount(db, level, value),
    }
    for field, metric_key in SUMMARY_METRICS.items():
        result[field] = metric_value(db, level, value, metric_key) or 0
    result["ytd_target"] = _period_target(db, level, value, PerformancePeriod.YTD) or 0
    return result


def _period_target(db: Session, level: str, value: str, period) -> Optional[float]:
    total = (
        db.query(func.sum(Performance.target))
        .select_from(Advisor)
        .join(Performance, Performance.wid == Advisor.wid)
        .filter(scope(level, value), Advisor.in_master_sheet.is_(True),
                Performance.period == period)
        .scalar()
    )
    return float(total) if total is not None else None
