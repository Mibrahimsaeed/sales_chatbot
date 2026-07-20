"""
Query Compiler (Part 5.5) — reads a VALID QueryIR and builds one
SQLAlchemy query generically, using the column bindings declared in
metric_ontology.py. Replaces sql_generator.py's RESOLVERS[(metric, level)]
registry: adding a new filter/metric/level combination is now an ontology
entry, not a new hand-written function.

Safety property preserved from the old design: 100% parameterized
SQLAlchemy ORM calls. The LLM never touches a SQL string — it only ever
populates a validated QueryIR; everything here is deterministic and
auditable.

Scope of this reference implementation: supports any single sort metric
plus any number of AND-combined filters (entity filters on team/company/
attendance_status, and comparator filters on OTHER metrics), at advisor or
team level. Company-level rollups and TeamTarget-sourced metrics (which
have no advisor `wid` to roll up from) are supported where the ontology
binding allows; a metric filtered/sorted at a level with no binding
degrades to "I don't have a way to answer that yet" — the same fail-soft
behavior sql_generator.py had, just derived from the ontology instead of a
missing dict entry.

Joins are deduplicated by (model, period), not model class alone (see
`_join_fact_table`): a query that sorts by one Performance-backed metric
and filters by a DIFFERENT Performance-backed metric with a different
period (e.g. "sort by MTD cleared, filter YTD cleared > X") joins the
second period via `aliased()` instead of reusing the first join, so the
filter binds to its own period's row.
"""

from __future__ import annotations

import operator as op

from sqlalchemy import asc, desc, func
from sqlalchemy.orm import Session, aliased

from app.database.models import Advisor, Attendance, PerformancePeriod
from app.llm.metric_ontology import METRICS, ColumnBinding
from app.llm.query_ir import QueryIR, Filter

_LEVEL_GROUP_COLUMN = {"team": Advisor.team, "company": Advisor.company}

_COMPARATORS = {
    "=": op.eq, "!=": op.ne, ">": op.gt, ">=": op.ge, "<": op.lt, "<=": op.le,
}


def _apply_comparator(column, operator: str, value):
    """`in` needs SQLAlchemy's Column.in_() — it isn't a Python `operator`
    module function like the rest of _COMPARATORS, so it's handled as an
    explicit branch rather than jammed into that dict."""
    if operator == "in":
        values = value if isinstance(value, (list, tuple)) else [value]
        return column.in_(values)
    return _COMPARATORS.get(operator, op.eq)(column, value)


def _order(column, direction: str):
    return asc(column) if direction == "asc" else desc(column)


def _binding_for(metric_key: str, level: str) -> ColumnBinding | None:
    metric = METRICS.get(metric_key)
    if not metric:
        return None
    return metric.bindings.get(level)


def _value_expr(binding: ColumnBinding, agg_for_rollup: bool, level: str):
    """The SQL expression to select/order/filter by for this binding at
    this level: the raw per-advisor expression, or the team/company
    rollup aggregation (sum/avg) when the level requires grouping."""
    if not agg_for_rollup or level == "advisor" or binding.team_named:
        return binding.expr
    return func.avg(binding.expr) if binding.agg == "avg" else func.sum(binding.expr)


def compile_and_run(db: Session, ir: QueryIR) -> list[dict] | None:
    """Returns a list of {wid, name, team, company, value, ...} rows, or
    None if the IR (after validation) still can't be answered — e.g. the
    sort metric has no binding at the requested level."""
    level = ir.subject_level
    sort_metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
    if not sort_metric_key:
        return None

    binding = _binding_for(sort_metric_key, level)
    if not binding:
        return None

    if binding.team_named:
        return _run_team_named(db, ir, binding)
    return _run_advisor_rooted(db, ir, binding, sort_metric_key)


def _run_team_named(db, ir: QueryIR, binding: ColumnBinding) -> list[dict] | None:
    """TeamTarget-style metrics: the row already IS the team, no Advisor
    join, no roll-up. Filters here are limited to team-name membership
    (e.g. comparisons) since there's no per-advisor row to join other
    metrics' fact tables against."""
    model = binding.model
    value_col = binding.expr

    query = db.query(model.team.label("name"), value_col.label("value"))

    for f in ir.filters:
        if f.field == "team":
            query = query.filter(model.team.ilike(f"%{f.value}%"))

    subject_names = [s.resolved_id or s.value for s in ir.subjects if s.type == "team"]
    if subject_names:
        query = query.filter(model.team.in_(subject_names))

    query = query.order_by(_order(value_col, ir.sort.direction))
    if ir.limit:
        query = query.limit(ir.limit)

    rows = query.all()
    return [{"wid": None, "name": r.name, "team": r.name, "company": None, "value": r.value} for r in rows]


def _apply_entity_filters(query, ir: QueryIR):
    for f in ir.filters:
        if f.field == "team":
            query = query.filter(Advisor.team.ilike(f"%{f.value}%"))
        elif f.field == "company":
            query = query.filter(Advisor.company.ilike(f"%{f.value}%"))
        elif f.field == "advisor":
            query = query.filter(Advisor.name.ilike(f"%{f.value}%"))
    return query


def _apply_subject_filter(query, ir: QueryIR):
    """Comparisons compile as a normal query with an added 'subject name is
    one of these' filter — no separate comparison code path needed."""
    if ir.intent != "comparison" or not ir.subjects:
        return query
    if ir.subject_level == "advisor":
        names = [s.resolved_id or s.value for s in ir.subjects if s.type == "advisor"]
        if names:
            query = query.filter(Advisor.name.in_(names))
    elif ir.subject_level == "team":
        names = [s.resolved_id or s.value for s in ir.subjects if s.type == "team"]
        if names:
            query = query.filter(Advisor.team.in_(names))
    elif ir.subject_level == "company":
        names = [s.resolved_id or s.value for s in ir.subjects if s.type == "company"]
        if names:
            query = query.filter(Advisor.company.in_(names))
    return query


def _join_fact_table(query, joined: dict, model: type, period):
    """Joins `model` to the query, keyed by (model, period) rather than
    model alone. If this exact model is already joined under a DIFFERENT
    period, the new join uses aliased() instead of reusing the first
    join — otherwise a filter on e.g. YTD cleared would silently bind to
    an MTD join already present for the sort metric. Returns (query,
    entity), where entity is the model class (first join) or its alias
    (subsequent joins of the same model at a different period); all
    column access for this join must go through entity, not the raw
    model class.
    """
    key = (model, period)
    if key in joined:
        return query, joined[key]

    already_joined_other_period = any(m is model for m, _p in joined)
    entity = aliased(model) if already_joined_other_period else model
    query = query.join(entity, entity.wid == Advisor.wid)
    if period is not None:
        query = query.filter(entity.period == period)
    joined[key] = entity
    return query, entity


def _rebind_to_entity(expr, entity, model: type):
    """binding.expr is written against the declared model class. If the
    join for this binding used an alias (only happens when the SAME
    model is joined twice in one query under different `period` values),
    the expression must be rebound onto that alias. Every period-bearing
    binding in metric_ontology.py is a single raw column, never a
    computed expression, so a plain attribute lookup is sufficient — fail
    loudly instead of silently compiling against the wrong table if that
    invariant is ever broken by a future ontology entry.
    """
    if entity is model:
        return expr
    key = getattr(expr, "key", None)
    if key is None:
        raise NotImplementedError(
            "A computed (non-column) metric expression needs an aliased "
            "join — extend _rebind_to_entity to support this case."
        )
    return getattr(entity, key)


def _apply_attendance_filter(query, ir: QueryIR, joined: dict):
    status_filters = [f for f in ir.filters if f.field == "attendance_status"]
    if not status_filters:
        return query
    query, entity = _join_fact_table(query, joined, Attendance, None)
    for f in status_filters:
        query = query.filter(entity.biometric_status == f.value)
    return query


def _apply_metric_filters(query, ir: QueryIR, level: str, joined: dict):
    """Filters on a metric OTHER than the sort metric (Root Cause #1/#3 fix:
    "high sales but poor attendance" filters on late_count while sorting
    by mtd_cleared). Each distinct (model, period) fact table is joined once."""
    for f in ir.filters:
        if f.field not in METRICS:
            continue
        f_binding = _binding_for(f.field, level)
        if not f_binding or f_binding.team_named:
            continue  # can't combine a team-named metric filter into an advisor-rooted query
        query, entity = _join_fact_table(query, joined, f_binding.model, f_binding.period)
        expr = _rebind_to_entity(f_binding.expr, entity, f_binding.model)
        query = query.filter(_apply_comparator(expr, f.operator, f.value))
    return query


def _run_advisor_rooted(db, ir: QueryIR, binding: ColumnBinding, sort_metric_key: str) -> list[dict] | None:
    level = ir.subject_level
    joined: dict = {}
    is_rollup = level in ("team", "company")

    value_expr = _value_expr(binding, agg_for_rollup=is_rollup, level=level)

    if level == "advisor":
        query = db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company)
        query, _sort_entity = _join_fact_table(query, joined, binding.model, binding.period)
        query = query.add_columns(value_expr.label("value"))
    else:
        group_col = _LEVEL_GROUP_COLUMN[level]
        query = db.query(group_col.label("name"))
        query, _sort_entity = _join_fact_table(query, joined, binding.model, binding.period)
        query = query.add_columns(value_expr.label("value"))
        query = query.filter(group_col.isnot(None))

    query = _apply_entity_filters(query, ir)
    query = _apply_subject_filter(query, ir)
    query = _apply_attendance_filter(query, ir, joined)
    query = _apply_metric_filters(query, ir, level, joined)

    if level != "advisor":
        query = query.group_by(_LEVEL_GROUP_COLUMN[level])

    # value_expr is already the right expression to order by: the raw
    # per-advisor column at advisor level, or the sum()/avg()-wrapped
    # rollup at team/company level (from _value_expr above).
    query = query.order_by(_order(value_expr, ir.sort.direction))
    if ir.limit:
        query = query.limit(ir.limit)

    rows = query.all()

    if level == "advisor":
        return [dict(r._mapping) for r in rows]
    return [{"wid": None, "name": r.name, "team": r.name if level == "team" else None,
              "company": r.name if level == "company" else None, "value": r.value} for r in rows]
