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

from sqlalchemy import asc, desc, func, select
from sqlalchemy.orm import Session, aliased

from app.database.models import Advisor, Attendance, PerformancePeriod
from app.llm import aggregation, hierarchy
from app.llm.metric_ontology import (
    METRICS, ColumnBinding, lower_is_better, metric_for_period,
)
from app.llm.query_ir import QueryIR, Filter

# Derived from the single hierarchy mapping (app/llm/hierarchy.py) instead
# of hardcoding one dict entry per level here — team/company keep their
# original entries (now sourced the same way), unit_head/zonal_head/
# business_center are included for free.
_LEVEL_GROUP_COLUMN = {level: hierarchy.column_for(level) for level in hierarchy.GROUP_LEVELS}

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


# The person-holding levels, senior first — hierarchy.CHAIN without
# `team`, whose values are group names rather than people. Same
# derivation as nlu_pipeline._ROLE_LEVELS, so there is still exactly one
# statement of the ranking in the codebase.
_ROLE_LEVELS = [lvl for lvl in hierarchy.CHAIN if lvl != "team"]


def _exclude_more_senior_roles(query, level: str):
    """Keep only the people whose HIGHEST role is this one.

    A role level is a column of names, and one person appears in several
    of them: a Unit Head is named in `rm` by his 75 advisors and in
    `management_lead` by the handful directly beneath him. So "all BCMs"
    listed 181 people of whom 87 are really Zonal Heads or Unit Heads,
    and the same person was counted at two levels of the same answer.

    The rule is the hierarchy's own: someone belongs at the senior-most
    level they hold. Expressed here as "not named in any column above
    this one", which is the same statement read from the columns rather
    than from a per-person lookup — no traversal, no second ranking, and
    nothing to keep in sync when CHAIN is rebound.

    ADVISOR IS DEDUCED, NOT FILTERED. Excluding every manager from
    `advisor` would silently drop 181 people from "all advisors" and from
    every advisor leaderboard — the metric answers this must not disturb.
    An advisor row is a PERSON rather than a role column, so the leaf
    level keeps everyone and only the manager levels dedupe.
    """
    if level not in _ROLE_LEVELS or level == "advisor":
        return query
    seniors = _ROLE_LEVELS[:_ROLE_LEVELS.index(level)]
    column = _LEVEL_GROUP_COLUMN.get(level)
    if column is None or not seniors:
        return query

    for senior in seniors:
        senior_column = _LEVEL_GROUP_COLUMN.get(senior)
        if senior_column is None:
            continue
        holders = (
            select(senior_column)
            .where(senior_column.isnot(None),
                   Advisor.in_master_sheet.is_(True))
        )
        query = query.filter(~column.in_(holders))
    return query


def _tiebreak(ir: QueryIR):
    """A deterministic second sort key, so paging can't repeat or skip.

    `ORDER BY value DESC` alone leaves rows with EQUAL values in whatever
    order the database happens to produce, and it is free to choose
    differently per query. 136 advisors here tie at 0 connects, so a
    LIMIT/OFFSET walk across that block returned some people twice and
    others never: 573 rows came back carrying only 570 distinct people.

    Invisible while every answer was one capped page — nobody paged past
    the first 10. It became reachable the moment "connects of all BCMs"
    started paging through the whole list, and a list that silently drops
    people is not the list that was asked for.

    The key is the group's own identity — the advisor row at advisor
    level, the grouping column above it — which is unique per output row
    by construction, so it totally orders every tie without changing
    which rows come back or how they rank.
    """
    level = ir.subject_level
    if level == "advisor":
        return Advisor.wid
    return _LEVEL_GROUP_COLUMN.get(level)


def resolve_metric_for_period(metric_key: str, period=None) -> str | None:
    """THE authority for "which metric key answers this measure at this
    period" (Phase 2).

    Period used to be resolved in two incompatible ways: the metric key
    encoded it (mtd_cleared vs ytd_cleared) and ir_patcher owned a
    hardcoded six-entry swap table covering only the cleared family. Any
    other metric kept its MTD binding while the IR recorded the requested
    period — so "top 5 by connects" followed by "what about YTD" returned
    MTD numbers labelled YTD, invisibly.

    Returning None means the measure has NO data at that period, which is
    the truthful answer for every SalesFunnel-sourced metric (those
    columns are MTD-only). Callers must degrade, not substitute.
    """
    return metric_for_period(metric_key, period)


def effective_metric(ir) -> str | None:
    """Public alias for _effective_metric.

    Everything that NAMES the metric in a reply must resolve it the same
    way the value was computed. Once a stated period could reach the
    compiler (Step 2), reading the raw `ir.sort.metric` for a label
    produced a YTD number under the header "MTD Revenue Cleared" — the
    right figure with the wrong name on it, which is its own kind of
    silently wrong answer.
    """
    return _effective_metric(ir)


def _effective_metric(ir) -> str | None:
    """The metric key this IR should actually be compiled against.

    THE one place (metric, period) becomes a binding. The IR carries the
    measure in `sort.metric`/`metric.key` and the period in
    `time_range.period`; only reading the first is how a YTD request
    returned MTD numbers. Both are read here, and a request the measure
    cannot answer returns None — which `compile_and_run` already turns
    into the caller's "I don't have a way to answer that yet" path.

    Consistency note: plan_to_ir sets `time_range.period` from the chosen
    metric, so a rule-built IR always agrees with itself and this is a
    no-op for it. It bites only on an LLM-built IR that names a measure
    and a period that don't match — exactly the case worth catching.
    """
    metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
    if not metric_key:
        return None
    period = getattr(ir.time_range, "period", None)
    return resolve_metric_for_period(metric_key, period)


def default_direction(metric_key: str | None) -> str:
    """"asc" | "desc" — the sort a ranking should use when the user
    didn't say. Read from the metric's own `lower_is_better`, so a
    metric where less is better (overdue, late arrivals) can no longer
    be ranked worst-first and presented as the leaders."""
    return "asc" if lower_is_better(metric_key) else "desc"


def _binding_for(metric_key: str, level: str) -> ColumnBinding | None:
    """Explicit ontology bindings win when they exist — team/company keep
    their exact prior behavior (including team-named TeamTarget bindings
    that have no advisor-rooted equivalent) unchanged.

    For the three NEW hierarchy levels (unit_head/zonal_head/business_
    center), metric_ontology.py declares no per-metric binding at all —
    requiring one would mean duplicating every metric's advisor-level
    binding ~3x, exactly the per-level hardcoding requirement 4 says to
    avoid. Since these levels are plain Advisor-column rollups (never
    team-named — there's no per-unit-head source table the way TeamTarget
    is a genuine team-level source), any such level generically reuses the
    metric's advisor-level binding: `_value_expr` already knows how to
    sum/avg it for a rollup, and `_LEVEL_GROUP_COLUMN` already knows which
    Advisor column to group by. Company is deliberately NOT included in
    this fallback — its existing binding-or-None behavior (some metrics
    were never given a company binding) stays exactly as it was.

    PHASE 4. That advisor-binding fallback was the right rule scoped too
    narrowly: it applied to the three NEW levels only, so `team` still
    preferred its team-named TeamTarget binding while unit_head/zonal_
    head/bcm rolled advisors up. That is one of the ways a single team's
    achievement had two answers. Binding selection now comes from the
    aggregation engine, which applies the fallback at EVERY group level.

    The TeamTarget figure is not lost — it answers a different question
    (the sheet's own team target, not a roll-up of that team's advisors),
    and team_service reads it explicitly under its own keys."""
    metric = METRICS.get(metric_key)
    if not metric:
        return None
    if level == "advisor":
        return metric.bindings.get("advisor")
    return aggregation.binding_for(metric_key, level)


def is_answerable(metric_key: str, level: str) -> bool:
    """Public predicate: can the compiler actually build a query for this
    (metric, level) pair — an explicit ontology binding, or the generic
    new-hierarchy-level rollup fallback in `_binding_for` above. Used by
    ir_validator.py so it never resets a subject_level the compiler would
    in fact have honored (that used to only check membership in
    METRICS[key].bindings directly, which doesn't know about the fallback)."""
    return _binding_for(metric_key, level) is not None


def compile_and_run(db: Session, ir: QueryIR, offset: int = 0) -> list[dict] | None:
    """Returns a list of {wid, name, team, company, value, ...} rows, or
    None if the IR (after validation) still can't be answered — e.g. the
    sort metric has no binding at the requested level. `offset` (Part 8's
    pagination) skips this many rows past the start of the ordered
    result — default 0 preserves the exact prior behavior for every
    existing caller."""
    level = ir.subject_level
    sort_metric_key = _effective_metric(ir)
    if not sort_metric_key:
        return None

    binding = _binding_for(sort_metric_key, level)
    if not binding:
        return None

    if binding.team_named:
        return _run_team_named(db, ir, binding, offset)
    return _run_advisor_rooted(db, ir, binding, sort_metric_key, offset)


def count_ir(db: Session, ir: QueryIR) -> int | None:
    """Total rows the IR's filters would match, ignoring limit/offset —
    same unanswerable conditions (None) as compile_and_run, since this is
    meant to be called as the answerability gate before it (Part 8:
    pagination needs the true total to decide whether/how many pages
    exist, decoupled from however many rows get displayed)."""
    level = ir.subject_level
    sort_metric_key = _effective_metric(ir)
    if not sort_metric_key:
        return None

    binding = _binding_for(sort_metric_key, level)
    if not binding:
        return None

    if binding.team_named:
        query, _value_col = _build_team_named_query(db, ir, binding)
    else:
        query, _value_expr = _build_advisor_rooted_query(db, ir, binding)

    return db.query(func.count()).select_from(query.subquery()).scalar()


def _build_team_named_query(db, ir: QueryIR, binding: ColumnBinding):
    """Filtered-but-unordered/unlimited query for TeamTarget-style metrics
    — shared by _run_team_named (adds order_by/limit/offset) and
    count_ir (wraps in a count) so the filter logic lives in one place."""
    model = binding.model
    value_col = binding.expr

    query = db.query(model.team.label("name"), value_col.label("value"))

    for f in ir.filters:
        if f.field == "team":
            query = query.filter(model.team.ilike(f"%{f.value}%"))

    subject_names = [s.resolved_id or s.value for s in ir.subjects if s.type == "team"]
    if subject_names:
        query = query.filter(model.team.in_(subject_names))

    return query, value_col


def _run_team_named(db, ir: QueryIR, binding: ColumnBinding, offset: int = 0) -> list[dict] | None:
    """TeamTarget-style metrics: the row already IS the team, no Advisor
    join, no roll-up. Filters here are limited to team-name membership
    (e.g. comparisons) since there's no per-advisor row to join other
    metrics' fact tables against."""
    query, value_col = _build_team_named_query(db, ir, binding)

    # Same tie-determinism as the advisor-rooted path below; the team
    # name is this row's identity here.
    query = query.order_by(_order(value_col, ir.sort.direction), binding.model.team)
    if ir.limit:
        query = query.limit(ir.limit)
    if offset:
        query = query.offset(offset)

    rows = query.all()
    return [{"wid": None, "name": r.name, "team": r.name, "company": None, "value": r.value} for r in rows]


def _apply_entity_filters(query, ir: QueryIR):
    """Any filter field that names a hierarchy level (team/company/advisor/
    unit_head/zonal_head/business_center) filters Advisor by that level's
    column, generically via hierarchy.LEVEL_COLUMNS — metric filters and
    attendance_status are handled by their own dedicated functions below
    and simply aren't in that mapping, so they're a no-op here."""
    for f in ir.filters:
        if hierarchy.column_for(f.field) is None:
            continue
        # Scope through hierarchy.scope_filter, which its own docstring
        # calls "THE one definition of 'in scope'... so a query cannot
        # scope one way in a leaderboard and another in a comparison".
        # This function was the exception that made that untrue: it built
        # its own predicate, `column.ilike(f"%{value}%")`, a SUBSTRING
        # match where every other layer matches the whole name.
        #
        # With a team called "Blue Area" and a sibling "Blue Area North",
        # "Blue Area revenue" compiled to
        #     WHERE team LIKE '%Blue Area%' GROUP BY team
        # which partitions into TWO groups. The ranking operators the
        # group-metric compilation relies on being no-ops then stop being
        # no-ops: ORDER BY sorts the two, and the reply answered
        # "Blue Area North has 10,000 ... ranking 1st of 2 teams" for a
        # question about Blue Area's 2,750.
        #
        # The IR says `operator="="`. Compiling an equality as containment
        # was also a contract break between the IR and the compiler:
        # _apply_subject_filter, ten lines below, already matches exactly
        # via column.in_(names), which is why comparisons were unaffected.
        #
        # Filter values are canonical: entity extraction grounds them
        # against the gazetteer before they ever reach a filter, so a
        # partial name the user typed is already expanded here and needs
        # no wildcard. scope_filter's ilike keeps the match
        # case-insensitive.
        query = query.filter(hierarchy.scope_filter(f.field, f.value))
    return query


def _apply_subject_filter(query, ir: QueryIR):
    """Any grounded subject scopes the query down to it — not just for
    intent=="comparison" (comparisons compile as a normal query with an
    added 'subject name is one of these' filter, no separate code path
    needed), but for ANY intent. Bug fix: a "leaderboard"-shaped question
    about one specific named entity ("show me unit head X's performance")
    has the parser correctly ground X into ir.subjects while keeping
    intent="leaderboard" (it isn't comparing two things) — gating this
    filter to comparison-only meant that subject was silently dropped and
    the query ran as an unfiltered top-N ranking of everyone instead of
    the one row asked about. A single grounded subject is exactly as real
    a scoping signal as an explicit team/company/... filter; response_
    planner already collapses a single resulting row to shape=
    "single_value" once this actually filters it down to one. The
    team-named path below (_build_team_named_query) already applied its
    subject filter unconditionally — this brings the advisor-rooted path
    in line with that instead of the other way around.

    Phase 1 identity refactor: for advisor subjects carrying a resolved
    WID, the filter binds to Advisor.wid instead of Advisor.name. Name
    equality cannot address one specific person — 238 name groups in
    production map to more than one human, so `Advisor.name.in_(["Yasir
    Ali"])` silently matches 8 different people and sums their numbers
    into one row. WIDs are used only when EVERY advisor subject has one;
    a partially-resolved set falls back to name matching rather than
    silently dropping the unresolved subjects out of a comparison."""
    if not ir.subjects:
        return query
    column = hierarchy.column_for(ir.subject_level)
    if column is None:
        return query

    subjects = [s for s in ir.subjects if s.type == ir.subject_level]
    if not subjects:
        return query

    if ir.subject_level == "advisor":
        wids = [s.resolved_wid for s in subjects if s.resolved_wid is not None]
        if wids and len(wids) == len(subjects):
            return query.filter(Advisor.wid.in_(wids))

    names = [s.resolved_id or s.value for s in subjects]
    if names:
        query = query.filter(column.in_(names))
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
    # Advisor IS the query root, so "joining" it is a no-op. A metric
    # whose column lives on the advisor row itself (1-Unit ownership)
    # binds to Advisor; without this guard the compiler would emit
    # `FROM advisors JOIN advisors ON advisors.wid = advisors.wid`.
    if model is Advisor:
        return query, Advisor

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
    by mtd_cleared). Each distinct (model, period) fact table is joined once.

    A filter field is taken LITERALLY and is deliberately not re-resolved
    against ir.time_range. "Rank by MTD revenue, but only advisors whose
    YTD revenue exceeds 1500" is a real query, and a filter naming a
    period-specific metric is already saying which window it means —
    forcing it to the IR's period would silently rewrite it (see
    test_filter_on_different_period_than_sort_metric_binds_to_its_own_period).

    A threshold the user stated without its own period is a different
    thing, and is resolved where that distinction is still visible: at
    IR construction, in query_ir._threshold_filters().
    """
    for f in ir.filters:
        if f.field not in METRICS:
            continue
        f_binding = _binding_for(f.field, level)
        if not f_binding or f_binding.team_named:
            continue  # can't combine a team-named metric filter into an advisor-rooted query
        query, entity = _join_fact_table(query, joined, f_binding.model, f_binding.period)
        query = _join_declared_models(query, joined, f_binding)

        # THE METRIC'S VALUE, not its raw column. This read
        # `f_binding.expr`, which for a RATIO is the NUMERATOR ALONE — so
        # "answered calls % above 60" compiled to
        # `answered_calls * 100 > 60`, i.e. `answered_calls > 0.6`, and
        # returned all 404 advisors who answered any call where 185
        # actually clear 60%. The denominator was not mis-scaled; it was
        # absent. Asking the aggregation engine for the expression is the
        # same call the SORT metric makes, so a threshold and a ranking on
        # one measure can no longer disagree about what it is.
        expr = aggregation.value_expression(
            f_binding, f.field, level,
            numerator=_rebind_to_entity(f_binding.ratio_numerator, entity, f_binding.model)
            if f_binding.ratio_numerator is not None else None,
            denominator=_rebind_to_entity(f_binding.ratio_denominator, entity, f_binding.model)
            if f_binding.ratio_denominator is not None else None,
            expr=_rebind_to_entity(f_binding.expr, entity, f_binding.model),
        )
        predicate = _apply_comparator(expr, f.operator, f.value)

        # AND IT IS A CONDITION ON THE GROUP, so above the leaf it belongs
        # AFTER the grouping. As a WHERE it selected individual advisor
        # rows and the group was then aggregated over only those — a BCM
        # qualified because one of her people did, and the figure shown
        # was the partial sum: "BCMs with answered calls below 60%"
        # answered with a BCM at 283, and a leaderboard row read 102 where
        # the aggregation engine said 68 for the same person.
        if level == "advisor":
            query = query.filter(predicate)
        else:
            query = query.having(predicate)
    return query


def _join_declared_models(query, joined: dict, binding: ColumnBinding):
    """JOIN the extra tables a binding's expression references.

    FIX 3. A ratio may span two fact tables (Connect->CR divides client
    registrations by answered calls). Referencing the second table
    without joining it does not raise — SQLAlchemy appends it to the FROM
    clause, producing a cartesian product that silently inflates the
    denominator. `join_models` declares the requirement on the binding;
    this honours it, reusing the same `joined` registry so a table is
    never joined twice.
    """
    for model in getattr(binding, "join_models", ()) or ():
        query, _entity = _join_fact_table(query, joined, model, None)
    return query


def _build_advisor_rooted_query(db, ir: QueryIR, binding: ColumnBinding):
    """Filtered/joined/grouped-but-unordered/unlimited query, rooted at
    Advisor — shared by _run_advisor_rooted (adds order_by/limit/offset)
    and count_ir (wraps in a count) so the join/filter logic lives in one
    place. Returns (query, value_expr) — value_expr is needed by the
    caller to order by (the raw per-advisor column at the leaf, or the
    engine's roll-up expression at a group level)."""
    level = ir.subject_level
    joined: dict = {}
    # PHASE 4: the compiler no longer decides whether or how to roll up.
    # It used to compute `is_rollup = level != "advisor"` and pick sum()
    # or avg() itself — a second copy of a rule that also lived in
    # comparison_service, which is how ranking teams by achievement and
    # comparing the same two teams gave different numbers. The engine
    # returns the raw column at the leaf and the declared roll-up above
    # it. The metric comes from the IR via the same resolver the rest of
    # the compiler uses, so a period-resolved sibling reads its own rule.
    value_expr = aggregation.value_expression(
        binding, _effective_metric(ir), level,
    )

    if level == "advisor":
        query = db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company)
        query, _sort_entity = _join_fact_table(query, joined, binding.model, binding.period)
        query = _join_declared_models(query, joined, binding)
        query = query.add_columns(value_expr.label("value"))
    else:
        group_col = _LEVEL_GROUP_COLUMN[level]
        query = db.query(group_col.label("name"))
        query, _sort_entity = _join_fact_table(query, joined, binding.model, binding.period)
        # The group path needs the declared joins as much as the leaf
        # does. A cross join happens to leave a pure RATIO unchanged (both
        # sums scale by the same factor), so the number looks right — but
        # the SQL is wrong, it is quadratic, and any filter or non-ratio
        # use of the same binding would be wrong too.
        query = _join_declared_models(query, joined, binding)
        query = query.add_columns(value_expr.label("value"))
        query = query.filter(group_col.isnot(None))

    # every advisor-rooted query (advisor level, or a team/company rollup
    # over Advisor) excludes WIDs that only appear in a raw source sheet
    # and were never actually on the MasterSheet — see models.py's
    # in_master_sheet column docstring.
    query = query.filter(Advisor.in_master_sheet.is_(True))

    query = _apply_entity_filters(query, ir)
    query = _apply_subject_filter(query, ir)
    query = _apply_attendance_filter(query, ir, joined)
    query = _apply_metric_filters(query, ir, level, joined)

    query = _exclude_more_senior_roles(query, level)

    if level != "advisor":
        query = query.group_by(_LEVEL_GROUP_COLUMN[level])

    return query, value_expr


def _run_advisor_rooted(db, ir: QueryIR, binding: ColumnBinding, sort_metric_key: str, offset: int = 0) -> list[dict] | None:
    level = ir.subject_level
    query, value_expr = _build_advisor_rooted_query(db, ir, binding)

    # value_expr is already the right expression to order by: the raw
    # per-advisor column at advisor level, or the sum()/avg()-wrapped
    # rollup at team/company level (from _value_expr above).
    tiebreak = _tiebreak(ir)
    query = query.order_by(_order(value_expr, ir.sort.direction),
                           *( [tiebreak] if tiebreak is not None else [] ))
    if ir.limit:
        query = query.limit(ir.limit)
    if offset:
        query = query.offset(offset)

    rows = query.all()

    if level == "advisor":
        return [dict(r._mapping) for r in rows]
    return [{"wid": None, "name": r.name, "team": r.name if level == "team" else None,
              "company": r.name if level == "company" else None, "value": r.value} for r in rows]
