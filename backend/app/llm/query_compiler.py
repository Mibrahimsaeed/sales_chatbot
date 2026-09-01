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

from sqlalchemy import and_, asc, desc, func, not_, or_, select
from sqlalchemy.orm import Session, aliased

from app.database.models import Advisor, Attendance, PerformancePeriod
from app.llm import aggregation, hierarchy
from app.llm.metric_ontology import (
    METRICS, ColumnBinding, lower_is_better, metric_for_period,
)
from app.llm.query_ir import QueryIR, Filter, FilterGroup

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
    level = ir.grouping_level()
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
    # THE SAME READING the validator uses (QueryIR.primary_metric), so
    # "this query has a measure" and "this is the measure to compute"
    # can never disagree. Reading `sort`/`metric` alone returned None for
    # a filter-only query — the compiler then answered "no data" for a
    # question whose measure was stated plainly in the filter.
    #
    # It returns None for a population, which is correct and load-bearing:
    # that operation asks WHO, so there is no value column to compute and
    # the rows come back without one.
    metric_key = ir.primary_metric()
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
    # A POPULATION needs no measure — see _is_population_query. Checked
    # before the metric requirement below, which would otherwise refuse
    # it for lacking something it deliberately does not have.
    if _is_population_query(ir):
        return _run_population(db, ir, offset)

    level = ir.grouping_level()
    sort_metric_key = _effective_metric(ir)
    if not sort_metric_key:
        return None

    binding = _binding_for(sort_metric_key, level)
    if not binding:
        return None

    if binding.team_named:
        return _run_team_named(db, ir, binding, offset)
    return _run_advisor_rooted(db, ir, binding, sort_metric_key, offset)


def _is_population_query(ir: QueryIR) -> bool:
    """Does this IR ask WHO, with no measure to rank by?

    "list the advisors excluding Blue Area" is a population, and forcing
    a metric onto it changes the answer: every metric is reached through
    its fact table, and that join drops the rows with no record in it —
    13 advisors here have no `calls` row, so a connects-ranked
    "population" returned 507 of 518.

    So a population is expressed by the ABSENCE of a metric rather than
    by a placeholder one. `operation="population"` says the absence is
    deliberate, which is what separates it from an IR that simply failed
    to resolve a measure — that one must still be refused.
    """
    if ir.is_hierarchy_read():
        # A hierarchy read is a population WHEN IT NAMES NO MEASURE, and
        # only then. "who reports to X" enumerates people with nothing to
        # rank them by, and belongs here. "connects of advisors under X"
        # names one — and this branch used to return True regardless, so
        # the measure was dropped on the way to a metric-free query and
        # every row came back with value=None. The header read "Total
        # Connects" and the cell read "—", while the two companion
        # columns, which are re-read per row rather than taken from the
        # row's own value, showed real numbers. Six of six phrasings
        # across connects, meetings, revenue and rates lost the figure,
        # the ranked form included: "top advisors under X by connects"
        # was ranked by nothing.
        #
        # The rule this restores is the one stated above: a population is
        # the ABSENCE of a measure. `primary_metric()` is the single
        # reading of whether one is present, shared with the validator
        # and `_effective_metric`, so the three cannot disagree.
        return ir.primary_metric() is None
    return ir.resolved_operation() == "population"


def _hierarchy_subject(db, ir: QueryIR):
    """The person (or group) the target level sits beneath.

    Two ways a subject reaches here, and the IR says which:

      named outright   subjects[0].type == subject_of, so the named
                       value IS the manager.
      named by ROLE    subjects[0] is a SCOPE and `subject_of` is the
        WITHIN A SCOPE role inside it ("the Unit Head in AMD"). The
                       holder is read out of the scope with
                       hierarchy_service.get_manager_of_group — the same
                       resolver reverse lookups already use, rather than
                       a second way of asking who holds a role.

    Returns (level, value) or None when nothing can be resolved.
    """
    if not ir.subjects:
        return None
    subject = ir.subjects[0]
    manager_level = ir.subject_of or subject.type

    if subject.type == manager_level:
        return manager_level, (subject.resolved_id or subject.value)

    from app.services import hierarchy_service

    holder = hierarchy_service.get_manager_of_group(
        db, subject.type, (subject.resolved_id or subject.value), manager_level
    )
    managers = (holder or {}).get("managers") or []
    # Exactly one holder, or nothing. Several holders is a question for
    # the user, not a row this layer may pick — the dispatcher owns that
    # clarification and this returning None keeps the two from disagreeing.
    if len(managers) != 1:
        return None
    return manager_level, managers[0]


def _apply_hierarchy_scope(db, query, ir: QueryIR):
    """Scope a hierarchy read to the subject, at the requested depth.

    DELEGATES ENTIRELY. `direct_scope_filter` and `subtree_scope_filter`
    already encode what "directly beneath" and "anywhere beneath" mean —
    including the self-exclusion that stops a manager counting as one of
    their own reports — so this only chooses between them from
    `ir.relation` and hands over.
    """
    if not ir.is_hierarchy_read():
        return query

    resolved = _hierarchy_subject(db, ir)
    if resolved is None:
        return query
    manager_level, manager_value = resolved

    predicate = (
        hierarchy.direct_scope_filter(manager_level, manager_value, ir.target_level)
        if ir.relation == "direct"
        else hierarchy.subtree_scope_filter(manager_level, manager_value, ir.target_level)
    )
    if predicate is None:
        return query
    return query.filter(predicate)


def _build_population_query(db, ir: QueryIR):
    """Members matching the filters, with no metric and no fact join.

    Every filter helper the metric path uses is reused unchanged, so an
    OR or a NOT means the same thing here as it does in a ranking — the
    only difference is that nothing is joined to rank by.
    """
    level = ir.grouping_level()
    joined: dict = {}

    if level == "advisor":
        query = db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company)
    else:
        group_col = _LEVEL_GROUP_COLUMN[level]
        query = db.query(group_col.label("name")).select_from(Advisor)
        query = query.filter(group_col.isnot(None))

    query = query.filter(Advisor.in_master_sheet.is_(True))
    query = _apply_hierarchy_scope(db, query, ir)
    query = _apply_entity_filters(query, ir)
    query = _apply_subject_filter(query, ir)
    query = _apply_attendance_filter(query, ir, joined)
    query = _apply_metric_filters(query, ir, level, joined)
    query = _apply_filter_tree(query, ir, level, joined)
    query = _exclude_more_senior_roles(query, level)

    if level != "advisor":
        query = query.group_by(_LEVEL_GROUP_COLUMN[level])
    return query, level


def _run_population(db, ir: QueryIR, offset: int = 0) -> list[dict]:
    query, level = _build_population_query(db, ir)

    # Ordered by identity, because there is no value to order by. Still
    # TOTAL, so paging cannot repeat or skip a row.
    query = query.order_by(Advisor.name if level == "advisor"
                           else _LEVEL_GROUP_COLUMN[level])
    if ir.limit:
        query = query.limit(ir.limit)
    if offset:
        query = query.offset(offset)

    rows = query.all()
    if level == "advisor":
        # `value` is None, not 0: there is no measure here, and a zero
        # would render as one.
        return [{**dict(r._mapping), "value": None} for r in rows]
    return [{"wid": None, "name": r.name, "team": None, "company": None,
             "value": None} for r in rows]


def count_ir(db: Session, ir: QueryIR) -> int | None:
    """Total rows the IR's filters would match, ignoring limit/offset —
    same unanswerable conditions (None) as compile_and_run, since this is
    meant to be called as the answerability gate before it (Part 8:
    pagination needs the true total to decide whether/how many pages
    exist, decoupled from however many rows get displayed)."""
    if _is_population_query(ir):
        query, _level = _build_population_query(db, ir)
        return db.query(func.count()).select_from(query.subquery()).scalar()

    level = ir.grouping_level()
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

    # SAME MECHANISM AS THE ADVISOR-ROOTED PATH, within what this table
    # can express. `binding.team_named` means the ROW IS A TEAM: this
    # model carries a team column and nothing else, so `team` here is not
    # a chosen level but the only one the source has. Subjects are still
    # read through _subjects_by_level so the grouping and the OR-within-a-
    # level rule cannot drift from the other path.
    #
    # A subject at any OTHER level cannot be honoured here — there is no
    # company or region column on this row to match, and reaching one
    # would mean joining Advisor, which is a different query shape than
    # this path exists to build. It is left unapplied, as before.
    team_subjects = _subjects_by_level(ir).get("team") or []
    if team_subjects:
        query = query.filter(or_(*(
            model.team.ilike(s.resolved_id or s.value) for s in team_subjects
        )))

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


def _subjects_by_level(ir: QueryIR) -> dict:
    """Grounded subjects grouped by THEIR OWN level.

    THE DEFECT THIS EXISTS FOR. `_apply_subject_filter` used to build its
    predicate from `hierarchy.column_for(ir.subject_level)` — the level
    the answer is REPORTED at — and keep only the subjects whose type
    already equalled it. A subject naming a container therefore matched
    nothing and was dropped in silence: "top teams in <a company> by
    revenue" carries subject_level="team" and a `company` subject, so the
    scope vanished and the query ranked every team in the business. The
    same shape leaked on company->advisor, team->advisor, region->team
    and region->advisor, on every operation that reaches the compiler.

    A subject's level is a property OF THE SUBJECT, not of the reporting
    level, so it is read from `s.type` here. Scoping across levels needs
    no join: every level is a column on the advisor row (see
    hierarchy.scope_filter, "THE one definition of in scope"), which is
    why this is a grouping rather than a traversal.

    COMBINATION RULES, both taken from behaviour that already exists
    rather than invented for this:
      - within one level, OR. Two teams mean either team, which is what
        the previous `column.in_(names)` meant and what a comparison
        needs.
      - across levels, AND — each level appends its own `query.filter`,
        exactly as `_apply_entity_filters` already does for two filters
        on different fields.
    A subject whose type names no hierarchy column is skipped rather than
    guessed at.
    """
    # A COMPARISON'S SUBJECTS ARE ITS SIDES, not a scope around it.
    # "compare <team A> and <team B>" names the two things to set beside
    # each other, so a subject of some OTHER type is not a container to
    # scope into — it is noise, and ignoring it is the behaviour this
    # path has always had (test_comparison_still_requires_exact_subject_
    # type_match pins it deliberately). Scoping it instead would
    # intersect the sides away and answer with nothing.
    #
    # The same split the validator already draws between an operation
    # whose subject IS the answer and one whose subject CONTAINS it.
    subjects = ir.subjects
    if ir.resolved_operation() == "comparison":
        subjects = [s for s in subjects if s.type == ir.subject_level]

    grouped: dict = {}
    for subject in subjects:
        if hierarchy.column_for(subject.type) is not None:
            grouped.setdefault(subject.type, []).append(subject)
    return grouped


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

    grouped = _subjects_by_level(ir)
    if not grouped:
        return query

    for level, subjects in grouped.items():
        # An advisor subject addresses ONE person, and a name cannot:
        # 238 name groups in production map to more than one human, so
        # matching by name sums several people into one row. Unchanged.
        if level == "advisor":
            wids = [s.resolved_wid for s in subjects if s.resolved_wid is not None]
            if wids and len(wids) == len(subjects):
                query = query.filter(Advisor.wid.in_(wids))
                continue

        query = query.filter(or_(*(
            hierarchy.scope_filter(level, s.resolved_id or s.value)
            for s in subjects
        )))
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


class UncompilableFilterTree(Exception):
    """A boolean combination this compiler cannot express in one clause.

    Raised rather than approximated. The only case is a disjunction that
    mixes a ROW predicate with a GROUP AGGREGATE above the leaf: "BCMs in
    Blue Area OR with team size > 5" needs `WHERE team=... OR HAVING
    count(*)>5`, and SQL has no such clause. Approximating it — pushing
    the aggregate into WHERE, or the row test into HAVING — returns a
    plausible, wrong set, which is the failure mode this whole change
    exists to remove.
    """


def _metric_leaf_expression(f: Filter, level: str, joined: dict, query):
    """(query, predicate, is_aggregate) for one metric filter leaf.

    The predicate is built the way _apply_metric_filters builds it — the
    metric's VALUE expression from the aggregation engine, not the raw
    binding column, so a ratio filter compares the ratio rather than its
    numerator. Extracted here so the flat path and the tree path cannot
    compare a measure two different ways.
    """
    f_binding = _binding_for(f.field, level)
    if not f_binding or f_binding.team_named:
        return query, None, False

    query, entity = _join_fact_table(query, joined, f_binding.model, f_binding.period)
    query = _join_declared_models(query, joined, f_binding)

    expr = aggregation.value_expression(
        f_binding, f.field, level,
        numerator=_rebind_to_entity(f_binding.ratio_numerator, entity, f_binding.model)
        if f_binding.ratio_numerator is not None else None,
        denominator=_rebind_to_entity(f_binding.ratio_denominator, entity, f_binding.model)
        if f_binding.ratio_denominator is not None else None,
        expr=_rebind_to_entity(f_binding.expr, entity, f_binding.model),
    )
    # Above the leaf the metric's value is an aggregate over the group, so
    # the predicate belongs in HAVING — the same rule _apply_metric_filters
    # follows, and for the same reason: as a WHERE it selects individual
    # advisor rows and the group is then aggregated over only those.
    return query, _apply_comparator(expr, f.operator, f.value), level != "advisor"


def _leaf_predicate(f: Filter, ir: QueryIR, level: str, joined: dict, query):
    """(query, predicate, is_aggregate) for any filter leaf.

    Dispatches on the same three families the flat path handles in three
    separate functions — a hierarchy level, attendance status, or a
    metric. A leaf naming none of them contributes nothing rather than
    raising: an ungroundable field is the validator's business, and
    silently dropping it here matches what the flat path already does.
    """
    if hierarchy.column_for(f.field) is not None:
        return query, hierarchy.scope_filter(f.field, f.value), False

    if f.field == "attendance_status":
        query, entity = _join_fact_table(query, joined, Attendance, None)
        return query, entity.biometric_status == f.value, False

    if f.field in METRICS:
        return _metric_leaf_expression(f, level, joined, query)

    return query, None, False


def _compile_filter_node(node, ir: QueryIR, level: str, joined: dict, query):
    """(query, predicate, is_aggregate) for a node of the filter tree.

    Recursive over FilterGroup. `is_aggregate` propagates upward so the
    caller knows whether the finished predicate belongs in WHERE or
    HAVING, and so a mixed disjunction can be refused rather than
    silently compiled wrong.
    """
    if isinstance(node, Filter):
        return _leaf_predicate(node, ir, level, joined, query)

    parts = []
    any_aggregate = False
    all_aggregate = True
    for child in node.children:
        query, predicate, is_aggregate = _compile_filter_node(child, ir, level, joined, query)
        if predicate is None:
            # DROPPING IS ONLY SAFE UNDER `and`. There it narrows less
            # than asked, which is what the flat list already does for an
            # ungroundable field. Under `or` it WIDENS the result and
            # under `not` it inverts the meaning, so a leaf that compiles
            # to nothing there makes the whole group unanswerable.
            if node.op == "and":
                continue
            raise UncompilableFilterTree(
                f"a leaf of a {node.op!r} group could not be compiled "
                f"({getattr(child, 'field', 'group')!r}); dropping it would "
                "change what the query means"
            )
        parts.append(predicate)
        any_aggregate = any_aggregate or is_aggregate
        all_aggregate = all_aggregate and is_aggregate

    if not parts:
        return query, None, False

    # A CONJUNCTION can be split between the two clauses, so a mix is
    # fine: the caller applies the row parts as WHERE and the aggregate
    # parts as HAVING. A DISJUNCTION cannot — the whole expression has to
    # live in one clause, and there is no clause that can hold both.
    if node.op in ("or", "not") and any_aggregate and not all_aggregate:
        raise UncompilableFilterTree(
            f"a {node.op!r} group mixes a row filter with a group aggregate at "
            f"level {level!r}; SQL cannot combine WHERE and HAVING in one clause"
        )

    if node.op == "and":
        combined = and_(*parts)
    elif node.op == "or":
        combined = or_(*parts)
    else:
        # `not(A, B)` negates the CONJUNCTION of its children, which is
        # what "excluding X and Y" means.
        combined = not_(and_(*parts))
    return query, combined, any_aggregate


def _apply_filter_tree(query, ir: QueryIR, level: str, joined: dict):
    """AND the IR's filter tree onto `query`, if it has one.

    Applied ALONGSIDE the flat `filters` list rather than instead of it —
    `QueryIR.filter_tree` is documented as a further conjunct — so an IR
    with no tree compiles down exactly the path it always did.
    """
    if ir.filter_tree is None:
        return query

    query, predicate, is_aggregate = _compile_filter_node(
        ir.filter_tree, ir, level, joined, query)
    if predicate is None:
        return query
    return query.having(predicate) if is_aggregate else query.filter(predicate)


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
    level = ir.grouping_level()
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
    # A HIERARCHY READ KEEPS ITS DEPTH WHEN IT CARRIES A MEASURE.
    #
    # This was reached only from `_build_population_query`, so a metric-
    # bearing hierarchy read — which no longer detours through that path
    # — would be scoped by the subject filter alone. That filter is a
    # plain column match, i.e. the whole subtree: it cannot express
    # "directly under", and it does not exclude a manager from their own
    # reports. Both meanings live in `direct_scope_filter` /
    # `subtree_scope_filter`, and `_apply_hierarchy_scope` already picks
    # between them from `ir.relation`, so it is called here rather than
    # restated. Measured without it, "directly under" returned the whole
    # subtree (7 people where 0 are direct reports) and the manager
    # appeared among their own reports.
    #
    # A no-op for every other query: it returns immediately unless
    # `is_hierarchy_read()`.
    query = _apply_hierarchy_scope(db, query, ir)
    query = _apply_attendance_filter(query, ir, joined)
    query = _apply_metric_filters(query, ir, level, joined)
    # P0: the boolean structure, ANDed on after the flat conjuncts above.
    # A no-tree IR returns unchanged from here.
    query = _apply_filter_tree(query, ir, level, joined)

    query = _exclude_more_senior_roles(query, level)

    if level != "advisor":
        query = query.group_by(_LEVEL_GROUP_COLUMN[level])

    return query, value_expr


def _run_advisor_rooted(db, ir: QueryIR, binding: ColumnBinding, sort_metric_key: str, offset: int = 0) -> list[dict] | None:
    level = ir.grouping_level()
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
