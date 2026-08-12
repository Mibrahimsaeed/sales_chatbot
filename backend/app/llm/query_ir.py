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

Part 10 (confidence-aware generation) added three fields beyond the
per-field confidences that already existed (metric.confidence,
filters[].confidence, subjects[].match_confidence, time_range.confidence):
  - intent_confidence     — how sure the parser is about intent/shape
                            itself, independent of any one field's value
  - ambiguity_reasons     — human-readable reasons, POPULATED BY
                            ir_validator.py during grounding, not by the
                            LLM — same "validator is the safety layer,
                            not the parser" split `missing[]` already uses
  - confidence_level      — "high" | "medium" | "low", also populated by
                            ir_validator.py; see its module docstring for
                            what each tier means for execution
All three default to values that make an IR built the OLD way (rule-based
plan_to_ir, ir_patcher, or any hand-built QueryIR that never sets them)
behave exactly as it did before this field existed — this is additive,
not a breaking change to the model.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from app.llm import hierarchy, periods

# Step 3: DERIVED from the hierarchy registry, not written out here.
#
# This was a hand-maintained Literal listing the levels a second time.
# It happened to be correct while llm_client's schema enum — built from
# the same registry — had silently narrowed to the chain, so the two
# disagreed about company/office/region and nothing noticed: the only
# test compared them with `<=`, which a narrowing satisfies.
#
# The accepted set is every addressable level plus the legacy aliases, so
# a stored QueryIR or an older API client still validates;
# hierarchy.canonical_level maps an alias to its current name wherever it
# is used. Sorted for a stable, diffable order.
#
# Literal[tuple] is the runtime spelling of Literal[a, b, c] — pydantic
# builds the same validator from it.
LEVEL_NAMES: tuple[str, ...] = tuple(
    sorted(set(hierarchy.HIERARCHY_LEVELS) | set(hierarchy.LEVEL_ALIASES))
)
Level = Literal[LEVEL_NAMES]
Operator = Literal["=", "!=", ">", ">=", "<", "<=", "in"]
Intent = Literal["leaderboard", "comparison", "lookup", "trend", "filtered_list", "breakdown", "clarify"]
ConfidenceLevel = Literal["high", "medium", "low"]


class Subject(BaseModel):
    type: Level
    value: str
    resolved_id: Optional[str] = None
    # Phase 1 identity refactor: for type=="advisor", the resolved primary
    # key. When populated, query_compiler filters on Advisor.wid instead of
    # Advisor.name — the only way to address one specific person when 238
    # name groups in production map to more than one human. Stays None for
    # non-advisor levels (a team/company IS its name) and for any advisor
    # subject the LLM produced that grounding could not resolve to exactly
    # one person, in which case name matching remains the fallback.
    resolved_wid: Optional[int] = None
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
    # DERIVED from app/llm/periods.py, not restated. The IR must be
    # able to CARRY every period the parser can recognise, including ones
    # no metric can answer yet — "revenue today" has to reach the
    # compiler as DAILY so it can be refused honestly. Restating the list
    # here would make "today" a pydantic ValidationError instead, which
    # degrades to the same silent MTD default this replaces.
    period: Literal[periods.PERIODS] = "MTD"
    compare_to: Optional[str] = None             # e.g. previous period key — Phase 4, not compiled yet
    confidence: float = 1.0                      # how sure the parser is this is the intended period


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
    # intent=="breakdown" only: nested-by-team (default False) vs a flat
    # advisor list for the single named subject — see hierarchy_service.
    # get_level_breakdown / get_level_flat_list. Ignored by every other
    # intent, defaults to False so every existing caller (rule-based
    # plan_to_ir, ir_patcher, hand-built IRs, ir_examples.py from before
    # this field existed) is unaffected.
    flat: bool = False
    overall_confidence: float = 1.0
    intent_confidence: float = 1.0
    missing: list[str] = Field(default_factory=list)
    # both populated by ir_validator.validate_ir(), not by the LLM —
    # human-readable version of `missing[]`, and the three-tier execution
    # gate derived from it plus overall_confidence (Part 10)
    ambiguity_reasons: list[str] = Field(default_factory=list)
    confidence_level: Optional[ConfidenceLevel] = None
    # observability only (persisted in ChatLog.resolved_ir): which NLU mode
    # served this IR — not part of the LLM output schema, never validated
    nlu_mode: Optional[str] = None


def _period_of(metric_key: str | None) -> str:
    """The period the chosen metric reports, as the IR's string enum.

    The FALLBACK only — see _period_for() below. Used when the user named
    no window at all, which is the majority of queries and the behaviour
    every existing caller depends on.
    """
    from app.llm.metric_ontology import METRICS

    metric = METRICS.get(metric_key) if metric_key else None
    return metric.period.value if metric else "MTD"


def _period_for(plan) -> str:
    """The period this IR should carry: what the USER said, falling back
    to the metric's own.

    F4. This used to be `_period_of(plan.metric)` unconditionally, so the
    measure decided the window and "revenue year to date" compiled as
    MTD — the user's words overwritten by a property of the key that
    happened to match "revenue". The stated period now wins, and
    query_compiler._effective_metric() resolves the (measure, period)
    pair, swapping mtd_cleared for ytd_cleared. That resolver already
    existed; nothing was ever passing it the period.

    Deliberately NOT resolved here. Keeping the IR as "the measure the
    user named plus the window the user named" leaves _effective_metric
    the single place the pair becomes a binding — resolving it in two
    places is what Phase 2 removed.
    """
    return getattr(plan, "period", None) or _period_of(plan.metric)


def _threshold_filters(plan) -> list["Filter"]:
    """Comparator/value pairs from extraction, as filters on the metric.

    F8. These had nowhere to live on QueryPlan, so "achievement above 80
    percent" produced an IR with no filters and the reply listed
    everybody — a superset, which reads as authoritative and is simply
    wrong.

    A threshold binds to `plan.metric` because that is the measure the
    sentence named ("advisors with ACHIEVEMENT above 80 percent"). With
    no metric there is nothing to compare against, and inventing a field
    would turn an unanswerable query into a confident wrong answer, so
    the thresholds are dropped rather than guessed at.

    The field is resolved for the plan's period HERE, not in the
    compiler. An extracted threshold carries no window of its own — "700"
    in "revenue above 700 this year" means 700 of whatever revenue the
    user asked about — so it must follow the same (measure, period)
    resolution the sort metric gets, or the query ranks by ytd_cleared
    while filtering mtd_cleared.

    The compiler cannot make this call: by then a filter on `ytd_cleared`
    is indistinguishable from one the user made period-specific on
    purpose ("rank by MTD revenue, but only advisors above 1500 YTD"),
    and re-resolving there would rewrite that deliberate query. This is
    the last point where the difference is still known.
    """
    metric_key = plan.metric
    if not metric_key:
        return []

    from app.llm.query_compiler import resolve_metric_for_period

    # `or metric_key`: a measure with no sibling for that period keeps
    # its own field rather than losing the filter. The sort metric will
    # independently fail to resolve and the query returns "can't answer",
    # so this cannot produce a wrong answer on its own.
    field = resolve_metric_for_period(metric_key, _period_for(plan)) or metric_key

    return [
        Filter(field=field, operator=t["operator"], value=t["value"])
        for t in getattr(plan, "thresholds", None) or []
        if t.get("operator") and t.get("value") is not None
    ]


def _direction_for(plan) -> str:
    from app.llm.query_compiler import default_direction

    if plan.ascending is None:
        return default_direction(plan.metric)
    return "asc" if plan.ascending else "desc"


# Fields that name WHO a query is about. On a comparison these are
# carried by `subjects` instead, and leaving them as filters would
# intersect the two sides — "Blue Area AND Downtown" matches nobody.
#
# DERIVED from the hierarchy rather than listed: a level added there must
# not need a second edit here to be excluded, which is the drift
# test_hierarchy_single_source.py exists to catch (and did catch, on the
# hardcoded first version of this).
_SUBJECT_FIELDS = frozenset(hierarchy.HIERARCHY_LEVELS) - {"advisor"}


def plan_to_ir(plan, entities: dict) -> QueryIR:
    """Fail-soft degrade path (Part 5.1 error handling): wraps the existing
    rule-based query_planner.QueryPlan into a minimal, single-metric,
    single-filter QueryIR. Used both as the normal fast path for simple
    leaderboard queries (skip the LLM call entirely when the rule-based
    planner already resolved it and the text doesn't look compound) and as
    the degrade target when the LLM call fails or returns invalid JSON.
    """
    from app.llm.hierarchy import NEW_GROUP_LEVELS

    filters: list[Filter] = []
    if entities.get("team"):
        filters.append(Filter(field="team", operator="=", value=entities["team"]))
    if entities.get("company"):
        filters.append(Filter(field="company", operator="=", value=entities["company"]))
    for level in NEW_GROUP_LEVELS:
        if entities.get(level):
            filters.append(Filter(field=level, operator="=", value=entities[level]))
    if entities.get("attendance_status"):
        filters.append(Filter(field="attendance_status", operator="=", value=entities["attendance_status"]))
    # F8: the comparators the user stated, on the metric they named.
    # Appended after the entity filters so the AND-combination reads in
    # the order the sentence did ("advisors in Blue Area above 80%").
    filters.extend(_threshold_filters(plan))

    # Phase 5B: a comparison becomes a comparison IR, not a leaderboard.
    #
    # Comparison used to execute on the rule-based PLAN path, through
    # comparison_service — a second pipeline that bypassed QueryIR and
    # therefore bypassed everything QueryIR owns: _effective_metric (so
    # "compare … year to date" resolved YTD and executed MTD),
    # conversation memory (so the next turn lost both subjects),
    # ir_validator, and the response planner.
    #
    # Nothing needed building to fix that. The IR path ALREADY supported
    # comparison end to end — ir_validator checks for >= 2 subjects,
    # query_compiler compiles subjects into a scoped query,
    # response_planner has the comparison mode, and
    # format_ir_comparison_reply renders it. The rule planner simply
    # never routed there. This is the whole integration.
    subjects: list[Subject] = []
    if plan.action == "comparison" and plan.comparison_targets:
        subjects = [
            Subject(type=level, value=value, match_confidence=1.0)
            for level, value in plan.comparison_targets
        ]

    if subjects:
        # A comparison's subjects ARE its scope. Entity filters would
        # narrow to the intersection of both sides and return nothing.
        filters = [f for f in filters if f.field not in _SUBJECT_FIELDS]

    return QueryIR(
        intent="comparison" if subjects else "leaderboard",
        subjects=subjects,
        # The subjects carry their own levels, so a mixed-level
        # comparison ("Waqar Haider vs his team") keeps both. This is the
        # level the ANSWER is grouped at; the first subject's level is
        # the right default because comparison_targets() orders the
        # leading subject first.
        subject_level=(subjects[0].type if subjects else (plan.level or "advisor")),
        metric=MetricRef(key=plan.metric) if plan.metric else None,
        filters=filters,
        # Phase 2: the user's explicit direction wins; when they named
        # none (plan.ascending is None) the metric's own polarity decides,
        # so a lower-is-better metric cannot be ranked worst-first.
        sort=Sort(metric=plan.metric, direction=_direction_for(plan)),
        # Phase 2 kept the IR self-consistent by taking the period FROM
        # the metric. Step 2 (F4) corrects the direction of that: the
        # user's stated window wins, and the metric's own period is the
        # fallback for when they stated none. See _period_for().
        time_range=TimeRange(period=_period_for(plan)),
        # `or 10` here re-imposed the cap the planner had just lifted:
        # a query saying ALL reached this with limit=None and left it
        # as 10. The plan's own default is 10 (query_planner.
        # _DEFAULT_LIMIT), so an unset limit already arrives as 10 and
        # None now means exactly what it says — no cap, page through
        # the true match count.
        limit=plan.limit,
        # a deterministic rule-based match is genuinely high-confidence —
        # not 1.0 (an LLM-confirmed shape can still be more certain, e.g.
        # matching an explicit business-phrase gloss), but well clear of
        # the confidence_high_threshold gate in ir_validator.py (Part 10)
        overall_confidence=0.9,
    )
