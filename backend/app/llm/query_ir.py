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
    # P0: the measure THIS subject was asked about, when the turn pairs a
    # different one with each. "Zainab's connects and Awais's answered
    # calls" is two (subject, measure) pairs, and with one metric on the
    # IR it could only be answered by attaching both measures to whichever
    # person resolved — not a partial answer but the wrong person's number
    # under the right label, which is why nlu_pipeline._distributes_metrics
    # refuses the shape outright today.
    #
    # None means "use the query's own metric", so every existing subject
    # behaves exactly as before.
    metric: Optional[MetricRef] = None


class MetricRef(BaseModel):
    key: str
    confidence: float = 1.0


class Filter(BaseModel):
    field: str                                   # a metric key, or "team" | "company" | "advisor" | "attendance_status"
    operator: Operator = "="
    value: Optional[Union[str, float, int, list]] = None
    confidence: float = 1.0


class FilterGroup(BaseModel):
    """A boolean combination of filters — the shape a flat list cannot hold.

    P0. `QueryIR.filters` is AND-combined by construction, so "BCMs in
    Blue Area OR Downtown" had no representation: the disjunction either
    became a conjunction matching nobody, or one branch was dropped
    before compilation. Same for exclusion — `excluding` routed a query
    to the LLM and was then unrepresentable, so the word changed the PATH
    without changing the ANSWER.

    `not` takes its children as a group: `not(A)` negates one, and
    `not(A, B)` negates their conjunction, which is what "excluding X and
    Y" means. Written that way rather than as a unary node so the shape
    is uniform and the LLM cannot emit a `not` with the wrong arity.
    """

    op: Literal["and", "or", "not"] = "and"
    children: list[Union["FilterGroup", Filter]] = Field(default_factory=list)

    def leaves(self) -> list[Filter]:
        """Every Filter in this subtree, depth-first, left to right.

        The BOOLEAN STRUCTURE IS LOST here, deliberately. Callers that
        ask "which fields did this query constrain" — grounding, the
        condition columns, the filters summary — need the set, not the
        shape; only the compiler needs the shape.
        """
        found: list[Filter] = []
        for child in self.children:
            found.extend(child.leaves() if isinstance(child, FilterGroup) else [child])
        return found


FilterGroup.model_rebuild()
# Subject.metric forward-references MetricRef, which is declared below it
# (`from __future__ import annotations` makes every annotation a string).
Subject.model_rebuild()


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
    # THE operation this query performs, from the one registry that
    # declares them (app/llm/operations.py).
    #
    # `intent` below is the older, narrower name for the same idea: seven
    # values, one of five vocabularies that named these concepts without
    # any of them deriving from another. `operation` is the single field
    # those collapse into, and it can carry operations `intent` has no
    # value for — a roster, an ancestry walk, a manager lookup.
    #
    # None means "read it from `intent`", which resolved_operation() does,
    # so an IR built before this field existed answers identically and
    # nothing has to set both.
    operation: Optional[str] = None
    intent: Intent
    subject_level: Level = "advisor"
    subjects: list[Subject] = Field(default_factory=list)
    # THE PRIMARY measure: what the answer is ranked and sorted by, and
    # what every existing consumer reads. Unchanged.
    metric: Optional[MetricRef] = None
    # P0: EVERY measure the turn named, primary first. The IR held one,
    # so a two-measure question lost one before compilation — QueryPlan
    # has carried a `metrics` list for exactly this reason and the IR was
    # never widened to match.
    #
    # Empty means "just the primary", so an IR built the old way is
    # unaffected; read it through metric_keys() rather than directly, and
    # the two cases collapse.
    metrics: list[MetricRef] = Field(default_factory=list)
    filters: list[Filter] = Field(default_factory=list)      # AND-combined
    # P0: the boolean structure a flat list cannot express, AND-combined
    # WITH `filters` above rather than replacing it. None — every query
    # that needs only conjunction — compiles down exactly the path it
    # always did. See FilterGroup.
    filter_tree: Optional[FilterGroup] = None
    time_range: TimeRange = Field(default_factory=TimeRange)
    sort: Sort = Field(default_factory=Sort)
    limit: Optional[int] = 10
    group_by: Optional[Level] = None
    # ---- HIERARCHY READS -------------------------------------------
    #
    # "how many advisors report directly to the Unit Head in AMD" needs
    # THREE levels and a relation, and the IR carried one. `subject_level`
    # conflates "who the query is about" with "what to return", so it can
    # express "advisors in AMD" but not "advisors beneath AMD's unit
    # head" — and certainly not the difference between the whole subtree
    # and the immediate reports.
    #
    # Because the shape had no representation, `roster`/`direct_reports`/
    # `scoped_reports` were marked plan-only and the LLM was never asked.
    # Routing then hinged on the literal token "directly": drop it and
    # the same question re-routed and answered something else.
    #
    # All three default to the values an IR built before they existed
    # would have, so nothing that does not set them changes behaviour.

    # WHICH level to enumerate. None means "the same level the query is
    # grouped at", which is every non-hierarchy query.
    target_level: Optional[Level] = None
    # The level the target sits BENEATH. When it differs from the named
    # subject's own level the subject is the SCOPE and the role holder is
    # read out of it ("the Unit Head in AMD" -> scope=team AMD,
    # subject_of=unit_head), which is what get_manager_of_group already
    # does for reverse lookups.
    subject_of: Optional[Level] = None
    # How far down to look. "subtree" is every descendant — the reading
    # one denormalised column match gives for free. "direct" is only the
    # immediate reports. This is the field that carries "directly" /
    # "immediately" as MEANING rather than as a keyword in a router.
    relation: Literal["subtree", "direct"] = "subtree"

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
    # EVERY MEANING-CHANGING REPAIR the validator made, in order.
    #
    # ir_validator does not only reject — it rewrites: it normalises the
    # intent from the registry, copies a subject's level over
    # `subject_level`, corrects a near-miss metric key, re-types a subject
    # the parser mislabelled, drops an ungroundable one, prunes a filter
    # it cannot ground, and re-labels a metric-free list as a population.
    # Each is defensible and each CHANGES WHAT THE QUERY MEANS, and until
    # now they were invisible: the logs showed the model's raw output and
    # the final IR, with no record of which layer moved anything in
    # between. "The LLM got it wrong" and "we rewrote it afterwards" were
    # indistinguishable after the fact — the single hardest thing to
    # establish when a production answer is wrong.
    #
    # Each entry is {"field", "from", "to", "why"}, so the raw parse is
    # reconstructible by replaying them backwards from the final IR.
    #
    # Observability only, like `nlu_mode`: not in the LLM output schema,
    # never validated, and never rendered to the user — it is persisted in
    # ChatLog.resolved_ir and carried on the request trace.
    repairs: list[dict] = Field(default_factory=list)
    # PHASE 6 — the principal this query was built for.
    #
    # CARRIED, NOT ENFORCED. There is no authorization policy in this
    # system: tokens hold a `role` claim that nothing consumes, and the
    # posture is an open decision. Populating a scope here from an
    # invented policy would silently shrink results with no stated
    # reason, so this records who asked and nothing reads it. When a
    # policy is settled it is applied from this one field.
    authorization_scope: Optional[dict] = None

    # ---- accessors -------------------------------------------------
    #
    # Each of these exists so a caller reads ONE thing where the model now
    # holds two. The alternative — every consumer checking both the flat
    # field and the new one — is how the two representations drift apart,
    # and there are around twenty such consumers.

    def resolved_operation(self) -> str:
        """The operation, from `operation` or derived from `intent`.

        ONE reading, so a consumer never has to check both fields and the
        two can never disagree about what this query is. Falls back to
        the intent's own name for an operation the registry does not
        know, rather than raising: an unrecognised value is the
        validator's business, not this accessor's.
        """
        from app.llm import operations

        if self.operation and self.operation in operations.OPERATIONS:
            return self.operation
        mapped = operations.for_ir_intent(self.intent)
        return mapped.name if mapped else self.intent

    def filter_leaves(self) -> list[Filter]:
        """Every filter this query applies, flat, tree included.

        For callers asking WHICH fields are constrained (grounding, the
        condition columns, the reply's filter summary). The boolean shape
        is deliberately not preserved — only the compiler needs it.
        """
        leaves = list(self.filters)
        if self.filter_tree is not None:
            leaves.extend(self.filter_tree.leaves())
        return leaves

    def primary_metric(self) -> Optional[str]:
        """THE measure this query is valued and ordered by, from wherever
        the parser actually put it.

        A metric can legitimately live in four places, and which one gets
        used depends on how the question was phrased, not on what it
        means:

            sort.metric   an explicit ranking      "top advisors BY revenue"
            metric.key    the named primary        "revenue of Blue Area"
            metrics[0]    several measures named   "connects and answered calls"
            a filter      the measure is a CONDITION
                          "advisors with connects above 1000"

        Every consumer used to read only the first two, so the fourth
        shape — which is the natural one for every "X above N" question —
        looked like a query with no measure at all. `ir_validator` then
        asked the user which metric they meant, for a sentence that named
        one, and `_effective_metric` returned None, which would have made
        the compiler answer "no data" instead. One reading, in one place,
        is what stops those two disagreeing again.

        ORDER IS THE DOCUMENTED SEMANTICS, not a preference: an explicit
        sort wins over a named primary, which wins over the first of
        several (the prompt specifies "primary first" for `metrics`),
        which wins over a condition. A filter is last because it is the
        weakest evidence of intent — it says the measure is INTERESTING,
        not that the answer is ranked by it.

        THIS DOES NOT MUTATE. Nothing here writes `metric` or
        `sort.metric`, so a filtered list keeps `metric=None` and
        response_planner still sees `filtered_list` rather than
        `leaderboard`. Deriving the column to compute is a different
        question from deciding the answer's shape, and conflating them is
        how a "who matches this" question would start rendering as a
        ranking.

        A POPULATION HAS NO PRIMARY MEASURE, by definition — it is the
        operation for "who matches this constraint, with nothing to rank
        by". Returning a filter's metric for it would put a value column
        on a question that asked for names, which is the regression
        response_planner's own comment warns about ("printed 'no data'
        beside every name"). None is the correct answer here, not a
        missing one.
        """
        if self.resolved_operation() == "population":
            return None
        if self.sort and self.sort.metric:
            return self.sort.metric
        if self.metric:
            return self.metric.key
        if self.metrics:
            return self.metrics[0].key
        from app.llm.metric_ontology import METRICS

        for leaf in self.filter_leaves():
            if leaf.field in METRICS:
                return leaf.field
        return None

    def metric_keys(self) -> list[str]:
        """Every measure named, primary first, deduplicated in order.

        `metrics` empty means the primary alone, so an IR built before
        that field existed answers this identically.
        """
        ordered = ([self.metric.key] if self.metric else []) + [m.key for m in self.metrics]
        seen: set[str] = set()
        return [k for k in ordered if k and not (k in seen or seen.add(k))]

    def is_hierarchy_read(self) -> bool:
        """Does this query enumerate a level beneath a subject?

        True only when a target level is named AND something to scope it
        beneath exists. Both halves matter: a target with no subject is
        an ordinary population at that level, and a subject with no
        target is an ordinary scoped query.
        """
        return bool(self.target_level) and bool(self.subject_of or self.subjects)

    def grouping_level(self) -> str:
        """The level rows are grouped by.

        `group_by` was carried by the schema and read by NOTHING — the
        compiler grouped by `subject_level` unconditionally, so a model
        that filled it correctly got no benefit. It is honoured here when
        set, and falls back to the level that has always been used.
        """
        return self.target_level or self.group_by or self.subject_level

    def compare_period(self) -> Optional[str]:
        """The period this query is compared AGAINST, when it is one.

        `compare_to` was likewise inert. Reading it through here means the
        one place that decides "is this a period comparison" is also the
        one place that has to be taught to render it.
        """
        compare_to = getattr(self.time_range, "compare_to", None)
        if not compare_to or compare_to == self.time_range.period:
            return None
        return compare_to


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

    UNLESS THE SENTENCE NAMED SEVERAL. `plan.metric` is one key, so
    binding every threshold to it made "target achievement below 50% and
    answered calls % below 20%" compile as `achievement_pct < 50 AND
    achievement_pct < 20` — the second condition applied to the first
    condition's column, which reduces the pair to `< 20` and answers a
    question nobody asked. entity_extractor pairs each comparator with
    the measure beside it (_bind_threshold_metrics) and leaves the key on
    the threshold; this reads it, falling back to `plan.metric` for the
    single-measure queries that are the overwhelming majority and where
    no such key is set.

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

    period = _period_for(plan)
    filters: list["Filter"] = []
    for t in getattr(plan, "thresholds", None) or []:
        if not t.get("operator") or t.get("value") is None:
            continue
        # The measure this comparator was written beside, or the one the
        # plan resolved when the sentence named only one.
        named = t.get("metric") or metric_key
        # `or named`: a measure with no sibling for that period keeps its
        # own field rather than losing the filter. The sort metric will
        # independently fail to resolve and the query returns "can't
        # answer", so this cannot produce a wrong answer on its own.
        field = resolve_metric_for_period(named, period) or named
        filters.append(Filter(field=field, operator=t["operator"], value=t["value"]))
    return filters


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

    from app.llm import operations

    # The plan's action IS an operation — the registry knows it by that
    # name. Stamped here so a plan-built IR carries the same single field
    # a parser-built one does, rather than only the narrower `intent`.
    planned = operations.for_plan_action(getattr(plan, "action", None))

    return QueryIR(
        intent="comparison" if subjects else "leaderboard",
        operation=(planned.name if planned and planned.expressible_in_ir
                   else ("comparison" if subjects else "leaderboard")),
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
