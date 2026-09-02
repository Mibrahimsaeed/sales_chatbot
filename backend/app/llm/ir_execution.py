"""Executable QueryIR -> the existing SQL layer -> a VERIFIED result set.

    executable QueryIR -> existing SQL/query layer -> database -> result

NO NEW SQL. Every row here comes from `query_compiler.compile_and_run`
and every total from `count_ir` — the same functions chat_service and the
leaderboard API already call, so the new path and the legacy path cannot
compute different numbers for the same IR. Metric bindings, period
resolution, joins and aggregation are untouched.

WHAT THIS ADDS is the last word in the target flow: "verified". Phases
4-6 established what the query MEANS, that its entities exist, that its
relationship is real, and that the IR carries identifiers rather than
words. None of that says the rows that came back honour any of it. The
checks below close that gap by reading the RESULT:

    scope        every row is inside the entity that was grounded
    identifiers  advisor-addressed queries return only the resolved wids
    hierarchy    every row is one of the members verification found
    grouping     rows are keyed at the level the IR groups by
    time         the metric that produced the value is the one the
                 requested period resolves to

Each check can FAIL LOUDLY rather than silently, which is the point: a
scope that quietly stopped applying returns a plausible number for the
wrong population, and nothing downstream can tell.

WHAT IT DELIBERATELY DOES NOT CHECK. Determinism is not verified by
re-running the query on every request — that would double the cost of
every question to detect something that is a property of the SQL layer,
not of one execution. It is pinned by a test instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.llm import hierarchy
from app.llm.hierarchy_grounding import HierarchyGrounding
from app.llm.query_compiler import (
    compile_and_run, count_ir, effective_metric, resolve_metric_for_period,
)
from app.llm.query_ir import MetricRef, QueryIR, Sort

# A check that ran and held / ran and did not / could not be decided from
# the result alone.
PASSED = "passed"
FAILED = "failed"
SKIPPED = "skipped"

SCOPE = "scope"
IDENTIFIERS = "identifiers"
HIERARCHY = "hierarchy"
GROUPING = "grouping"
TIME = "time"
METRICS = "metrics"

# Per-row key holding every requested measure. `value` stays the primary,
# so a consumer that only knows about single-metric rows is unaffected.
METRIC_CELLS_KEY = "metrics"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict:
        return {"check": self.name, "status": self.status, "detail": self.detail}


@dataclass
class ExecutedResult:
    """Rows, plus what was confirmed about them.

    `rows` is None — distinct from empty — when the compiler could not
    answer the IR at all (no binding for the measure at this level). An
    empty list means the query ran and matched nothing, which is an
    answer.
    """
    ir: QueryIR
    rows: list[dict] | None = None
    total: int | None = None
    checks: list[Check] = field(default_factory=list)
    # Every measure the query asked for, primary first.
    metrics: list[str] = field(default_factory=list)
    # Measures that were asked for and CANNOT be computed at this
    # grouping level. Recorded rather than dropped: silently answering
    # with two of three requested measures is the defect this phase
    # exists to prevent, and it is invisible in the reply.
    unavailable_metrics: list[str] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return self.rows is not None

    @property
    def row_count(self) -> int:
        return len(self.rows) if self.rows else 0

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAILED]

    @property
    def verified(self) -> bool:
        """True when nothing that could be checked came back wrong.

        A SKIPPED check is not a failure: some properties cannot be read
        off the result (a unit_head scope does not appear in the returned
        columns), and treating "not checkable here" as "broken" would
        make the flag useless.
        """
        return self.answered and not self.failures

    def to_dict(self) -> dict:
        return {
            "answered": self.answered,
            "verified": self.verified,
            "row_count": self.row_count,
            "total": self.total,
            "metrics": list(self.metrics),
            "unavailable_metrics": list(self.unavailable_metrics),
            "checks": [c.to_dict() for c in self.checks],
        }


# Which returned column carries each level's value. compile_and_run
# projects a fixed row shape, so only these levels can be confirmed from
# the result itself; the rest are reported SKIPPED rather than guessed at.
_ROW_COLUMN_FOR_LEVEL = {"team": "team", "company": "company", "advisor": "name"}


def _check_scope(ir: QueryIR, rows: list[dict]) -> list[Check]:
    """Every row must sit inside every grounded subject.

    Reads `resolved_id or value` — the same expression the compiler's
    subject filter uses — so this confirms the filter's effect rather
    than restating its inputs.
    """
    checks: list[Check] = []
    for subject in ir.subjects:
        column = _ROW_COLUMN_FOR_LEVEL.get(subject.type)
        expected = subject.resolved_id or subject.value
        if column is None:
            checks.append(Check(
                SCOPE, SKIPPED,
                f"{subject.type} '{expected}' is not projected in the result rows"))
            continue
        if subject.type == "advisor" and ir.grouping_level() != "advisor":
            checks.append(Check(
                SCOPE, SKIPPED,
                f"advisor '{expected}' is aggregated into {ir.grouping_level()} rows"))
            continue
        stray = [r.get(column) for r in rows
                 if (r.get(column) or "").lower() != expected.lower()]
        checks.append(Check(
            SCOPE, FAILED if stray else PASSED,
            f"{len(stray)} row(s) outside {subject.type} '{expected}': {stray[:3]}"
            if stray else f"every row is inside {subject.type} '{expected}'"))
    return checks


def _check_identifiers(ir: QueryIR, rows: list[dict]) -> Check:
    """An advisor-addressed query must return only the resolved people.

    The defect this catches is the one wids exist to prevent: matching by
    name sums several people into one row, so a query about ONE Yasir Ali
    silently answers about all of them.
    """
    wids = [s.resolved_wid for s in ir.subjects
            if s.type == "advisor" and s.resolved_wid is not None]
    if not wids:
        return Check(IDENTIFIERS, SKIPPED, "no advisor subject addressed by wid")
    if ir.grouping_level() != "advisor":
        return Check(IDENTIFIERS, SKIPPED,
                     f"rows are grouped at {ir.grouping_level()}, not per advisor")

    stray = [r.get("wid") for r in rows if r.get("wid") not in wids]
    return Check(IDENTIFIERS, FAILED if stray else PASSED,
                 f"{len(stray)} row(s) outside the resolved wids {wids}: {stray[:3]}"
                 if stray else f"every row is one of the resolved wids {wids}")


def _check_hierarchy(ir: QueryIR, rows: list[dict],
                     verified: HierarchyGrounding | None) -> Check:
    """Rows must be the members hierarchy grounding actually found.

    This is the strongest check available, because the member list came
    from running the relationship against the data — so it catches a
    scope that compiled to a DIFFERENT population than the one that was
    verified, which no amount of reading the IR would reveal.
    """
    if verified is None or not verified.is_hierarchy:
        return Check(HIERARCHY, SKIPPED, "not a hierarchy read")
    if verified.member_count > len(verified.members):
        return Check(HIERARCHY, SKIPPED,
                     f"member list truncated at {len(verified.members)} of "
                     f"{verified.member_count}")

    expected = {m.lower() for m in verified.members}
    stray = [r.get("name") for r in rows if (r.get("name") or "").lower() not in expected]
    return Check(HIERARCHY, FAILED if stray else PASSED,
                 f"{len(stray)} row(s) not among the {verified.member_count} verified "
                 f"members: {stray[:3]}" if stray else
                 f"every row is one of the {verified.member_count} verified members")


def _check_grouping(ir: QueryIR, rows: list[dict]) -> Check:
    """One row per distinct value at the grouping level.

    A duplicated key means the grouping silently did not apply, which
    shows up as a plausible list with the same entity twice rather than
    as an error.
    """
    level = ir.grouping_level()
    column = _ROW_COLUMN_FOR_LEVEL.get(level)
    if column is None:
        return Check(GROUPING, SKIPPED, f"{level} is not projected in the result rows")
    if level == "advisor":
        keys = [r.get("wid") for r in rows]
    else:
        keys = [(r.get(column) or "").lower() for r in rows]
    duplicates = len(keys) - len(set(keys))
    return Check(GROUPING, FAILED if duplicates else PASSED,
                 f"{duplicates} duplicate {level} key(s) in the result"
                 if duplicates else f"one row per {level}, {len(keys)} in total")


def _check_time(ir: QueryIR) -> Check:
    """The value must be produced by the metric the requested period
    resolves to.

    `effective_metric` is the authority the compiler itself uses, so
    reading it here confirms the period reached execution rather than
    being recorded on the IR and ignored — which is exactly what happened
    before it existed: "what about YTD" kept the MTD binding and reported
    an MTD number under a YTD question.
    """
    resolved = effective_metric(ir)
    if resolved is None:
        return Check(TIME, SKIPPED, "no measure to resolve (population query)")
    return Check(TIME, PASSED,
                 f"{ir.time_range.period} resolved to '{resolved}'")


def _row_key(ir: QueryIR, row: dict):
    """How a row is identified when merging measures onto it.

    Identity differs by level and getting it wrong is not cosmetic: an
    ADVISOR is addressed by wid, because names are not identifiers here
    (238 duplicate-name groups in production) and merging by name would
    show one person's answered calls beside another's connects. A group
    level is addressed by the value it groups on, unique per row by
    construction. The same split `_companion_value` already makes.
    """
    if ir.grouping_level() == "advisor":
        return row.get("wid")
    return (row.get("name") or "").lower()


def _values_for_metric(db: Session, ir: QueryIR, key: str) -> dict | None:
    """Every row's value for ONE measure, keyed by row identity.

    ONE QUERY PER MEASURE, not one per row. The existing bundling in
    chat_service fetches companions a cell at a time — 48 rows by 3
    measures is 144 round trips — because it starts from rows that are
    already rendered. Starting from the IR instead means the same
    compiler, the same bindings and the same period resolution produce
    each column, which is what guarantees a measure read as a companion
    equals the same measure read as the primary.

    `limit` is dropped deliberately: the companion query is ordered by
    its OWN measure, so the rows the primary selected are not necessarily
    within its top N. Ordering is irrelevant here — the result is indexed
    by identity and merged.
    """
    clone = ir.model_copy(deep=True)
    clone.metric = MetricRef(key=key)
    clone.sort = Sort(metric=key, direction=ir.sort.direction if ir.sort else "desc")
    clone.limit = None

    rows = compile_and_run(db, clone)
    if rows is None:
        return None
    return {_row_key(ir, row): row.get("value") for row in rows}


def _attach_metrics(db: Session, ir: QueryIR, rows: list[dict]) -> tuple[list[str], list[str]]:
    """Populate every requested measure on every row.

    Returns (available, unavailable). The primary's value is REUSED
    rather than recomputed: it is the number the rows were ranked and
    rendered by, and fetching it twice is how a row's headline and its
    own column start to disagree.
    """
    requested = ir.metric_keys()
    if not requested:
        return [], []

    primary = effective_metric(ir)
    available: list[str] = []
    unavailable: list[str] = []
    columns: dict[str, dict] = {}

    for key in requested:
        resolved = resolve_metric_for_period(key, getattr(ir.time_range, "period", None))
        if resolved is not None and resolved == primary:
            available.append(key)
            continue
        values = _values_for_metric(db, ir, key)
        if values is None:
            unavailable.append(key)
            continue
        columns[key] = values
        available.append(key)

    for row in rows:
        identity = _row_key(ir, row)
        row[METRIC_CELLS_KEY] = {
            key: (row.get("value") if key not in columns
                  else columns[key].get(identity))
            for key in available
        }
    return available, unavailable


def _check_metrics(ir: QueryIR, result: "ExecutedResult") -> Check:
    """Every measure the query named must come back.

    The failure this catches is a silent reduction to the first metric:
    "answered calls, connects and meeting rate for Blue Area" returning
    only connects, with nothing in the reply to say the other two were
    dropped.
    """
    requested = ir.metric_keys()
    if len(requested) < 2:
        return Check(METRICS, SKIPPED,
                     f"single measure: {requested[0] if requested else 'none'}")
    if result.unavailable_metrics:
        return Check(METRICS, FAILED,
                     f"{len(result.unavailable_metrics)} of {len(requested)} measures "
                     f"cannot be computed at level '{ir.grouping_level()}': "
                     f"{result.unavailable_metrics}")
    return Check(METRICS, PASSED,
                 f"all {len(requested)} measures computed: {result.metrics}")


def execute(db: Session, ir: QueryIR, *, offset: int = 0,
            verified_hierarchy: HierarchyGrounding | None = None,
            with_total: bool = False) -> ExecutedResult:
    """Run the IR through the existing compiler and verify the result.

    `verified_hierarchy` is the Phase 5 report for this same
    interpretation. Optional, because an ordinary metric query has none —
    when supplied it enables the strongest check available.
    """
    rows = compile_and_run(db, ir, offset=offset)
    result = ExecutedResult(ir=ir, rows=rows)

    if rows is None:
        result.checks.append(Check(
            GROUPING, SKIPPED,
            f"the compiler cannot answer this IR at level '{ir.grouping_level()}'"))
        return result

    if with_total:
        result.total = count_ir(db, ir)

    result.metrics, result.unavailable_metrics = _attach_metrics(db, ir, rows)
    result.checks.append(_check_metrics(ir, result))

    result.checks.extend(_check_scope(ir, rows))
    result.checks.append(_check_identifiers(ir, rows))
    result.checks.append(_check_hierarchy(ir, rows, verified_hierarchy))
    result.checks.append(_check_grouping(ir, rows))
    result.checks.append(_check_time(ir))
    return result


def execute_semantic_model(db: Session, model, *, offset: int = 0,
                           with_total: bool = False,
                           principal=None) -> tuple[ExecutedResult | None, object]:
    """The whole flow, from meaning to verified rows.

        semantic model -> grounding -> hierarchy grounding -> validation
                       -> executable QueryIR -> SQL -> verified result

    Returns (result, verdict). `result` is None when the interpretation
    did not survive validation, or when the operation has no IR
    representation and is answered from the plan instead — the verdict
    says which.

    Composed here rather than in each caller so the order cannot be got
    wrong: every step's output is the next step's input, and skipping
    validation is what would let an unresolved name reach a scope filter.

    NOT wired into nlu_pipeline. The live request path still executes
    through the legacy route, and running both would execute every query
    twice. Switching over changes what every query returns, which is a
    deliberate cutover rather than part of connecting the layers.
    """
    from app.llm import grounding as entity_grounding
    from app.llm import hierarchy_grounding as hierarchy_verification
    from app.llm import ir_conversion, semantic_validation

    grounded = entity_grounding.ground(model, db)
    verified = hierarchy_verification.verify(model, grounded, db)
    verdict = semantic_validation.validate(model, grounded, verified, db,
                                           principal=principal)
    if not verdict.is_executable:
        return None, verdict

    ir = ir_conversion.to_query_ir(model, grounded, verified, verdict,
                                   principal=principal)
    if ir is None:
        return None, verdict

    return execute(db, ir, offset=offset, verified_hierarchy=verified,
                   with_total=with_total), verdict
