"""
Deterministic follow-up patcher (P5 of the NLU rework). When the previous
turn produced a QueryIR and the new message is a short, single-purpose
modifier — "only Graana", "top 5", "sort ascending", "ytd", "achievement
above 80" — patch the prior IR directly instead of spending an LLM round
trip. Anything ambiguous returns None and falls through to the normal
parse path, where the LLM sees the prior IR in its prompt anyway.

Guardrails (the escape hatch): a patch is only applied when the message
is short (few residual tokens beyond what a rule consumed), matches at
least one rule, and doesn't independently resolve as a self-standing
query via the rule planner. "Show top teams by overdue" after a revenue
leaderboard is a NEW query, not a patch — build_query_plan resolves it,
so try_patch declines it.
"""

from __future__ import annotations

import re

from app.llm import hierarchy, query_compiler, temporal_parser
from app.llm.entity_extractor import _extract_thresholds
from app.llm.fallback_reasoning import fuzzy_resolve_metric
from app.llm.query_ir import Filter, QueryIR
from app.core.logger import get_logger

log = get_logger("llm.ir_patcher")

_MAX_TOKENS = 8  # longer than this and it's a sentence, not a modifier

_ONLY_RE = re.compile(r"^(?:only|just|for)\b\s*(.*)$", re.I)
_TOP_N_RE = re.compile(r"\b(top|bottom)\s+(\d+)\b", re.I)
_SORT_DIR_RE = re.compile(r"\b(ascending|descending|asc|desc|lowest first|highest first)\b", re.I)
_SORT_BY_RE = re.compile(r"\bsort(?:ed)?\s+by\s+(.+)$", re.I)
_REMOVE_FILTER_RE = re.compile(r"\b(all teams|all companies|everyone|remove filters?|clear filters?)\b", re.I)


def _patch_only_entity(ir: QueryIR, text: str, entities: dict) -> bool:
    """'only Graana' / 'just Blue Area' / 'only unit head John' — replace
    the same-typed entity filter with the grounded match. Requires the
    extractor to have grounded exactly one entity, of any single hierarchy
    level (team/company/unit_head/zonal_head/business_center), from the
    message — driven generically from hierarchy.LEVEL_ENTITY_KEYS instead
    of a hardcoded team/company-only check."""
    if not _ONLY_RE.match(text.strip()):
        return False
    matches = [
        (level, entities[entity_key][0])
        for level, entity_key in hierarchy.LEVEL_ENTITY_KEYS.items()
        if entities.get(entity_key)
    ]
    if len(matches) != 1:
        return False
    field, value = matches[0]
    ir.filters = [f for f in ir.filters if f.field != field]
    ir.filters.append(Filter(field=field, operator="=", value=value))
    return True


def _patch_top_n(ir: QueryIR, text: str) -> bool:
    m = _TOP_N_RE.search(text)
    if not m:
        return False
    ir.limit = int(m.group(2))
    ir.sort.direction = "asc" if m.group(1).lower() == "bottom" else "desc"
    return True


def _patch_sort_direction(ir: QueryIR, text: str) -> bool:
    m = _SORT_DIR_RE.search(text)
    if not m:
        return False
    token = m.group(1).lower()
    ir.sort.direction = "asc" if token in ("ascending", "asc", "lowest first") else "desc"
    return True


def _patch_sort_metric(ir: QueryIR, text: str) -> bool:
    m = _SORT_BY_RE.search(text)
    if not m:
        return False
    metric = fuzzy_resolve_metric(m.group(1))
    if not metric:
        return False
    ir.sort.metric = metric
    return True


def _patch_period(ir: QueryIR, text: str) -> bool:
    """'what about ytd' — re-point the IR at the same measure, other period.

    Phase 2 removed two pieces of local knowledge from this function.

    The PHRASES now come from temporal_parser, the one module that also
    knows which windows this data model cannot answer. The substring
    table that used to live here had no such notion, so "last quarter"
    mapped to 3M here while temporal_parser correctly refused it.

    The METRIC now comes from query_compiler, the single authority on
    which key answers a measure at a period. The hardcoded swap table
    this replaces covered only the cleared family, so every other metric
    kept its MTD binding while `time_range.period` recorded the new
    period — a wrong answer that looked right in the trace.

    When the measure has no data at the requested period the patch is
    DECLINED (returns False) rather than applied partially: the message
    then goes through the normal parse path, which can say so, instead of
    this function quietly relabelling MTD numbers.
    """
    period = temporal_parser.match_period(text)
    if period is None:
        return False

    current_metric = ir.sort.metric or (ir.metric.key if ir.metric else None)
    if current_metric is not None:
        resolved = query_compiler.resolve_metric_for_period(current_metric, period)
        if resolved is None:
            return False
        if ir.metric:
            ir.metric.key = resolved
        if ir.sort.metric:
            ir.sort.metric = resolved

    ir.time_range.period = period
    return True


def _patch_threshold_filter(ir: QueryIR, text: str) -> bool:
    """'achievement above 80', 'attendance rate below 90%' — a metric
    mention plus exactly one comparator/number pair appends a filter."""
    thresholds = _extract_thresholds(text.lower())
    if len(thresholds) != 1:
        return False
    metric = fuzzy_resolve_metric(text)
    if not metric:
        return False
    t = thresholds[0]
    ir.filters = [f for f in ir.filters if f.field != metric]
    ir.filters.append(Filter(field=metric, operator=t["operator"], value=t["value"]))
    return True


def _patch_remove_filters(ir: QueryIR, text: str) -> bool:
    if not _REMOVE_FILTER_RE.search(text):
        return False
    ir.filters = []
    ir.subjects = []
    return True


_RULES = [
    _patch_remove_filters,
    _patch_top_n,
    _patch_sort_metric,       # before _patch_sort_direction: "sorted by X descending" wants both anyway
    _patch_sort_direction,
    _patch_threshold_filter,
    _patch_period,
]


def try_patch(prior: QueryIR, text: str, entities: dict, plan_action: str) -> QueryIR | None:
    """A patched copy of `prior`, or None when this message should go
    through the full parse path instead. `plan_action` is the rule
    planner's verdict on the message standing alone — anything it
    resolved as its own query ("leaderboard", "lookup", ...) is treated
    as a new question, not a modifier."""
    stripped = text.strip()
    # "summary"/"breakdown" is what the rule planner says for a BARE entity
    # mention ("graana", "unit head john") — that's a legitimate new
    # question. But with an explicit modifier prefix ("only graana") it's
    # unambiguously a narrowing of the previous query, so it stays patchable.
    if plan_action in ("summary", "breakdown"):
        if not _ONLY_RE.match(stripped):
            return None
    elif plan_action != "unresolved":
        return None
    if len(re.findall(r"\S+", stripped)) > _MAX_TOKENS:
        return None

    ir = prior.model_copy(deep=True)
    applied = [rule.__name__ for rule in _RULES if rule(ir, stripped)]
    # entity narrowing is checked separately: it shares its trigger words
    # with ordinary phrasing, so it requires the extractor to have grounded
    # a real entity, not just a keyword hit
    if _patch_only_entity(ir, stripped, entities):
        applied.append("_patch_only_entity")

    if not applied:
        return None

    ir.missing = []
    log.info(f"Patched prior IR via {applied} for follow-up: '{stripped}'")
    return ir
