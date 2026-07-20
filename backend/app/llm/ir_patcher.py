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
_PERIOD_PHRASES = {
    "ytd": "YTD", "this year": "YTD", "year to date": "YTD",
    "3m": "3M", "quarter": "3M", "three month": "3M", "3 month": "3M", "last 3 months": "3M",
    "mtd": "MTD", "this month": "MTD",
}


def _patch_only_entity(ir: QueryIR, text: str, entities: dict) -> bool:
    """'only Graana' / 'just Blue Area' — replace the same-typed entity
    filter with the grounded match. Requires the extractor to have
    grounded exactly one team or one company from the message."""
    if not _ONLY_RE.match(text.strip()):
        return False
    teams = entities.get("teams", [])
    companies = entities.get("companies", [])
    if len(teams) + len(companies) != 1:
        return False
    field, value = ("team", teams[0]) if teams else ("company", companies[0])
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


# The compiler encodes period in the metric key (mtd_cleared vs
# ytd_cleared bind to different Performance rows) — changing
# time_range.period alone wouldn't change the result, so a period
# follow-up also swaps the metric within the cleared family.
_PERIOD_METRIC_SWAP = {
    ("mtd_cleared", "YTD"): "ytd_cleared",
    ("mtd_cleared", "3M"): "three_month_cleared",
    ("ytd_cleared", "MTD"): "mtd_cleared",
    ("ytd_cleared", "3M"): "three_month_cleared",
    ("three_month_cleared", "MTD"): "mtd_cleared",
    ("three_month_cleared", "YTD"): "ytd_cleared",
}


def _patch_period(ir: QueryIR, text: str) -> bool:
    q = text.lower()
    for phrase, period in _PERIOD_PHRASES.items():
        if phrase in q:
            ir.time_range.period = period
            current_metric = ir.sort.metric or (ir.metric.key if ir.metric else None)
            swapped = _PERIOD_METRIC_SWAP.get((current_metric, period))
            if swapped:
                if ir.metric:
                    ir.metric.key = swapped
                if ir.sort.metric:
                    ir.sort.metric = swapped
            return True
    return False


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
    # "summary" is what the rule planner says for a BARE entity mention
    # ("graana") — that's a legitimate new question. But with an explicit
    # modifier prefix ("only graana") it's unambiguously a narrowing of
    # the previous query, so it stays patchable.
    if plan_action == "summary":
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
