"""THE operation registry — one declaration per thing this system does.

WHY. Five vocabularies named the same concepts, none derived from
another, all maintained by hand:

    query_planner build lambdas   17 plan actions
    intent_catalog.PRIOR          13 scoring priors
    query_ir.Intent                7 IR intents
    response_planner.DISPATCH_MODES 18 response modes
    llm_planner._INTENT_TO_ACTION  12 LLM intent names

Adding one capability meant six coordinated edits across five modules
with nothing forcing them to agree, and a missed one failed SILENTLY —
`direct_reports` was added, left out of one registry, and every query
using it answered "I'm not tracking that one" while the planner was
building the plan correctly.

WHAT THIS IS. Each operation declared ONCE, with the name each layer
knows it by. The other tables become views of this one: `verify()`
asserts every existing vocabulary member is accounted for, so a drift
that used to be a silent wrong answer is now a failing test.

WHAT THIS IS NOT. It does not change what any operation DOES, and it
does not delete any. `expressible_in_ir` records the real constraint the
consolidation runs into: several operations answer with a shape QueryIR
cannot hold — the profile and summary CARDS carry many measures at once,
the hierarchy READS enumerate a reporting line — so for those the plan
is still the only path. Widening the IR to cover them is the work that
unblocks dispatching on `operation` alone; naming them here is what makes
that list explicit instead of implicit in a routing condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Operation:
    """One thing the system can do, under the names each layer uses.

    `name` is canonical and deliberately equals `plan_action` wherever
    one exists — renaming settled concepts would add churn without
    adding meaning, and the point of this file is to have ONE name per
    operation, not a new one.
    """

    name: str
    # The rule planner's name for it. None when only the IR can express
    # the operation (a filtered list is produced by the parser, never by
    # a scorer).
    plan_action: Optional[str]
    # The QueryIR intent, when the IR can express this at all.
    ir_intent: Optional[str]
    # The response mode chat_service dispatches it as.
    dispatch_mode: str
    # False when the answer's SHAPE has no IR representation — a card of
    # many measures, or a hierarchy enumeration. These stay plan-served
    # until the IR can hold them.
    expressible_in_ir: bool
    summary: str


_ALL: tuple[Operation, ...] = (
    # ---- analytical: the IR expresses these, and the LLM leads ----
    Operation("leaderboard", "leaderboard", "leaderboard", "leaderboard", True,
              "rank subjects at one level by one measure"),
    Operation("filtered_list", None, "filtered_list", "filtered_list", True,
              "the members matching a constraint — parser-only, no scorer builds it"),
    # WHO, with no measure. Distinct from `roster`, which is plan-served
    # and takes a single entity filter: this one carries the IR's full
    # filter machinery, so "advisors excluding Blue Area" or "in A or B"
    # is expressible — and it joins no fact table, so the population is
    # not reduced to the rows that happen to have one.
    Operation("population", None, None, "population", True,
              "the members matching a constraint, with no ranking measure"),
    Operation("comparison", "comparison", "comparison", "comparison", True,
              "several named subjects set side by side"),
    # COMPILES AS a leaderboard scoped to one group, but is not the
    # canonical owner of that intent — `ir_intent` is the reverse mapping,
    # and two operations claiming one intent would make it ambiguous
    # (it silently did: for_ir_intent("leaderboard") returned whichever
    # was declared last).
    Operation("group_metric", "group_metric", None, "metric_value", True,
              "one group's own figure for one measure"),
    Operation("advisor_metric", "advisor_metric", None, "advisor_metric", False,
              "one person's figure for one measure"),

    # ---- card shapes: many measures at once, which QueryIR cannot hold ----
    Operation("lookup", "lookup", "lookup", "profile", False,
              "one person's whole profile"),
    Operation("summary", "summary", None, "hierarchy_summary", False,
              "a group's KPI card"),
    Operation("breakdown", "breakdown", "breakdown", "breakdown", False,
              "a group nested by team, with totals"),

    # ---- hierarchy reads: no IR intent describes enumerating a line ----
    Operation("roster", "roster", None, "roster", False,
              "who is in a group"),
    Operation("ancestry", "ancestry", None, "ancestry", False,
              "every level above someone"),
    Operation("reverse_hierarchy", "reverse_hierarchy", None, "manager", False,
              "who is above someone"),
    Operation("direct_reports", "direct_reports", None, "roster", False,
              "who reports to someone immediately"),
    Operation("scoped_reports", "scoped_reports", None, "roster", False,
              "everyone at a named level anywhere beneath someone"),

    # ---- other ----
    Operation("attendance_filter", "attendance_filter", None, "attendance", False,
              "advisors by attendance status"),
    Operation("trend", "trend", "trend", "unsupported", True,
              "change over time — declared, not yet answerable"),

    # ---- questions rather than answers ----
    Operation("clarify_person", "clarify_person", None, "clarification", False,
              "which of several people with this name"),
    Operation("clarify_ambiguous", "clarify_ambiguous", None, "clarification", False,
              "which level a name refers to"),
    Operation("clarify_metric", "clarify_metric", "clarify", "clarification", True,
              "which measure was meant"),
    Operation("comparison_incomplete", "comparison_incomplete", None, "clarification", False,
              "only one side of a comparison resolved"),
    Operation("unresolved", "unresolved", None, "no_data", False,
              "nothing could be resolved"),
)

OPERATIONS: dict[str, Operation] = {op.name: op for op in _ALL}

# Views. Derived, so they cannot drift from the declarations above.
BY_PLAN_ACTION: dict[str, Operation] = {
    op.plan_action: op for op in _ALL if op.plan_action
}
BY_IR_INTENT: dict[str, Operation] = {
    op.ir_intent: op for op in _ALL if op.ir_intent
}
# An IR intent must name exactly ONE operation, or the reverse lookup is
# ambiguous and silently resolves to whichever was declared last.
assert len(BY_IR_INTENT) == len([op for op in _ALL if op.ir_intent]), (
    "two operations claim the same ir_intent"
)
OPERATION_NAMES: tuple[str, ...] = tuple(OPERATIONS)
# The operations whose answer shape the IR can hold. The complement is
# the concrete to-do list for finishing the consolidation.
IR_EXPRESSIBLE: frozenset[str] = frozenset(
    op.name for op in _ALL if op.expressible_in_ir
)
PLAN_ONLY: frozenset[str] = frozenset(
    op.name for op in _ALL if not op.expressible_in_ir
)


def for_plan_action(action: str | None) -> Optional[Operation]:
    return BY_PLAN_ACTION.get(action) if action else None


def for_ir_intent(intent: str | None) -> Optional[Operation]:
    return BY_IR_INTENT.get(intent) if intent else None


def dispatch_mode_for(name: str | None) -> Optional[str]:
    op = OPERATIONS.get(name) if name else None
    return op.dispatch_mode if op else None


def verify() -> list[str]:
    """Every vocabulary member accounted for, or the reasons why not.

    Called by a test rather than at import: the modules it checks import
    each other, and a cycle at import time would be a worse failure than
    the drift it guards against. A test is still a mechanism forcing
    agreement — which is exactly what these five tables never had.
    """
    from app.llm import query_ir
    from app.llm.intent_catalog import PRIOR
    from app.llm.nlu_pipeline import _RULE_BASED_ACTIONS
    from app.llm.response_planner import DISPATCH_MODES
    import typing

    problems: list[str] = []

    for intent in typing.get_args(query_ir.Intent):
        if intent not in BY_IR_INTENT:
            problems.append(f"QueryIR.Intent {intent!r} has no operation")

    for action in _RULE_BASED_ACTIONS:
        if action not in BY_PLAN_ACTION:
            problems.append(f"_RULE_BASED_ACTIONS {action!r} has no operation")

    for prior in PRIOR:
        # PRIOR uses two SCORER names that are not plan actions —
        # `hierarchy` and the two profile/summary spellings — because a
        # scorer may build a different action than it is named for.
        if prior in BY_PLAN_ACTION or prior in _SCORER_ALIASES:
            continue
        problems.append(f"intent_catalog.PRIOR {prior!r} has no operation")

    for op in _ALL:
        if op.dispatch_mode not in DISPATCH_MODES:
            problems.append(
                f"operation {op.name!r} dispatches as {op.dispatch_mode!r}, "
                "which response_planner does not know")

    return problems


# Scorer names in PRIOR that are not themselves plan actions: a scorer is
# named for the QUESTION it recognises and may build a different action.
# Mapped rather than added as operations, because they are not separate
# things the system does.
_SCORER_ALIASES: dict[str, str] = {
    "hierarchy": "breakdown",          # _score_hierarchy builds breakdown/group_metric
    "advisor_profile": "lookup",       # builds action="lookup"
    "entity_summary": "summary",       # builds action="summary"
}
