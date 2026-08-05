"""THE owner of "what KIND of answer is this?".

Phase 3. Response-shape selection existed here already, but only as a
choice between formatter functions, and two of its four shapes rendered
identically. The mode the API reported was decided elsewhere entirely —
`chat_service` returned `"type": ir.intent`, passing the QUESTION's
structural shape through as the ANSWER's mode. Those are different
things, and conflating them produced:

    "What is Downtown revenue?"
      intent=leaderboard, one row  ->  "🏆 Top 1 by MTD Revenue Cleared"

The number and the scope were right by then (Phases 1 and 2). Only the
KIND of answer was wrong: a leaderboard of one is not a leaderboard.

WHAT WAS COMPETING FOR THIS DECISION

  chat_service._dispatch   `"type"` from resolution.kind, plan.action, or
                           ir.intent — 15 return sites
  response_planner         `shape`, consulted only by the formatter
  response_formatter       shape -> formatter, with single_value and
                           ranked_list mapped to the SAME function, so
                           the shape decision had no effect
  ir_validator             the capability registry, reachable only if the
                           intent was produced in the first place

This module is now the single owner. Everything downstream renders the
mode it chose; nothing re-derives it.

INPUTS are QueryIR, the aggregation result, and the capability registry —
never the raw user text. Response planning happens after understanding is
finished; re-reading the text here would be a second, competing parser.

CAPABILITY is read from ir_validator._UNSUPPORTED_INTENTS rather than
copied. That registry already carries the written reason for each
unsupported intent, and a second copy would drift from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from app.llm.query_compiler import effective_metric
from app.llm.query_ir import QueryIR

# The KIND of answer, not the shape of the question. `empty` and the
# capability states are modes in their own right: "no rows matched" and
# "this system cannot do that" are different answers, and collapsing them
# is how a missing capability came to look like missing data.
ResponseMode = Literal[
    "metric_value",       # one subject, one measure
    "leaderboard",        # a ranking over several subjects
    "comparison",         # subjects set side by side
    "breakdown",          # a group split into its parts
    "trend",              # movement over time
    "profile",            # everything about one subject
    "hierarchy_summary",  # a group's headline figures
    "roster",             # the members of a group, unranked
    "manager",            # who someone reports to
    "ancestry",           # the whole chain above someone
    "attendance",         # an attendance sweep
    "multi",              # several sub-answers stitched together
    "text",               # a canned conversational reply
    "clarification",      # we need one more thing from the user
    "unsupported",        # the system cannot answer this shape of question
    "not_found",          # the subject named does not exist
    "no_data",            # the query was fine; nothing matched
]

# Every mode the dispatcher may return. Declared here rather than inline
# at 39 return sites so a new response kind has to be introduced to the
# planner before it can reach the API — which is what stops the
# dispatcher quietly growing a second vocabulary of response types, as it
# had before Phase 4 (`roster`, `manager`, `advisor`, `team`, `company`,
# `unknown` all originated in chat_service and appeared nowhere else).
#
# The legacy names on the right are what the API has always emitted for
# these paths; they are preserved exactly, because renaming them is a
# breaking change for the frontend and this phase is about ownership, not
# about the wire format.
DISPATCH_MODES: dict[str, str] = {
    "metric_value": "metric_value",
    "leaderboard": "leaderboard",
    "comparison": "comparison",
    "breakdown": "breakdown",
    "profile": "advisor",
    "advisor_metric": "advisor_metric",
    "hierarchy_summary": "team",
    "company_summary": "company",
    "roster": "roster",
    "manager": "manager",
    "ancestry": "ancestry",
    "attendance": "attendance",
    "multi": "multi",
    "text": "text",
    "clarification": "clarification",
    "unsupported": "unsupported",
    "not_found": "not_found",
    "no_data": "unknown",
}

# Kept for backward compatibility: response_formatter dispatches on these
# and existing tests assert on them. `shape` is the RENDERING of a mode,
# which is why the two are separate — several modes render as a list.
Shape = Literal["single_value", "ranked_list", "comparison_table", "filtered_table", "empty"]

_MODE_TO_SHAPE: dict[str, Shape] = {
    "metric_value": "single_value",
    "leaderboard": "ranked_list",
    "comparison": "comparison_table",
    "breakdown": "filtered_table",
    "no_data": "empty",
}


@dataclass
class ResponsePlan:
    shape: Shape
    show_insights: bool
    # Whether a narrative explanation adds anything ahead of the rendered
    # answer. False for a single value: the explanation and the
    # single-value sentence state the same fact, so prepending it printed
    # "Downtown has 1,100 MTD Revenue Cleared." twice. A ranking is
    # different — there the explanation says WHY the order is what it is,
    # which the list itself does not.
    show_explanation: bool = True
    mode: ResponseMode = "leaderboard"
    why: str = ""
    # (mode, reason) for each alternative considered and dropped. The
    # observability requirement: a plan that records only its winner
    # cannot explain what it displaced.
    rejected: tuple[tuple[str, str], ...] = ()
    # Populated for `unsupported` — the registry's written explanation.
    reason: Optional[str] = None
    # The metric whose value completes this answer, when the ontology
    # pairs one (a count with its rate). Only ever set for a single
    # subject: on a ranking the companion would need a value per row and
    # the reply is already a list. The VALUE is fetched by the caller
    # from the aggregation engine — this names it, it does not compute it.
    companion_metric: Optional[str] = None

    def trace(self) -> str:
        parts = [f"{self.mode} ({self.why})"]
        parts += [f"not {m}: {r}" for m, r in self.rejected]
        return " | ".join(parts)


def respond(mode: str, reply, data=None, *, why: str = "", **extra) -> dict:
    """The one exit through which every response leaves the dispatcher.

    Phase 4. The dispatcher had 39 return sites, each writing its own
    `"type"` string inline, and only two of them consulted the response
    planner. That is a second owner of response selection, and it showed:
    `ir.intent` was passed through as the response type in two places,
    `unknown` was returned for three unrelated situations, and a
    capability limit was indistinguishable from an empty result.

    This does not move the dispatcher's BRANCHING here — which branch runs
    is a data question (does this advisor exist, did the compiler return
    rows) that belongs where the data is fetched. It moves the NAMING:
    every branch now declares a planner-known mode, is recorded in the
    trace with its reason, and is translated to the wire format in one
    place. A branch cannot invent a response type any more.
    """
    from app.llm import routing

    if mode not in DISPATCH_MODES:
        raise ValueError(
            f"{mode!r} is not a response mode. Add it to "
            f"response_planner.DISPATCH_MODES — the dispatcher may not "
            f"introduce response types of its own."
        )
    routing.decide("Response", mode, why or f"dispatched as {mode}")
    return {"type": DISPATCH_MODES[mode], "reply": reply, "data": data, **extra}


def _companion_for(ir: QueryIR) -> Optional[str]:
    """The paired metric, resolved to THIS query's period.

    Both steps are delegated. The pairing comes from the ontology; the
    period swap from query_compiler, which already owns "which sibling
    answers this measure at this period".

    Resolving the period is the whole point. The ontology names the MTD
    member of each family, so taking it verbatim produced the one state
    this must never reach: a YTD count reported beside an MTD rate, in a
    single sentence, with both labelled. `effective_metric` is used for
    the primary too, so the pair is always read off the metric that was
    actually EXECUTED rather than the one the IR was built with.
    """
    from app.llm.metric_ontology import METRICS, metric_for_period

    primary = effective_metric(ir)
    metric = METRICS.get(primary) if primary else None
    if metric is None or not metric.companion:
        return None

    period = ir.time_range.period if ir.time_range else None
    if period is None:
        return metric.companion
    # None here means the companion has no member for this period — a
    # YTD-only or DAILY-only gap. Reporting the MTD one anyway is the
    # mismatch this function exists to prevent, so the companion is
    # simply omitted and the primary answer stands alone.
    return metric_for_period(metric.companion, period)


def _capability_problem(ir: QueryIR) -> Optional[str]:
    """The registry's reason this intent cannot be served, or None.

    Read from ir_validator so there is one list of what this system
    cannot do, with one wording for each entry.
    """
    from app.llm.ir_validator import _UNSUPPORTED_INTENTS

    return _UNSUPPORTED_INTENTS.get(ir.intent)


def plan_response(ir: QueryIR, rows: list[dict]) -> ResponsePlan:
    """The one response-mode decision.

    Order matters and encodes precedence: a capability limit outranks
    everything (answering the wrong question well is worse than saying
    no), then emptiness, then the intent, and row count last.
    """
    rejected: list[tuple[str, str]] = []

    # ---- capability first -------------------------------------------
    problem = _capability_problem(ir)
    if problem is not None:
        return ResponsePlan(
            shape="empty", show_insights=False, mode="unsupported",
            why=f"{ir.intent!r} is not a shape this system can answer",
            reason=problem,
            rejected=(("no_data", "there are no rows because the query was never "
                                  "run, not because nothing matched"),),
        )

    # ---- no rows -----------------------------------------------------
    if not rows:
        return ResponsePlan(
            shape="empty", show_insights=False, mode="no_data",
            why="the query ran and matched nothing",
            rejected=(("unsupported", "the query shape IS supported — this is an "
                                      "empty result, not a missing capability"),),
        )

    if ir.intent == "comparison":
        return ResponsePlan(
            shape="comparison_table", show_insights=len(rows) >= 3,
            mode="comparison", why="the query names several subjects to set side by side",
        )

    if ir.intent == "filtered_list":
        return ResponsePlan(
            shape="filtered_table", show_insights=len(rows) >= 3,
            mode="breakdown", why="the query asks for the members matching a constraint",
        )

    # ---- leaderboard vs a single value -------------------------------
    # The distinction Phase 3 exists for. One row is one subject's figure,
    # however the question was phrased: after Phase 1 gave the named
    # subject ownership of the level, "Downtown revenue" produces exactly
    # one row, and rendering it as a ranking of one is the last place the
    # old "answers a different question" defect survived.
    if len(rows) == 1:
        return ResponsePlan(
            shape="single_value", show_insights=False, show_explanation=False,
            mode="metric_value",
            companion_metric=_companion_for(ir),
            why="the result is one subject's figure, not a ranking",
            rejected=(("leaderboard", "a ranking needs more than one subject to rank"),),
        )

    return ResponsePlan(
        shape="ranked_list", show_insights=len(rows) >= 3, mode="leaderboard",
        why=f"the result ranks {len(rows)} subjects",
        rejected=(("metric_value", "more than one subject came back, so there is no "
                                   "single figure to report"),),
    )
