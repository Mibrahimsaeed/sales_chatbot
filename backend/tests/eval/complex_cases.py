"""Evaluation corpus: complex query understanding, stage by stage.

WHY TWO RUNS PER CASE. Final-answer accuracy cannot tell an LLM outage
apart from a defect in the IR, the validator or the compiler — every one
of them shows up as "wrong answer". So each case is evaluated twice:

  LIVE    the query through nlu_pipeline.resolve(), exactly what a user
          gets today
  ORACLE  a hand-built QueryIR expressing what the user MEANT, run
          through validate -> compile -> dispatch

The pair localises the failure. If ORACLE is right and LIVE is wrong, the
loss is upstream of the IR — the parser or the routing. If ORACLE cannot
be BUILT at all, the IR itself cannot represent the question. If it is
built and validation strips it, that is the validator. And so on down.

`oracle=None` is a finding, not an omission: it records a question the IR
has no shape for.

`truth` is computed by independent SQL against the same database, never
by asking the system under test what it thinks the answer is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from app.llm.query_ir import (
    Filter, FilterGroup, MetricRef, QueryIR, Sort, Subject,
)


@dataclass
class EvalCase:
    category: str
    query: str
    # What the user means, in words — the yardstick for "was the meaning
    # understood", which no assertion on rows can capture by itself.
    meaning: str
    # The IR that correctly expresses `meaning`, or None when the IR has
    # no shape for it. None is a RESULT.
    oracle: Optional[QueryIR] = None
    # (sql, params) -> the set/scalar the answer must match. None skips
    # the answer check (used where the case is about representation).
    truth_sql: Optional[str] = None
    truth_params: dict = field(default_factory=dict)
    # How to reduce compiled rows to something comparable with `truth`.
    rows_to_answer: Optional[Callable] = None
    # What the LIVE run SHOULD do. "answer" for questions the system can
    # serve some faithful reading of; "refuse" where no path can express
    # the question, so producing a confident reply is a SILENT WRONG
    # ANSWER rather than a success. Without this, "it replied" scores the
    # same as "it replied correctly", which is the exact confusion the
    # recent safety work exists to remove.
    live_expectation: str = "answer"
    notes: str = ""


def _names(rows):
    return sorted(str(r.get("name")) for r in rows)


def _count(rows):
    return len(rows)


_MASTER = "a.in_master_sheet"

# Connects live on calls.connects_mtd (Phase 17 repointed the family).
_TEAM_CONNECTS = f"""
select a.team name, sum(coalesce(cl.connects_mtd,0)) v
from advisors a left join calls cl on cl.wid = a.wid
where {_MASTER} group by a.team
"""

_BCM_SIZE = f"""
select a.management_lead name, count(*) v
from advisors a where {_MASTER} and a.management_lead is not null
  and a.management_lead not in (select distinct portfolio_lead from advisors where portfolio_lead is not null)
  and a.management_lead not in (select distinct rm from advisors where rm is not null)
group by a.management_lead
"""


def _leaderboard(level, metric, **kw):
    return QueryIR(intent="leaderboard", operation="leaderboard",
                   subject_level=level, metric=MetricRef(key=metric),
                   sort=Sort(metric=metric, direction="desc"), limit=None, **kw)


CASES: list[EvalCase] = [

    # ------------------------------------------------ multiple metrics
    EvalCase(
        category="multiple metrics",
        query="connects and answered calls of all BCMs",
        meaning="every BCM, with BOTH measures shown per row",
        oracle=_leaderboard("bcm", "total_connects",
                            metrics=[MetricRef(key="total_connects"),
                                     MetricRef(key="answered_calls")]),
        notes="P0 widened the IR to metrics[]; the question is whether "
              "anything produces it and whether both reach the table.",
    ),

    # --------------------------------------------- multiple conditions
    EvalCase(
        category="multiple conditions (AND)",
        query="advisors with target achievement below 50% and answered calls % below 20%",
        meaning="advisors satisfying BOTH thresholds, each on its own measure",
        oracle=QueryIR(
            intent="filtered_list", operation="filtered_list",
            subject_level="advisor", metric=MetricRef(key="achievement_pct"),
            filters=[Filter(field="achievement_pct", operator="<", value=50),
                     Filter(field="answered_calls_rate", operator="<", value=20)],
            sort=Sort(metric="achievement_pct", direction="desc"), limit=None),
        rows_to_answer=_count,
    ),

    EvalCase(
        category="AND",
        query="BCMs with team size greater than 5 and connects over 1000",
        meaning="BCMs satisfying both group-level conditions",
        oracle=QueryIR(
            intent="filtered_list", operation="filtered_list",
            subject_level="bcm", metric=MetricRef(key="team_size"),
            filters=[Filter(field="team_size", operator=">", value=5),
                     Filter(field="total_connects", operator=">", value=1000)],
            sort=Sort(metric="team_size", direction="desc"), limit=None),
        rows_to_answer=_count,
    ),

    # ------------------------------------------------------------ OR
    EvalCase(
        category="OR",
        query="advisors in Blue Area or DownTown",
        meaning="the union of two teams — NOT their intersection",
        oracle=_leaderboard(
            "advisor", "total_connects",
            filter_tree=FilterGroup(op="or", children=[
                Filter(field="team", operator="=", value="Blue Area"),
                Filter(field="team", operator="=", value="DownTown")])),
        truth_sql=f"select count(*) v from advisors a where {_MASTER} "
                  "and (a.team ilike 'Blue Area' or a.team ilike 'DownTown') "
                  "and exists (select 1 from calls cl where cl.wid = a.wid)",
        rows_to_answer=_count,
        notes="Truth models the metric JOIN: a connects ranking only "
              "contains advisors who have a calls row. See the roster-gap "
              "case below for why that matters for this wording.",
    ),

    EvalCase(
        category="OR",
        query="teams called Blue Area or DownTown ranked by connects",
        meaning="a two-team ranking, union semantics",
        oracle=_leaderboard(
            "team", "total_connects",
            filter_tree=FilterGroup(op="or", children=[
                Filter(field="team", operator="=", value="Blue Area"),
                Filter(field="team", operator="=", value="DownTown")])),
        truth_sql=f"select count(distinct a.team) v from advisors a where {_MASTER} "
                  "and (a.team ilike 'Blue Area' or a.team ilike 'DownTown')",
        rows_to_answer=_count,
    ),

    # ------------------------------------------------ NOT / exclusion
    EvalCase(
        category="NOT / exclusion",
        query="all advisors excluding Blue Area",
        meaning="every advisor whose team is not Blue Area",
        oracle=_leaderboard(
            "advisor", "total_connects",
            filter_tree=FilterGroup(op="not", children=[
                Filter(field="team", operator="=", value="Blue Area")])),
        truth_sql=f"select count(*) v from advisors a where {_MASTER} "
                  "and a.team not ilike 'Blue Area' "
                  "and exists (select 1 from calls cl where cl.wid = a.wid)",
        rows_to_answer=_count,
        notes="Same: truth models the metric JOIN.",
    ),

    EvalCase(
        category="NOT / exclusion",
        query="teams by connects excluding Blue Area and DownTown",
        meaning="rank teams, leaving two out",
        oracle=_leaderboard(
            "team", "total_connects",
            filter_tree=FilterGroup(op="and", children=[
                FilterGroup(op="not", children=[
                    Filter(field="team", operator="=", value="Blue Area")]),
                FilterGroup(op="not", children=[
                    Filter(field="team", operator="=", value="DownTown")])])),
        truth_sql=f"select count(distinct a.team) v from advisors a where {_MASTER} "
                  "and a.team not ilike 'Blue Area' and a.team not ilike 'DownTown'",
        rows_to_answer=_count,
    ),

    EvalCase(
        category="NOT / exclusion",
        query="list the advisors excluding Blue Area",
        meaning="ENUMERATE advisors, minus one team — no ranking implied",
        oracle=None,
        notes="THE GAP the two cases above exposed. A roster can enumerate "
              "but cannot express OR/NOT (it is PLAN_ONLY, with a single "
              "entity filter). The IR can express OR/NOT but every IR needs "
              "a sort metric, whose fact-table join silently drops the 13 "
              "advisors with no calls row — 518 becomes 507. Neither path "
              "answers the question faithfully.",
        truth_sql=f"select count(*) v from advisors a where {_MASTER} "
                  "and a.team not ilike 'Blue Area'",
    ),

    # ---------------------------------------------------- comparisons
    EvalCase(
        category="comparison",
        query="compare Blue Area and DownTown by connects",
        meaning="two named teams side by side on one measure",
        oracle=QueryIR(
            intent="comparison", operation="comparison", subject_level="team",
            subjects=[Subject(type="team", value="Blue Area"),
                      Subject(type="team", value="DownTown")],
            metric=MetricRef(key="total_connects"),
            sort=Sort(metric="total_connects", direction="desc"), limit=None),
        rows_to_answer=_count,
    ),

    # ------------------------- different metrics for different subjects
    EvalCase(
        category="different metrics per subject",
        query="Blue Area's connects and DownTown's revenue",
        meaning="one measure for one subject, a DIFFERENT measure for the other",
        oracle=QueryIR(
            intent="comparison", operation="comparison", subject_level="team",
            subjects=[Subject(type="team", value="Blue Area",
                              metric=MetricRef(key="total_connects")),
                      Subject(type="team", value="DownTown",
                              metric=MetricRef(key="mtd_cleared"))],
            metric=MetricRef(key="total_connects"),
            sort=Sort(metric="total_connects", direction="desc"), limit=None),
        notes="P0 added Subject.metric. Does anything COMPILE it?",
    ),

    # ----------------------------------------------------- hierarchy
    EvalCase(
        category="hierarchy",
        query="how many advisors directly report to the Unit Head in AMD",
        meaning="the unit head of team AMD, then HIS immediate advisor reports",
        oracle=None,
        notes="No IR intent expresses a direct-reports enumeration — "
              "operations.PLAN_ONLY records this.",
        truth_sql="select count(*) v from advisors a where a.in_master_sheet "
                  "and a.management_lead ilike (select distinct rm from advisors "
                  "where in_master_sheet and team ilike 'AMD' limit 1) "
                  "and lower(a.name) <> lower(a.management_lead)",
    ),

    EvalCase(
        category="hierarchy",
        query="all advisors in AMD",
        meaning="enumerate one team's members",
        oracle=None,
        notes="roster is PLAN_ONLY by declaration.",
        truth_sql=f"select count(*) v from advisors a where {_MASTER} and a.team ilike 'AMD'",
    ),

    # ---------------------------------------------- complex ranking
    EvalCase(
        category="complex ranking",
        query="top 3 teams by connects excluding Blue Area",
        meaning="a ranking, capped, with one member removed",
        oracle=QueryIR(
            intent="leaderboard", operation="leaderboard", subject_level="team",
            metric=MetricRef(key="total_connects"),
            filter_tree=FilterGroup(op="not", children=[
                Filter(field="team", operator="=", value="Blue Area")]),
            sort=Sort(metric="total_connects", direction="desc"), limit=3),
        rows_to_answer=_names,
    ),

    EvalCase(
        category="complex ranking",
        query="bottom 5 BCMs by team size with more than 2 people",
        meaning="ascending ranking, capped, with a group-level condition",
        oracle=QueryIR(
            intent="leaderboard", operation="leaderboard", subject_level="bcm",
            metric=MetricRef(key="team_size"),
            filters=[Filter(field="team_size", operator=">", value=2)],
            sort=Sort(metric="team_size", direction="asc"), limit=5),
        rows_to_answer=_count,
    ),

    # ------------------------------------------------ time comparison
    EvalCase(
        category="time comparison",
        query="revenue this month versus year to date",
        meaning="the same measure at two periods, set against each other",
        oracle=None,
        live_expectation="refuse",
        notes="time_range.compare_to is representable and READABLE after "
              "P0, but no executor runs the two-period pair — so answering "
              "means silently reporting ONE period.",
    ),

    # ------------------------------------------------- multi-part
    EvalCase(
        category="multi-part",
        query="who is the top advisor in Blue Area and what is their team size",
        meaning="two questions: a ranking, then a property of its winner",
        oracle=None,
        live_expectation="refuse",
        notes="Needs steps[] — deferred from P0 by design. Answering means "
              "silently dropping the second half of the question.",
    ),
]


# Multi-turn cases carry a conversation rather than one query.
@dataclass
class ChainCase:
    category: str
    turns: list[str]
    meaning: str
    expect_final: Callable  # (resolution, response) -> bool | str
    notes: str = ""


CHAINS: list[ChainCase] = [
    ChainCase(
        category="multi-turn follow-up",
        turns=["top 5 advisors by connects", "what about revenue"],
        meaning="the second turn keeps the ranking and swaps the measure",
        expect_final=lambda res, resp: (
            res.ir is not None and res.ir.metric
            and res.ir.metric.key in ("mtd_cleared", "ytd_cleared")),
    ),
    ChainCase(
        category="multi-turn follow-up",
        turns=["compare Blue Area and DownTown", "what about connects"],
        meaning="the second turn keeps BOTH subjects and swaps the measure",
        expect_final=lambda res, resp: (
            res.ir is not None and len(res.ir.subjects) == 2
            and res.ir.resolved_operation() == "comparison"),
    ),
    ChainCase(
        category="multi-turn follow-up",
        turns=["connects of Blue Area", "and DownTown"],
        meaning="the second turn swaps the subject and keeps the measure",
        expect_final=lambda res, resp: (
            res.ir is not None and res.ir.metric
            and res.ir.metric.key == "total_connects"),
    ),
]
