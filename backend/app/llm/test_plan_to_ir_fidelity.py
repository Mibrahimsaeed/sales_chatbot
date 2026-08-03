"""Step 2 — extracted meaning survives the trip to the compiler.

Entity extraction was already correct. The loss happened in the middle:
`QueryPlan` is a flat struct with no field for a period or a threshold,
so anything the extractor found that the struct could not hold was
dropped on the way to QueryIR.

  F4  "top advisors by revenue year to date"
      entities["period"] == "YTD", and plan_to_ir then set
      time_range from the METRIC's own period — overwriting the user's
      words with MTD. Phase 2's resolve_metric_for_period() was already
      capable of swapping mtd_cleared -> ytd_cleared; it was simply
      never told which period was wanted.

  F8  "advisors with achievement above 80 percent"
      entities["thresholds"] == [{">": 80.0}], and plan_to_ir built
      filters from team/company/level/attendance_status only. The
      threshold vanished and the reply listed everybody, including
      advisors at 10%.

Both were SILENT: a plausible, well-formatted answer to a different
question. These tests pin the whole chain — extraction, plan, IR, and the
rows the compiler actually returns — because an IR that merely holds the
right field is not the same as an answer that respects it.
"""

import pytest

from app.database.models import (
    Advisor, Performance, PerformancePeriod, SalesFunnel,
)
from app.llm import entity_extractor
from app.llm.entity_extractor import extract_entities
from app.llm.preprocessing import normalize
from app.llm.query_compiler import compile_and_run
from app.llm.query_ir import plan_to_ir
from app.llm.query_planner import build_query_plan


@pytest.fixture()
def org(db_session):
    """MTD and YTD deliberately rank the SAME advisors in the SAME order
    but with different magnitudes, so a period mix-up shows up as wrong
    numbers rather than as a wrong ordering that a sort test would catch
    anyway. Achievement spans the 80% threshold: two above, three below.
    """
    rows = [
        # wid, name,  mtd_cleared, ytd_cleared, pct
        (1, "Adv One",    100,  1000,  95.0),
        (2, "Adv Two",     90,   900,  85.0),
        (3, "Adv Three",   80,   800,  50.0),
        (4, "Adv Four",    70,   700,  30.0),
        (5, "Adv Five",    60,   600,  10.0),
    ]
    for wid, name, mtd, ytd, pct in rows:
        db_session.add(Advisor(wid=wid, name=name, team="Blue Area",
                               company="Graana", in_master_sheet=True))
        # target is set so that cleared/target x 100 == pct. achievement
        # is COMPUTED from the components now (it used to read the sheet's
        # `pct` column at advisor level and compute at group level, giving
        # one advisor two answers), so a fixture whose pct disagrees with
        # its own components no longer describes anything real.
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=mtd / pct * 100, cleared=mtd, pct=pct))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.YTD,
                                   target=1000, cleared=ytd, pct=pct))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=wid, mtd_followup_connect=0))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    return db_session


def _ir(text: str, db):
    cleaned = normalize(text)
    entities = extract_entities(cleaned, db)
    return plan_to_ir(build_query_plan(cleaned, entities), entities), entities


# ---------------------------------------------------------------------
# F4 — the period the user named
# ---------------------------------------------------------------------

def test_ytd_revenue_ranking_carries_the_period_the_user_asked_for(org):
    """The required case. 'top 3 advisors by YTD revenue'."""
    ir, entities = _ir("top 3 advisors by YTD revenue", org)

    assert entities["period"] == "YTD"
    assert ir.time_range.period == "YTD"
    assert ir.limit == 3
    assert ir.sort.direction == "desc"

    rows = compile_and_run(org, ir)
    assert [r["name"] for r in rows] == ["Adv One", "Adv Two", "Adv Three"]
    assert rows[0]["value"] == 1000      # YTD, not the 100 MTD figure


@pytest.mark.parametrize("phrasing", [
    "top advisors by revenue year to date",
    "top advisors by revenue this year",
    "top advisors by ytd revenue",
])
def test_every_ytd_phrasing_reaches_the_compiler(org, phrasing):
    """The audit case used 'year to date', which used to lose the period
    while 'ytd revenue' happened to survive — not because periods were
    handled, but because that exact string is a metric synonym. All
    phrasings must now go through the same path."""
    ir, _ = _ir(phrasing, org)
    assert ir.time_range.period == "YTD", phrasing
    assert compile_and_run(org, ir)[0]["value"] == 1000, phrasing


def test_the_metric_no_longer_overwrites_a_stated_period(org):
    """The actual defect: plan_to_ir set time_range from the metric's own
    period, so a metric that reports MTD forced the whole IR to MTD even
    when the user said otherwise."""
    ir, _ = _ir("top advisors by revenue year to date", org)

    # "revenue" resolves to the MTD-flavoured key...
    assert ir.sort.metric == "mtd_cleared"
    # ...and the user's period still wins; the compiler resolves the pair.
    assert ir.time_range.period == "YTD"


def test_a_query_naming_no_period_still_uses_the_metrics_own(org):
    """Backward compatibility. With nothing stated, the metric's period
    remains the answer — this is what every existing caller relies on."""
    ir, entities = _ir("top advisors by revenue", org)

    assert "period" not in entities
    assert ir.time_range.period == "MTD"
    assert compile_and_run(org, ir)[0]["value"] == 100


def test_an_explicitly_mtd_query_is_unchanged(org):
    ir, _ = _ir("top advisors by revenue this month", org)
    assert ir.time_range.period == "MTD"
    assert compile_and_run(org, ir)[0]["value"] == 100


def test_the_plan_itself_carries_the_period(org):
    """The struct, not just the IR — this is the field whose absence was
    the root cause."""
    cleaned = normalize("top advisors by revenue year to date")
    plan = build_query_plan(cleaned, extract_entities(cleaned, org))
    assert plan.period == "YTD"


def test_a_period_survives_on_every_action_not_only_leaderboard(org):
    """Carried centrally in build_query_plan, so an intent that grows a
    period-aware dispatch later inherits it rather than re-deriving it."""
    cleaned = normalize("how is Blue Area doing this year")
    plan = build_query_plan(cleaned, extract_entities(cleaned, org))
    assert plan.action == "summary"
    assert plan.period == "YTD"


# ---------------------------------------------------------------------
# F8 — the threshold the user set
# ---------------------------------------------------------------------

def test_achievement_threshold_becomes_a_filter(org):
    """The required case. 'Show advisors above 80% achievement'."""
    ir, entities = _ir("Show advisors above 80% achievement", org)

    assert entities["thresholds"] == [{"operator": ">", "value": 80.0}]

    threshold_filters = [f for f in ir.filters if f.field == "achievement_pct"]
    assert len(threshold_filters) == 1
    assert threshold_filters[0].operator == ">"
    assert threshold_filters[0].value == 80.0


def test_the_threshold_actually_narrows_the_answer(org):
    """The IR holding the filter is not the point — the rows are. This
    used to return all five advisors, including the one at 10%."""
    ir, _ = _ir("advisors with achievement above 80 percent", org)
    rows = compile_and_run(org, ir)

    assert [r["name"] for r in rows] == ["Adv One", "Adv Two"]
    assert all(r["value"] > 80 for r in rows)


@pytest.mark.parametrize("phrasing,operator,expected", [
    ("advisors with achievement above 80 percent", ">", ["Adv One", "Adv Two"]),
    ("advisors with achievement over 80 percent", ">", ["Adv One", "Adv Two"]),
    ("advisors with at least 85 percent achievement", ">=", ["Adv One", "Adv Two"]),
    ("advisors with achievement below 40 percent", "<", ["Adv Four", "Adv Five"]),
    ("advisors with at most 30 percent achievement", "<=", ["Adv Four", "Adv Five"]),
])
def test_every_comparator_survives_to_the_rows(org, phrasing, operator, expected):
    ir, _ = _ir(phrasing, org)
    assert [f.operator for f in ir.filters if f.field == "achievement_pct"] == [operator]
    assert sorted(r["name"] for r in compile_and_run(org, ir)) == sorted(expected)


def test_two_thresholds_both_survive(org):
    """"between 40 and 90" style banding — filters are AND-combined, so
    dropping either one widens the answer silently."""
    ir, _ = _ir("advisors with achievement above 40 percent and below 90 percent", org)

    operators = sorted(f.operator for f in ir.filters if f.field == "achievement_pct")
    assert operators == ["<", ">"]
    assert sorted(r["name"] for r in compile_and_run(org, ir)) == ["Adv Three", "Adv Two"]


def test_the_plan_itself_carries_the_thresholds(org):
    cleaned = normalize("advisors with achievement above 80 percent")
    plan = build_query_plan(cleaned, extract_entities(cleaned, org))
    assert plan.thresholds == [{"operator": ">", "value": 80.0}]


def test_a_threshold_needs_a_metric_to_attach_to(org):
    """A bare number with no resolved measure has nothing to filter on.
    It must not become a filter against some arbitrary field — that would
    turn an unanswerable query into a confident wrong answer, which is
    the failure mode this whole step exists to remove."""
    ir, _ = _ir("top advisors by revenue", org)
    assert [f for f in ir.filters if f.field == "achievement_pct"] == []


# ---------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------

def test_entity_filters_are_unchanged(org):
    ir, _ = _ir("top advisors by revenue in Blue Area", org)
    team_filters = [f for f in ir.filters if f.field == "team"]
    assert [(f.operator, f.value) for f in team_filters] == [("=", "Blue Area")]


def test_period_and_threshold_and_entity_filters_coexist(org):
    """All three kinds of constraint on one query. Revenue is used
    because it HAS a YTD sibling — see the next test for what happens
    when the measure doesn't."""
    ir, _ = _ir("top advisors in Blue Area with revenue above 700 this year", org)

    assert ir.time_range.period == "YTD"
    # The threshold follows the period too — ytd_cleared, not mtd_cleared,
    # or the query would rank one way and filter another.
    assert {f.field for f in ir.filters} == {"team", "ytd_cleared"}
    assert [r["name"] for r in compile_and_run(org, ir)] == ["Adv One", "Adv Two", "Adv Three"]


def test_a_period_the_metric_cannot_answer_refuses_instead_of_guessing(org):
    """A CONSEQUENCE of this step worth pinning deliberately.

    achievement_pct exists for MTD only — there is no YTD achievement
    metric. Before F4 was fixed the period never reached the compiler, so
    "achievement above 80% this year" quietly returned MTD numbers under
    a year-to-date question. Now the period arrives,
    resolve_metric_for_period() returns None, and compile_and_run()
    returns None — which the caller renders as "I can't answer that yet".

    Refusing is the right answer here: the alternative is the silent
    wrong one this whole remediation exists to remove.
    """
    from app.llm.metric_ontology import supported_periods
    from app.database.models import PerformancePeriod

    assert supported_periods("achievement_pct") == (PerformancePeriod.MTD,)

    ir, _ = _ir("advisors with achievement above 80 percent this year", org)
    assert ir.time_range.period == "YTD"
    assert compile_and_run(org, ir) is None


def test_a_deliberately_cross_period_filter_is_not_rewritten(org):
    """The boundary of the period resolution above, and the reason it
    lives in plan_to_ir rather than the compiler.

    "Rank by MTD revenue, but only advisors above 1500 YTD" is a real
    query. Its filter names a period-specific metric ON PURPOSE. By the
    time the compiler sees a Filter(field="ytd_cleared") it cannot tell
    that apart from a threshold that merely inherited the window, so
    re-resolving there would silently rewrite the deliberate one. An
    extracted threshold is resolved where the difference is still known;
    a hand-authored filter is left exactly as written.
    """
    from app.llm.query_ir import Filter, MetricRef, QueryIR, Sort

    ir = QueryIR(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="mtd_cleared"),
        filters=[Filter(field="ytd_cleared", operator=">", value=850)],
        sort=Sort(metric="mtd_cleared", direction="desc"),
    )
    rows = compile_and_run(org, ir)

    # Ranked on MTD, filtered on YTD — two different windows, both honoured.
    assert [r["name"] for r in rows] == ["Adv One", "Adv Two"]
    assert rows[0]["value"] == 100


def test_a_hand_built_plan_without_the_new_fields_still_works(org):
    """The new fields are optional with defaults, so any caller that
    builds a QueryPlan directly — tests, ir_patcher, older code — is
    unaffected."""
    from app.llm.query_planner import QueryPlan

    plan = QueryPlan(action="leaderboard", level="advisor", metric="mtd_cleared", limit=2)
    assert plan.period is None
    assert plan.thresholds == []

    ir = plan_to_ir(plan, {})
    assert ir.time_range.period == "MTD"
    assert ir.filters == []
    assert len(compile_and_run(org, ir)) == 2
