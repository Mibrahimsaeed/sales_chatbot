"""
Semantic layer: canonical business-friendly metric names, their phrasing
synonyms, which entity levels each one supports, and — new in this
version — the actual column/table binding the generic query compiler
(query_compiler.py) needs to build SQL without a hand-written resolver
function per (metric, level) pair.

INVARIANT this file alone now upholds (sql_generator.py's RESOLVERS
registry is retired — see query_compiler.py): every (metric.key, level)
pair listed in entity_levels below MUST have a matching entry in
`bindings`. Add a metric here without a binding for a level you listed
in entity_levels and the compiler will treat that level as unsupported
for that metric (same fail-soft "I don't have a way to rank by that yet"
behavior as before, just declared in one place instead of two).

Adding a new metric:
1. Add a MetricDef entry with its synonyms and entity_levels.
2. Add one ColumnBinding per level in `bindings`.
No other pipeline change is required — the compiler reads this generically.
"""

import enum
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import case, func, literal

from app.llm import metric_aliases
from app.database.models import (
    Advisor, SalesFunnel, Pipeline, Performance, PerformancePeriod,
    TeamTarget, Portfolio, Calls, Attendance,
)


# An advisor "has a unit" when the tally is a real, non-zero value.
# Declared once, used as both numerator predicate and (via the ratio)
# the spec's `advisorsWithUnits`.
_HAS_UNIT = Advisor.unit.isnot(None) & Advisor.unit.notin_(["0", ""])


# Declared once and referenced by both levels of one_unit_ratio — see
# that metric's comment for why it appears twice.
_ONE_UNIT_BINDING = None  # populated below, after ColumnBinding exists


class Rollup(str, enum.Enum):
    """How a metric's per-advisor values combine into a GROUP value.

    Phase 4 made this an explicit, declared property. It used to be
    `ColumnBinding.agg` ("sum" | "avg"), which could express only two of
    the three real shapes — and the missing one is the one that matters:

    - SUM   additive quantities (connects, revenue, bookings).
    - AVG   a genuine mean of per-advisor values. Correct only when every
            advisor contributes equally.
    - RATIO the quantity is a ratio, so the GROUP value is the ratio of
            the summed components — not the mean of the ratios. A team of
            two advisors at 90% (target 1000) and 10% (target 100) is at
            910/1100 = 82.7%, not 50%. Averaging weighted a 100-target
            advisor exactly as heavily as a 1000-target one.
    """
    SUM = "sum"
    AVG = "avg"
    RATIO = "ratio"


@dataclass
class ColumnBinding:
    """Tells the compiler how to reach a metric's value for one entity level.

    - model:     the fact-table model holding the column/expression.
    - expr:      the SQLAlchemy column or computed expression on `model`
                 (e.g. SalesFunnel.mtd_cleared, or
                 SalesFunnel.mtd_new_connect + SalesFunnel.mtd_followup_connect).
    - team_named: True for tables like TeamTarget that are keyed directly by
                 team name (no advisor `wid` join, no roll-up possible —
                 the row IS the team-level truth). False means join `model`
                 to Advisor via `wid`, and any group level rolls it up
                 using the metric's `rollup` rule.
    - period:    for the shared `Performance` table, which PerformancePeriod
                 row this metric reads.
    """
    model: type
    expr: Any
    team_named: bool = False
    period: Optional[PerformancePeriod] = None
    # Phase 4 — required when the metric declares Rollup.RATIO. The group
    # value becomes SUM(ratio_numerator) / SUM(ratio_denominator), so the
    # numerator carries any scaling (e.g. * 100.0 for a percentage).
    # `expr` stays the PER-ADVISOR value and is what an advisor-level
    # query still selects.
    ratio_numerator: Any = None
    ratio_denominator: Any = None
    # FIX 3. Extra fact tables this binding's expression references
    # besides `model`, which the query builder must JOIN.
    #
    # Needed because a ratio can legitimately span two tables: Connect->CR
    # is client registrations (SalesFunnel) over answered calls (Calls).
    # Referencing a second model WITHOUT joining it does not fail — SQLAlchemy
    # adds it to the FROM clause as a CARTESIAN PRODUCT
    # ("FROM advisors JOIN sales_funnel ..., calls"), which silently
    # multiplies the denominator by the row count. Declaring the join here
    # keeps the requirement with the binding that creates it.
    join_models: tuple = ()
    # The denominator is `teamSize x daily_target_rate x workingDays`
    # rather than a stored column (CR %, Connect %, Meeting %). Set here
    # rather than inferred, because only the binding knows whether its
    # denominator is data or a target. aggregation.value_expression
    # builds the scaled denominator; the per-day figure stays declared on
    # MetricDef.daily_target_rate and the working days come from
    # working_days.for_period.
    working_day_scaled: bool = False



def round_half_up(value: float) -> float:
    """Round to the nearest integer, halves AWAY FROM ZERO.

    The dashboard is JavaScript, where `Math.round(84.5) === 85`.
    Python's built-in round() is banker's rounding, so `round(84.5) == 84`
    — a one-off that lands exactly on the 85 boundary and flips the
    status colour. Matching the dashboard means matching its rounding
    rule, not just the fact that it rounds.
    """
    return float(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class Thresholds:
    """Status bands for a metric's value.

    Two shapes, because not every board is a percentage:
    - BANDED: `green`/`yellow` are floors — >= green is green, >= yellow
      is yellow, below that red. Most metrics use 85/60.
    - PASS/FAIL: `green` is None and `zero_is_green` says the direction —
      a count metric like overdue is green only at 0.

    Declared per metric so an override (1-Unit's 45/30) is a value on the
    metric, not a branch in whatever renders it.
    """
    green: Optional[float] = None
    yellow: Optional[float] = None
    zero_is_green: bool = False

    @property
    def is_banded(self) -> bool:
        return self.green is not None

    def status(self, value: Optional[float]) -> Optional[str]:
        """"green" | "yellow" | "red", or None when the value is unknown.

        FIX 2 (dashboard parity). The dashboard computes
        `pct = round(...)` and bands the ROUNDED figure; this banded the
        raw one, so anything in [84.5, 85) showed yellow here and green
        there, and [59.5, 60) showed red here and yellow there. Same
        number, different colour on each surface.

        Rounding lives HERE and nowhere else — it is the single point
        where a value becomes a status, so there is exactly one
        implementation and no caller can forget it.
        """
        if value is None:
            return None
        value = round_half_up(value)
        if self.zero_is_green:
            return "green" if value == 0 else "red"
        if not self.is_banded:
            return "green" if value > 0 else "red"
        if value >= self.green:
            return "green"
        if self.yellow is not None and value >= self.yellow:
            return "yellow"
        return "red"


# The default banding, referenced rather than repeated per metric.
DEFAULT_THRESHOLDS = Thresholds(green=85.0, yellow=60.0)

# FIX 4 (dashboard parity). The spec's grounding prompt #3:
# "Overdue/Portfolio/Pipeline/Conversion are pass/fail on whether the
# total is >0, not percentage bands."
#
# Overdue already had its own inverted form (green only at zero). These
# three had `thresholds=None`, so status_for() returned None and the
# boards had no status at all — not a wrong colour, an absent one.
#
# Thresholds.status() already implements the shape: with no bands and
# zero_is_green False it returns green above zero and red at zero. This
# constant names it so the rule is declared once and read three times.
PASS_FAIL_POSITIVE = Thresholds()

# The 1-Unit board is the spec's one banded exception: "1-Unit uses
# 45/30" rather than the 85/60 every other percentage board uses.
ONE_UNIT_THRESHOLDS = Thresholds(green=45.0, yellow=30.0)


@dataclass
class MetricDef:
    """Everything the system needs to know about one metric.

    Phase 2 made this the single source of truth for metric BEHAVIOUR,
    not just its SQL binding. Sort direction, status bands, target rates
    and which periods a metric can answer were previously either absent
    or scattered — sort direction defaulted to descending everywhere
    (so "top 5 by overdue" ranked the worst advisors first and called
    them the top), and period support was implied by a hardcoded swap
    table in ir_patcher.

    PERIODS. A metric that exists at several periods is declared once per
    period as its own key, with the keys sharing a `period_family`:
    mtd_cleared / ytd_cleared / three_month_cleared all have
    family="cleared". `supported_periods` is DERIVED from the family, so
    the set of answerable periods can never drift from the bindings that
    answer them. A metric with no family answers only its own period —
    which is the honest position for everything sourced from SalesFunnel,
    whose columns are MTD-only.
    """
    key: str
    label: str
    entity_levels: list[str]          # levels with a binding below
    primary_level: str                # level used when the query doesn't specify one
    bindings: dict[str, ColumnBinding] = field(default_factory=dict)
    # DECLARED IN metric_aliases.py and injected below the METRICS table.
    # It keeps living on MetricDef because five modules read
    # `metric.synonyms`, but the ontology no longer OWNS the phrasings —
    # having each metric carry its own list is what hid the fact that
    # "answered calls %" and "answered calls" resolved to the same count.
    synonyms: list[str] = field(default_factory=list)

    # ---- period ----
    # The period this key reports. MTD for everything sourced from the
    # MTD-only fact tables; set explicitly for the Performance family.
    period: PerformancePeriod = PerformancePeriod.MTD
    # Keys that are the same measure at different periods share a family.
    # None means this metric has no period siblings.
    period_family: Optional[str] = None

    # ---- aggregation ----
    # How per-advisor values become a group value. Declared per metric so
    # the rule lives with the metric instead of being re-decided by each
    # service that rolls it up.
    rollup: Rollup = Rollup.SUM

    # ---- behaviour ----
    # True when a LOWER value is better (overdue, late arrivals). Drives
    # the default sort direction so a ranking cannot present the worst
    # performers as the leaders.
    lower_is_better: bool = False
    # True when the percentage IS attainment of an assigned target, so a
    # reply may talk about "the target" and "the goal".
    #
    # The narrative used to apply that wording to EVERY percentage
    # metric, so a 1-Unit ratio was reported as "has achieved 66.7% of
    # the assigned target, remaining 33.3% short of the monthly goal" —
    # inventing a target that does not exist for unit ownership, an
    # attendance rate or a funnel ratio. Declared per metric rather than
    # inferred, because only the metric knows what its denominator means.
    measures_target_attainment: bool = False
    # Status bands. None means this metric has no defined status.
    thresholds: Optional[Thresholds] = None
    # Per-advisor per-working-day target, where the business defines one
    # (connects 10/day, CR 2/day, meetings 0.6/day). Declared here so the
    # working-day scaling Phase 5 introduces has a single place to read
    # it from. None means no daily target is defined.
    daily_target_rate: Optional[float] = None
    # The metric that completes this one's story, when a count and a rate
    # are two readings of the same question. "How many CRs?" and "what is
    # the CR rate?" are both about client registrations, and a reply
    # giving only one of them leaves the obvious follow-up unasked.
    #
    # Declared here, on the metric, because which measures pair up is a
    # property of the BUSINESS, not of the response layer — and computed
    # by the aggregation engine like any other value, never re-derived by
    # a formatter. Symmetric by convention: if A names B, B names A.
    companion: Optional[str] = None


_ONE_UNIT_BINDING = ColumnBinding(
    model=Advisor,
    expr=case((_HAS_UNIT, 100.0), else_=0.0),
    ratio_numerator=case((_HAS_UNIT, 100.0), else_=0.0),
    # SUM(1) over the advisors in scope IS team size, so the spec's
    # `advisorsWithUnits / teamSize` needs no new concept.
    ratio_denominator=literal(1),
)

METRICS: dict[str, MetricDef] = {

    "achievement_pct": MetricDef(
        key="achievement_pct",
        # The one metric whose denominator IS an assigned target.
        measures_target_attainment=True,
        label="Target Achievement %",
        thresholds=DEFAULT_THRESHOLDS,
        # A group's achievement is its cleared over its target, not the
        # mean of its advisors' percentages.
        rollup=Rollup.RATIO,
        entity_levels=["advisor", "team"],
        primary_level="team",
        bindings={
            "team": ColumnBinding(model=TeamTarget, expr=TeamTarget.achievement_pct, team_named=True),
            "advisor": ColumnBinding(
                model=Performance,
                # COMPUTED, not the sheet's precomputed `pct` column.
                # Reading `Performance.pct` here while the group level
                # computed cleared/target gave the SAME advisor two
                # answers — 99% asked one way, 84.7% asked another —
                # whenever the sheet's percentage disagreed with its own
                # components. The spec computes (`round(Cleared / Target
                # x 100)`), so computing is also the parity-correct side.
                expr=Performance.cleared * 100.0 / func.nullif(Performance.target, 0),
                period=PerformancePeriod.MTD,
                ratio_numerator=Performance.cleared * 100.0,
                ratio_denominator=Performance.target,
            ),
        },
    ),

    "mtd_cleared": MetricDef(
        key="mtd_cleared",
        label="MTD Revenue Cleared",
        period=PerformancePeriod.MTD,
        period_family="cleared",
        entity_levels=["advisor", "team", "company"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Performance, expr=Performance.cleared, period=PerformancePeriod.MTD),
            "team": ColumnBinding(model=Performance, expr=Performance.cleared, period=PerformancePeriod.MTD),
            "company": ColumnBinding(model=Performance, expr=Performance.cleared, period=PerformancePeriod.MTD),
        },
    ),

    "ytd_cleared": MetricDef(
        key="ytd_cleared",
        label="YTD Revenue Cleared",
        period=PerformancePeriod.YTD,
        period_family="cleared",
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Performance, expr=Performance.cleared, period=PerformancePeriod.YTD),
        },
    ),

    "three_month_cleared": MetricDef(
        key="three_month_cleared",
        label="3-Month Revenue Cleared",
        period=PerformancePeriod.THREE_M,
        period_family="cleared",
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Performance, expr=Performance.cleared, period=PerformancePeriod.THREE_M),
        },
    ),

    "mtd_target": MetricDef(
        key="mtd_target",
        label="MTD Target",
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Performance, expr=Performance.target, period=PerformancePeriod.MTD),
            "team": ColumnBinding(model=TeamTarget, expr=TeamTarget.target, team_named=True),
        },
    ),

    "total_connects": MetricDef(
        key="total_connects",
        # Period family — the YTD sibling below answers the same measure
        # at year-to-date, so resolve_metric_for_period can swap them.
        period_family="connects",
        label="Total MTD Connects",
        daily_target_rate=10.0,
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_new_connect + SalesFunnel.mtd_followup_connect),
            "team": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_new_connect + SalesFunnel.mtd_followup_connect),
        },
    ),

    "new_connects": MetricDef(
        key="new_connects",
        # Period family — the YTD sibling below answers the same measure
        # at year-to-date, so resolve_metric_for_period can swap them.
        period_family="new_connects",
        label="New MTD Connects",
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_new_connect),
            "team": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_new_connect),
        },
    ),

    "followup_connects": MetricDef(
        key="followup_connects",
        # Period family — the YTD sibling below answers the same measure
        # at year-to-date, so resolve_metric_for_period can swap them.
        period_family="followup_connects",
        label="Follow-up MTD Connects",
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_followup_connect),
            "team": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_followup_connect),
        },
    ),

    "total_meetings": MetricDef(
        key="total_meetings",
        # Period family — the YTD sibling below answers the same measure
        # at year-to-date, so resolve_metric_for_period can swap them.
        period_family="meetings",
        label="Total MTD Meetings",
        daily_target_rate=0.6,
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_new_meeting + SalesFunnel.mtd_followup_meeting),
            "team": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_new_meeting + SalesFunnel.mtd_followup_meeting),
        },
    ),

    "conversion": MetricDef(
        key="conversion",
        # Period family — the YTD sibling below answers the same measure
        # at year-to-date, so resolve_metric_for_period can swap them.
        period_family="conversions",
        # Pass/fail on the total, per the spec — not a percentage band.
        thresholds=PASS_FAIL_POSITIVE,
        label="MTD Conversions",
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_conversion),
            # A count, despite the historical label: 3 + 1 conversions is
            # 4, and averaging reported 2 — not a number of anything.
            "team": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_conversion),
        },
    ),

    # STEP 4. "CR" is Client Registration — a real funnel stage with a
    # real column (SalesFunnel.mtd_cr) that had no metric, so "top
    # advisors by CR%" resolved to NOTHING and the ranking silently fell
    # back to revenue. It is the single most common measure in the KPI
    # spec's own example questions.
    #
    # A COUNT, not a rate. The spec's "CR rate" is
    # `cr / (teamSize * 2/advisor/day * workingDays) * 100`, which needs
    # working-day data this system does not have. The count is the
    # honest, computable answer, and the reply header names it ("Top 5 by
    # MTD Client Registrations") so the substitution is disclosed rather
    # than silent. Until the rate exists, "cr %" resolves here too — a
    # named, labelled measure of the right funnel stage beats both
    # "revenue" and a refusal.
    "client_registrations": MetricDef(
        key="client_registrations",
        companion="cr_rate",
        # Period family — the YTD sibling below answers the same measure
        # at year-to-date, so resolve_metric_for_period can swap them.
        period_family="client_registrations",
        label="MTD Client Registrations",
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_cr),
            "team": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_cr),
        },
    ),

    # ---------------------------------------------------------------
    # 1 Unit — advisors holding at least one unit, over team size.
    #
    # The only metric bound to `Advisor` itself rather than a fact table:
    # unit ownership is a property of the advisor row. Advisor is already
    # the query root, so no join is needed (see
    # query_compiler._join_fact_table).
    #
    # RATIO with a COUNT denominator. `SUM(1)` over the joined advisor
    # rows IS team size, so the spec's `advisorsWithUnits / teamSize`
    # falls out of the existing ratio machinery with no new concept.
    #
    # `unit` is a STRING tally ("0".."4" observed). Compared as a string
    # set rather than cast to a number: a cast would raise on any
    # non-numeric value the sheet might grow, and the set is exact.
    # ---------------------------------------------------------------
    # primary_level is TEAM, not advisor: one advisor's "ratio" is 0% or
    # 100%, which is not a meaningful answer. The board is a group
    # measure, so an unqualified "1 unit ratio" should rank teams.
    #
    # Both levels reference the SAME binding object. Since Phase 4 the
    # aggregation engine rolls the ADVISOR binding up for every group
    # level anyway, so the team entry exists to satisfy the
    # primary_level/entity_levels declaration rather than to describe a
    # second source — and sharing the object keeps that literal.
    "one_unit_ratio": MetricDef(
        key="one_unit_ratio",
        label="1 Unit Ratio %",
        thresholds=ONE_UNIT_THRESHOLDS,
        rollup=Rollup.RATIO,
        entity_levels=["advisor", "team"],
        primary_level="team",
        bindings={
            "advisor": _ONE_UNIT_BINDING,
            "team": _ONE_UNIT_BINDING,
        },
    ),

    # ---------------------------------------------------------------
    # WorksApp Login — the on-time login rate.
    #
    # Same shape as attendance_rate, now that the ETL imports
    # `login_mtd_not_marked`; before that the login half had no
    # not-marked column and could not form the same denominator.
    #
    # NOTE the spec's denominator is `teamSize` scaled by working days,
    # not recorded days. Matching it needs the working-day calendar,
    # which is out of scope — so this mirrors attendance_rate's shape
    # and carries the same known divergence.
    # ---------------------------------------------------------------
    "login_rate": MetricDef(
        key="login_rate",
        label="WorksApp Login Rate % (MTD)",
        thresholds=DEFAULT_THRESHOLDS,
        rollup=Rollup.RATIO,
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(
                model=Attendance,
                expr=Attendance.login_mtd_ontime * 100.0 / func.nullif(
                    Attendance.login_mtd_ontime + Attendance.login_mtd_late
                    + Attendance.login_mtd_not_marked, 0
                ),
                ratio_numerator=Attendance.login_mtd_ontime * 100.0,
                ratio_denominator=(
                    Attendance.login_mtd_ontime + Attendance.login_mtd_late
                    + Attendance.login_mtd_not_marked
                ),
            ),
        },
    ),

    # ---------------------------------------------------------------
    # IBD meetings. The COUNT of planned meetings, and the conduction
    # RATIO — the spec's `round(Conducted / Planned x 100)`.
    #
    # The spec's "Meetings Planned" BOARD is a target rate
    # (`planned / (teamSize x 1 x workingDays)`) and remains blocked on
    # the working-day calendar. The count below is what the data
    # supports today, and is the conduction ratio's denominator.
    # ---------------------------------------------------------------
    "meetings_planned": MetricDef(
        key="meetings_planned",
        label="MTD Meetings Planned",
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_meetings_planned),
        },
    ),

    "meetings_conducted": MetricDef(
        key="meetings_conducted",
        label="MTD Meetings Conducted",
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_meetings_conducted),
        },
    ),

    "meeting_conduction_rate": MetricDef(
        key="meeting_conduction_rate",
        label="Meeting Conduction % (MTD)",
        thresholds=DEFAULT_THRESHOLDS,
        rollup=Rollup.RATIO,
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(
                model=SalesFunnel,
                expr=SalesFunnel.mtd_meetings_conducted * 100.0 / func.nullif(
                    SalesFunnel.mtd_meetings_planned, 0
                ),
                ratio_numerator=SalesFunnel.mtd_meetings_conducted * 100.0,
                ratio_denominator=SalesFunnel.mtd_meetings_planned,
            ),
        },
    ),

    "bookings": MetricDef(
        key="bookings",
        # Period family — the YTD sibling below answers the same measure
        # at year-to-date, so resolve_metric_for_period can swap them.
        period_family="bookings",
        label="MTD Bookings Stored",
        # "cr booked" / "cr bookings" are multi-word on purpose so they
        # beat the bare "cr" above — _SYNONYM_INDEX is longest-first.
        # (They were originally multi-word because synonym matching was
        # plain substring and "cr" fired inside "concrete"/"increase";
        # Step 4 made that matching token-aware, so the reason is now the
        # ordering rather than the collision.)
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_booking_stored),
            "team": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_booking_stored),
        },
    ),

    "pipeline_value": MetricDef(
        key="pipeline_value",
        # Period family — the YTD sibling below answers the same measure
        # at year-to-date, so resolve_metric_for_period can swap them.
        period_family="pipeline",
        # Pass/fail on the total, per the spec — not a percentage band.
        thresholds=PASS_FAIL_POSITIVE,
        label="MTD Open Pipeline",
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Pipeline, expr=Pipeline.pipeline),
            "team": ColumnBinding(model=Pipeline, expr=Pipeline.pipeline),
        },
    ),

    "overdue": MetricDef(
        key="overdue",
        # Period family — the YTD sibling below answers the same measure
        # at year-to-date, so resolve_metric_for_period can swap them.
        period_family="overdue",
        label="MTD Overdue Pipeline",
        lower_is_better=True,
        thresholds=Thresholds(zero_is_green=True),
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Pipeline, expr=Pipeline.overdue),
            "team": ColumnBinding(model=Pipeline, expr=Pipeline.overdue),
        },
    ),

    # REMOVED: "overdue_amount". It bound the SAME column as `overdue`
    # (Pipeline.overdue <- the single "Total Overdue" sheet column) under
    # a different label — "MTD Overdue Amount" vs "MTD Overdue Pipeline".
    # A count and an amount are different quantities, so one of the two
    # labels was always wrong, and both returned the same number. The
    # sheet has no separate amount column, so there is one measure here,
    # not two. Its phrasings moved onto `overdue`.
    #
    # WHICH unit that column actually holds is not determinable from the
    # repository — the header is just "Total Overdue". The spec's board
    # is "Overdue count", so the surviving label follows the spec.

    "portfolio_value": MetricDef(
        key="portfolio_value",
        # Pass/fail on the total, per the spec — not a percentage band.
        thresholds=PASS_FAIL_POSITIVE,
        label="Portfolio Value",
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Portfolio, expr=Portfolio.value),
        },
    ),

    "returned_value": MetricDef(
        key="returned_value",
        label="Portfolio Returned",
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Portfolio, expr=Portfolio.returned),
        },
    ),

    "answered_calls": MetricDef(
        key="answered_calls",
        label="Answered Calls (MTD)",
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Calls, expr=Calls.answered_calls_mtd),
            "team": ColumnBinding(model=Calls, expr=Calls.answered_calls_mtd),
        },
    ),

    "late_count": MetricDef(
        key="late_count",
        label="Late Attendance Count (MTD)",
        lower_is_better=True,
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Attendance, expr=Attendance.biometric_mtd_late),
            "team": ColumnBinding(model=Attendance, expr=Attendance.biometric_mtd_late),
        },
    ),

    # ---------------------------------------------------------------
    # Funnel ratios (spec: Connect->CR, CR->Meeting, Meeting->Conversion)
    #
    # Real percentages, unlike the spec's target-RATE leaderboards: every
    # component is a column this system already stores, so no working-day
    # calendar is needed. Rollup.RATIO, so a group's ratio is the ratio of
    # summed components rather than the mean of per-advisor ratios — the
    # Phase 4 rule.
    #
    # These exist because "conversion rate" and "cr rate" previously
    # resolved to COUNTS. A rate that is computable should be computed;
    # one that is not is declared unavailable in metric_aliases.py.
    # ---------------------------------------------------------------
    # FIX 3. The spec's Connect->CR board is `CR / AnsweredCalls x 100`.
    # This divided by CONNECTS (mtd_new_connect + mtd_followup_connect),
    # a different and larger denominator, so every value was too low —
    # a silently wrong number rather than a refusal.
    "connect_to_cr_rate": MetricDef(
        key="connect_to_cr_rate",
        label="Connect to CR % (MTD)",
        thresholds=DEFAULT_THRESHOLDS,
        rollup=Rollup.RATIO,
        # advisor only: since Phase 4 the aggregation engine rolls the
        # ADVISOR binding up for every group level (see
        # aggregation.binding_for), so a duplicated team binding would
        # be dead weight — and a second copy of a ratio is exactly how
        # two answers to one question start.
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(
                model=SalesFunnel,
                expr=SalesFunnel.mtd_cr * 100.0 / func.nullif(
                    Calls.answered_calls_mtd, 0
                ),
                ratio_numerator=SalesFunnel.mtd_cr * 100.0,
                ratio_denominator=Calls.answered_calls_mtd,
                # The denominator lives on a different table — see
                # ColumnBinding.join_models.
                join_models=(Calls,),
            ),
        },
    ),

    "cr_to_meeting_rate": MetricDef(
        key="cr_to_meeting_rate",
        label="CR to Meeting % (MTD)",
        thresholds=DEFAULT_THRESHOLDS,
        rollup=Rollup.RATIO,
        # advisor only: since Phase 4 the aggregation engine rolls the
        # ADVISOR binding up for every group level (see
        # aggregation.binding_for), so a duplicated team binding would
        # be dead weight — and a second copy of a ratio is exactly how
        # two answers to one question start.
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(
                model=SalesFunnel,
                expr=(SalesFunnel.mtd_new_meeting + SalesFunnel.mtd_followup_meeting)
                * 100.0 / func.nullif(SalesFunnel.mtd_cr, 0),
                ratio_numerator=(SalesFunnel.mtd_new_meeting + SalesFunnel.mtd_followup_meeting) * 100.0,
                ratio_denominator=SalesFunnel.mtd_cr,
            ),
        },
    ),

    "meeting_to_conversion_rate": MetricDef(
        key="meeting_to_conversion_rate",
        label="Meeting to Conversion % (MTD)",
        thresholds=DEFAULT_THRESHOLDS,
        rollup=Rollup.RATIO,
        # advisor only: since Phase 4 the aggregation engine rolls the
        # ADVISOR binding up for every group level (see
        # aggregation.binding_for), so a duplicated team binding would
        # be dead weight — and a second copy of a ratio is exactly how
        # two answers to one question start.
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(
                model=SalesFunnel,
                expr=SalesFunnel.mtd_conversion * 100.0 / func.nullif(
                    SalesFunnel.mtd_new_meeting + SalesFunnel.mtd_followup_meeting, 0
                ),
                ratio_numerator=SalesFunnel.mtd_conversion * 100.0,
                ratio_denominator=SalesFunnel.mtd_new_meeting + SalesFunnel.mtd_followup_meeting,
            ),
        },
    ),

    # ---- WORKING-DAY SCALED RATES ----------------------------------
    # The spec measures these three against a target rather than against
    # stored data:
    #
    #     CR %      = CR            / (teamSize x 2   x workingDays) x 100
    #     Connect % = AnsweredCalls / (teamSize x 10  x workingDays) x 100
    #     Meeting % = Meetings      / (teamSize x 0.6 x workingDays) x 100
    #
    # All three shipped in metric_aliases.UNAVAILABLE — declared, refused
    # with a written reason, and answered with the underlying COUNT —
    # because `workingDays` had no source. working_days.py is now that
    # source, so the rates are computable and the refusals are gone.
    #
    # Each declares only what it owns: the numerator column, the per-
    # advisor-per-day target, and the period. teamSize is never named
    # here — aggregation.value_expression sums a per-row constant, so the
    # row count IS the team size, and no metric has to know how a group
    # is counted.
    "cr_rate": MetricDef(
        key="cr_rate",
        label="CR % (MTD)",
        companion="client_registrations",
        # A period FAMILY, so query_compiler._effective_metric() can swap
        # this for its year-to-date sibling. Without one, "CR % year to
        # date" found no member for YTD and refused — while ytd_cr sat in
        # the table with the data. The count metric has had a family
        # since it was written; the rate simply never joined one.
        period_family="cr_rate",
        thresholds=DEFAULT_THRESHOLDS,
        rollup=Rollup.RATIO,
        daily_target_rate=2.0,
        entity_levels=["advisor", "team"],
        primary_level="team",
        bindings={
            "advisor": ColumnBinding(
                model=SalesFunnel,
                expr=SalesFunnel.mtd_cr * 100.0,
                ratio_numerator=SalesFunnel.mtd_cr * 100.0,
                working_day_scaled=True,
            ),
            # aggregation.binding_for() rolls the ADVISOR binding up for
            # any group level, so this exists to satisfy the declared
            # entity_levels — deliberately the SAME expression, because a
            # second divergent one is how a percentage came to have two
            # answers before the engine was unified.
            "team": ColumnBinding(
                model=SalesFunnel,
                expr=SalesFunnel.mtd_cr * 100.0,
                ratio_numerator=SalesFunnel.mtd_cr * 100.0,
                working_day_scaled=True,
            ),
        },
    ),

    # The YTD sibling. Declared here rather than in _YTD_SPECS because
    # that table generates COUNTS — additive columns with no denominator
    # — and this is a working-day scaled RATE. Same period_family, so
    # _effective_metric swaps them; same daily target and the same
    # working_day_scaled flag, so aggregation computes it identically and
    # working_days.for_period returns the YEAR's working days off this
    # metric's own declared period.
    "ytd_cr_rate": MetricDef(
        key="ytd_cr_rate",
        label="CR % (YTD)",
        companion="ytd_client_registrations",
        thresholds=DEFAULT_THRESHOLDS,
        rollup=Rollup.RATIO,
        daily_target_rate=2.0,
        period=PerformancePeriod.YTD,
        period_family="cr_rate",
        entity_levels=["advisor", "team"],
        primary_level="team",
        bindings={
            "advisor": ColumnBinding(
                model=SalesFunnel,
                expr=SalesFunnel.ytd_cr * 100.0,
                ratio_numerator=SalesFunnel.ytd_cr * 100.0,
                working_day_scaled=True,
            ),
            "team": ColumnBinding(
                model=SalesFunnel,
                expr=SalesFunnel.ytd_cr * 100.0,
                ratio_numerator=SalesFunnel.ytd_cr * 100.0,
                working_day_scaled=True,
            ),
        },
    ),

    "answered_calls_rate": MetricDef(
        key="answered_calls_rate",
        label="Answered Calls % (MTD)",
        thresholds=DEFAULT_THRESHOLDS,
        rollup=Rollup.RATIO,
        daily_target_rate=10.0,
        entity_levels=["advisor", "team"],
        primary_level="team",
        bindings={
            "advisor": ColumnBinding(
                model=Calls,
                expr=Calls.answered_calls_mtd * 100.0,
                ratio_numerator=Calls.answered_calls_mtd * 100.0,
                working_day_scaled=True,
            ),
            # aggregation.binding_for() rolls the ADVISOR binding up for
            # any group level, so this exists to satisfy the declared
            # entity_levels — deliberately the SAME expression, because a
            # second divergent one is how a percentage came to have two
            # answers before the engine was unified.
            "team": ColumnBinding(
                model=Calls,
                expr=Calls.answered_calls_mtd * 100.0,
                ratio_numerator=Calls.answered_calls_mtd * 100.0,
                working_day_scaled=True,
            ),
        },
    ),

    "meeting_rate": MetricDef(
        key="meeting_rate",
        label="Meeting % (MTD)",
        thresholds=DEFAULT_THRESHOLDS,
        rollup=Rollup.RATIO,
        daily_target_rate=0.6,
        entity_levels=["advisor", "team"],
        primary_level="team",
        bindings={
            "advisor": ColumnBinding(
                model=SalesFunnel,
                expr=(SalesFunnel.mtd_new_meeting + SalesFunnel.mtd_followup_meeting) * 100.0,
                ratio_numerator=(
                    SalesFunnel.mtd_new_meeting + SalesFunnel.mtd_followup_meeting
                ) * 100.0,
                working_day_scaled=True,
            ),
            # See the note on cr_rate's team binding: same shape, rolled
            # up by the engine rather than computed differently here.
            "team": ColumnBinding(
                model=SalesFunnel,
                expr=(SalesFunnel.mtd_new_meeting + SalesFunnel.mtd_followup_meeting) * 100.0,
                ratio_numerator=(
                    SalesFunnel.mtd_new_meeting + SalesFunnel.mtd_followup_meeting
                ) * 100.0,
                working_day_scaled=True,
            ),
        },
    ),

    "attendance_rate": MetricDef(
        key="attendance_rate",
        label="Attendance Rate % (MTD)",
        thresholds=DEFAULT_THRESHOLDS,
        # On-time days over recorded days for the WHOLE group.
        rollup=Rollup.RATIO,
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            # NULLIF guards the advisor with zero attendance rows of any kind
            # (no on-time, no late, no not-marked) from dividing by zero —
            # their rate compiles to NULL, not a crash.
            "advisor": ColumnBinding(
                model=Attendance,
                expr=Attendance.biometric_mtd_ontime * 100.0 / func.nullif(
                    Attendance.biometric_mtd_ontime + Attendance.biometric_mtd_late + Attendance.biometric_mtd_not_marked, 0
                ),
                ratio_numerator=Attendance.biometric_mtd_ontime * 100.0,
                ratio_denominator=(
                    Attendance.biometric_mtd_ontime + Attendance.biometric_mtd_late
                    + Attendance.biometric_mtd_not_marked
                ),
            ),
            "team": ColumnBinding(
                model=Attendance,
                expr=Attendance.biometric_mtd_ontime * 100.0 / func.nullif(
                    Attendance.biometric_mtd_ontime + Attendance.biometric_mtd_late + Attendance.biometric_mtd_not_marked, 0
                ),
                ratio_numerator=Attendance.biometric_mtd_ontime * 100.0,
                ratio_denominator=(
                    Attendance.biometric_mtd_ontime + Attendance.biometric_mtd_late
                    + Attendance.biometric_mtd_not_marked
                ),
            ),
        },
    ),
}

# Longest synonym first, so e.g. "ytd cleared" matches before the bare word
# "cleared" could, and "new connects" before bare "connects".

# =====================================================================
# YTD siblings (tabs "YTD CCMC" and "YTD P1 & Overdue").
#
# Each pairs with its MTD metric through `period_family`, so
# resolve_metric_for_period() swaps them and a user asking for
# "connects year to date" gets ytd_connects rather than a refusal.
#
# Built from a table rather than 9 hand-written MetricDefs: they differ
# only by key, label, column and family, and writing them out would be
# nine chances for a column to be paired with the wrong family.
# =====================================================================

# (key, label, family, model, expr, thresholds, lower_is_better)
_YTD_SPECS: tuple[tuple, ...] = (
    ("ytd_connects", "Total YTD Connects", "connects", SalesFunnel,
     SalesFunnel.ytd_new_connect + SalesFunnel.ytd_followup_connect, None, False),
    ("ytd_new_connects", "New YTD Connects", "new_connects", SalesFunnel,
     SalesFunnel.ytd_new_connect, None, False),
    ("ytd_followup_connects", "Follow-up YTD Connects", "followup_connects", SalesFunnel,
     SalesFunnel.ytd_followup_connect, None, False),
    ("ytd_client_registrations", "YTD Client Registrations", "client_registrations", SalesFunnel,
     SalesFunnel.ytd_cr, None, False),
    ("ytd_meetings", "Total YTD Meetings", "meetings", SalesFunnel,
     SalesFunnel.ytd_new_meeting + SalesFunnel.ytd_followup_meeting, None, False),
    ("ytd_conversion", "YTD Conversions", "conversions", SalesFunnel,
     SalesFunnel.ytd_conversion, PASS_FAIL_POSITIVE, False),
    ("ytd_bookings", "YTD Bookings Stored", "bookings", SalesFunnel,
     SalesFunnel.ytd_booking_stored, None, False),
    ("ytd_pipeline_value", "YTD Open Pipeline", "pipeline", Pipeline,
     Pipeline.ytd_pipeline, PASS_FAIL_POSITIVE, False),
    # Overdue keeps its polarity and its green-only-at-zero rule at every
    # period — lower is better in January as much as in December.
    ("ytd_overdue", "YTD Overdue Pipeline", "overdue", Pipeline,
     Pipeline.ytd_overdue, Thresholds(zero_is_green=True), True),
)

for _key, _label, _family, _model, _expr, _thresholds, _lower_better in _YTD_SPECS:
    METRICS[_key] = MetricDef(
        key=_key,
        label=_label,
        entity_levels=["advisor"],
        primary_level="advisor",
        period=PerformancePeriod.YTD,
        period_family=_family,
        lower_is_better=_lower_better,
        thresholds=_thresholds,
        # period=None on the binding: unlike Performance, these live in
        # their own COLUMNS rather than period rows, so there is no
        # period column to filter on.
        bindings={"advisor": ColumnBinding(model=_model, expr=_expr)},
    )

# Synonyms are DECLARED in metric_aliases.py and injected here, so a
# phrasing is written down in exactly one place. MetricDef keeps the
# field (five modules read `metric.synonyms`) but no longer owns it —
# which is what let "answered calls %" and "answered calls" sit in
# separate lists nobody compared.
# The YTD count is generated by _YTD_SPECS, which carries no companion
# column — set here rather than widening that table for one entry.
METRICS["ytd_client_registrations"].companion = "ytd_cr_rate"

# Symmetry is the contract: a one-way pairing renders the companion on
# one phrasing of a question and not on its mirror.
for _a, _b in [(k, m.companion) for k, m in METRICS.items() if m.companion]:
    assert METRICS[_b].companion == _a, (  # pragma: no cover - declaration error
        f"{_a} names {_b} as its companion but {_b} names "
        f"{METRICS[_b].companion!r}"
    )

for _key, _metric in METRICS.items():
    _metric.synonyms = metric_aliases.phrases_for(_key)

# Every metric must be nameable, or it exists but cannot be asked for.
_UNNAMED = [k for k, m in METRICS.items() if not m.synonyms]
if _UNNAMED:  # pragma: no cover - a declaration error, caught at import
    raise RuntimeError(f"metrics with no alias in metric_aliases.py: {_UNNAMED}")


_SYNONYM_INDEX: list[tuple[str, str]] = sorted(
    ((syn, m.key) for m in METRICS.values() for syn in m.synonyms + [m.label.lower()]),
    key=lambda pair: -len(pair[0]),
)


def family_members(family: str) -> dict[PerformancePeriod, str]:
    """{period: metric key} for one period family, e.g. "cleared" ->
    {MTD: mtd_cleared, YTD: ytd_cleared, 3M: three_month_cleared}."""
    return {
        metric.period: metric.key
        for metric in METRICS.values()
        if metric.period_family == family
    }


def supported_periods(metric_key: str) -> tuple[PerformancePeriod, ...]:
    """Every period this metric can actually answer.

    DERIVED from the family rather than stored, so it cannot disagree
    with the bindings. A metric with no family answers only its own
    period — which is the truthful answer for anything sourced from a
    fact table that has no period dimension.
    """
    metric = METRICS.get(metric_key)
    if metric is None:
        return ()
    if metric.period_family is None:
        return (metric.period,)
    return tuple(family_members(metric.period_family))


def metric_for_period(metric_key: str, period: PerformancePeriod | str | None) -> str | None:
    """The metric key reporting `metric_key`'s measure at `period`.

    Returns the key unchanged when `period` is None or already correct;
    a sibling key when the family has one; and **None when the measure
    has no data for that period** — the case that used to be answered
    with MTD numbers under a YTD label, because ir_patcher's swap table
    covered only the cleared family and silently left every other metric
    alone.

    Callers must treat None as "cannot answer", not as "no change".
    """
    metric = METRICS.get(metric_key)
    if metric is None:
        return None
    if period is None:
        return metric_key
    if isinstance(period, str):
        try:
            period = PerformancePeriod(period)
        except ValueError:
            return None
    if metric.period == period:
        return metric_key
    if metric.period_family is None:
        return None
    return family_members(metric.period_family).get(period)


def measures_target_attainment(metric_key: str | None) -> bool:
    """Is this percentage attainment of an assigned target?

    Only such a metric may be described with target/goal language. Every
    other percentage — attendance, unit ownership, a funnel ratio — has a
    denominator that is not a target, and saying otherwise invents a
    business concept the data does not contain.
    """
    metric = METRICS.get(metric_key) if metric_key else None
    return bool(metric and metric.measures_target_attainment)


def lower_is_better(metric_key: str | None) -> bool:
    metric = METRICS.get(metric_key) if metric_key else None
    return bool(metric and metric.lower_is_better)


def thresholds_for(metric_key: str | None) -> Optional[Thresholds]:
    metric = METRICS.get(metric_key) if metric_key else None
    return metric.thresholds if metric else None


def status_for(metric_key: str | None, value: Optional[float]) -> Optional[str]:
    """"green" | "yellow" | "red", or None when the metric declares no
    bands. The single place status is computed."""
    bands = thresholds_for(metric_key)
    return bands.status(value) if bands else None


def daily_target_rate(metric_key: str | None) -> Optional[float]:
    metric = METRICS.get(metric_key) if metric_key else None
    return metric.daily_target_rate if metric else None


def resolve_metric_evidence(text: str) -> tuple[str, str] | None:
    """(metric key, the synonym that matched), or None.

    The matched synonym is returned because not every synonym is equally
    strong evidence that a SPECIFIC measure was requested — "performance"
    resolves achievement_pct, but "performance of X" is a question about
    the person, not about that one percentage. A caller that needs to
    weigh the phrasing (see query_planner._score_advisor_metric) cannot
    do so from the key alone.

    STEP 4: token-aware. This was `if synonym in q`, plain substring —
    the same class Step 1 removed from the keyword tables, missed here
    because no synonym was short enough to collide. "cr" is: it sits
    inside "across", "describe", "increase" and "acre". Longest-first
    ordering is unchanged, so a specific phrase still beats a bare word.

    Resolution goes through metric_aliases, the one registry that knows
    BOTH the available phrasings and the declared-but-uncomputable ones.
    Consulting it first is what stops "answered calls %" from falling
    through to the "answered calls" COUNT sitting inside it: the registry
    is ordered longest-phrase-first across every entry, available or not,
    so the rate is seen before the count.

    A known-but-unavailable measure returns None here — it names no
    metric — and the caller distinguishes it from an unknown phrase via
    metric_aliases.resolve(), which returns a match with a reason.
    """
    match = metric_aliases.resolve(text)
    if match is not None:
        return (match.metric, match.phrase) if match.available else None
    return None


def resolve_metric(text: str) -> str | None:
    evidence = resolve_metric_evidence(text)
    return evidence[0] if evidence else None


def metric_label(metric_key: str | None) -> str:
    """Shared by response_formatter.py and narrative.py so 'what do we call
    this metric in a reply' has one answer, not two independently
    maintained copies."""
    if not metric_key:
        return "value"
    return METRICS[metric_key].label if metric_key in METRICS else metric_key


# Acronyms that must stay upper-case when a label is lowered for prose.
_LABEL_ACRONYMS = ("MTD", "YTD", "3M", "%")


def metric_phrase(metric_key: str | None) -> str:
    """The metric named as it would be SPOKEN inside a sentence —
    "2 MTD connects", not "2 Total MTD Connects".

    Derived from the label rather than stored as a second field: a label
    is already curated per metric, and a parallel phrase list would be
    one more thing to keep in sync for no new information. The two
    transformations are the ones English needs — drop a leading "Total"
    (it is a table-header word, not a sentence word) and lower-case
    everything except the acronyms.
    """
    label = metric_label(metric_key)
    if label == "value":
        return label
    if label.lower().startswith("total "):
        label = label[len("total "):]
    return " ".join(
        word if word.upper() in _LABEL_ACRONYMS else word.lower()
        for word in label.split()
    )


def is_percentage_metric(metric_key: str | None) -> bool:
    """A metric is percentage-shaped if its own label says so ("%" in the
    label, e.g. achievement_pct's 'Target Achievement %') — a data-driven
    signal already curated in METRICS, not a second hand-maintained list
    that could drift out of sync. Adding "%" to a future metric's label is
    the only step needed to make narrative.py explain it as a percentage."""
    return bool(metric_key) and metric_key in METRICS and "%" in METRICS[metric_key].label


def describe_available_metrics() -> str:
    return ", ".join(f"{m.label.lower()} ({'/'.join(m.entity_levels)})" for m in METRICS.values())


def metric_catalog_for_prompt() -> str:
    """Grounds the LLM semantic parser in exactly what's queryable — same
    idea prompt_builder.py already used for teams/companies, extended to
    metrics so the model can't invent a metric key that has no binding.
    Emits ALL synonyms per metric (the whole catalog is only a few hundred
    tokens): with the LLM now the primary parser, the synonym list is the
    model's main phrasing-to-key lookup table, not a truncated hint."""
    lines = []
    for m in METRICS.values():
        lines.append(f"- {m.key}: {m.label} (levels: {', '.join(m.entity_levels)}; phrasings: {', '.join(m.synonyms)})")
    return "\n".join(lines)
