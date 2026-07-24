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

from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import func

from app.database.models import (
    Advisor, SalesFunnel, Pipeline, Performance, PerformancePeriod,
    TeamTarget, Portfolio, Calls, Attendance,
)


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
                 to Advisor via `wid`, and for team/company level the
                 compiler aggregates with `agg`.
    - agg:       "sum" | "avg" — how to roll an advisor-level column up to
                 team/company. Rate-like metrics (conversion, achievement)
                 must average, not sum.
    - period:    for the shared `Performance` table, which PerformancePeriod
                 row this metric reads.
    """
    model: type
    expr: Any
    team_named: bool = False
    agg: str = "sum"
    period: Optional[PerformancePeriod] = None


@dataclass
class MetricDef:
    key: str
    label: str
    synonyms: list[str]
    entity_levels: list[str]          # levels with a binding below
    primary_level: str                # level used when the query doesn't specify one
    bindings: dict[str, ColumnBinding] = field(default_factory=dict)


METRICS: dict[str, MetricDef] = {

    "achievement_pct": MetricDef(
        key="achievement_pct",
        label="Target Achievement %",
        synonyms=["target achievement", "achievement", "hit rate", "on target", "achieved target",
                   "target hit", "performance against target", "top performer", "performer", "performance"],
        entity_levels=["advisor", "team"],
        primary_level="team",
        bindings={
            "team": ColumnBinding(model=TeamTarget, expr=TeamTarget.achievement_pct, team_named=True),
            "advisor": ColumnBinding(model=Performance, expr=Performance.pct, period=PerformancePeriod.MTD, agg="avg"),
        },
    ),

    "mtd_cleared": MetricDef(
        key="mtd_cleared",
        label="MTD Revenue Cleared",
        synonyms=["revenue", "cleared", "sales", "closed revenue", "closed", "highest sales", "highest closer", "top closer"],
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
        synonyms=["ytd cleared", "ytd revenue", "year to date revenue", "year to date cleared", "annual cleared"],
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Performance, expr=Performance.cleared, period=PerformancePeriod.YTD),
        },
    ),

    "three_month_cleared": MetricDef(
        key="three_month_cleared",
        label="3-Month Revenue Cleared",
        synonyms=["3 month cleared", "three month cleared", "quarterly cleared", "3m cleared", "quarter revenue"],
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Performance, expr=Performance.cleared, period=PerformancePeriod.THREE_M),
        },
    ),

    "mtd_target": MetricDef(
        key="mtd_target",
        label="MTD Target",
        synonyms=["mtd target", "monthly target", "this month's target", "month target", "target"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Performance, expr=Performance.target, period=PerformancePeriod.MTD),
            "team": ColumnBinding(model=TeamTarget, expr=TeamTarget.target, team_named=True),
        },
    ),

    "total_connects": MetricDef(
        key="total_connects",
        label="Total MTD Connects",
        synonyms=["connects", "connections", "connect", "total connects", "all connects", "most connects"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_new_connect + SalesFunnel.mtd_followup_connect),
            "team": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_new_connect + SalesFunnel.mtd_followup_connect),
        },
    ),

    "new_connects": MetricDef(
        key="new_connects",
        label="New MTD Connects",
        synonyms=["new connects", "new connect", "fresh connects", "first connects"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_new_connect),
            "team": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_new_connect),
        },
    ),

    "followup_connects": MetricDef(
        key="followup_connects",
        label="Follow-up MTD Connects",
        synonyms=["follow-up connects", "followup connects", "follow up connects", "repeat connects"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_followup_connect),
            "team": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_followup_connect),
        },
    ),

    "total_meetings": MetricDef(
        key="total_meetings",
        label="Total MTD Meetings",
        synonyms=["meetings", "meeting", "total meetings", "most meetings", "meetings held"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_new_meeting + SalesFunnel.mtd_followup_meeting),
            "team": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_new_meeting + SalesFunnel.mtd_followup_meeting),
        },
    ),

    "conversion": MetricDef(
        key="conversion",
        label="Conversion Rate",
        synonyms=["conversion", "conversions", "conversion rate"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_conversion),
            # a rate metric averages, it doesn't sum, across a team
            "team": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_conversion, agg="avg"),
        },
    ),

    "bookings": MetricDef(
        key="bookings",
        label="Bookings Stored",
        synonyms=["booking", "bookings", "booked units", "booking stored"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_booking_stored),
            "team": ColumnBinding(model=SalesFunnel, expr=SalesFunnel.mtd_booking_stored),
        },
    ),

    "pipeline_value": MetricDef(
        key="pipeline_value",
        label="Open Pipeline",
        synonyms=["pipeline", "pipeline value", "open pipeline", "open deals"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Pipeline, expr=Pipeline.pipeline),
            "team": ColumnBinding(model=Pipeline, expr=Pipeline.pipeline),
        },
    ),

    "overdue": MetricDef(
        key="overdue",
        label="Overdue Pipeline",
        synonyms=["overdue", "past due"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Pipeline, expr=Pipeline.overdue),
            "team": ColumnBinding(model=Pipeline, expr=Pipeline.overdue),
        },
    ),

    "overdue_amount": MetricDef(
        key="overdue_amount",
        label="Overdue Amount",
        synonyms=["overdue amount", "overdue value", "amount overdue"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            # Same underlying column as `overdue` — the schema has one overdue
            # field, not separate count-vs-amount columns.
            "advisor": ColumnBinding(model=Pipeline, expr=Pipeline.overdue),
            "team": ColumnBinding(model=Pipeline, expr=Pipeline.overdue),
        },
    ),

    "portfolio_value": MetricDef(
        key="portfolio_value",
        label="Portfolio Value",
        synonyms=["portfolio", "portfolio value", "book size"],
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Portfolio, expr=Portfolio.value),
        },
    ),

    "returned_value": MetricDef(
        key="returned_value",
        label="Portfolio Returned",
        synonyms=["returned", "returned value", "returns", "portfolio returned"],
        entity_levels=["advisor"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Portfolio, expr=Portfolio.returned),
        },
    ),

    "answered_calls": MetricDef(
        key="answered_calls",
        label="Answered Calls (MTD)",
        synonyms=["answered calls", "calls answered", "call answered"],
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
        synonyms=["late count", "how many late", "number of late", "late arrivals", "late"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
        bindings={
            "advisor": ColumnBinding(model=Attendance, expr=Attendance.biometric_mtd_late),
            "team": ColumnBinding(model=Attendance, expr=Attendance.biometric_mtd_late),
        },
    ),

    "attendance_rate": MetricDef(
        key="attendance_rate",
        label="Attendance Rate % (MTD)",
        synonyms=["attendance rate", "attendance percentage", "attendance %", "attendance",
                  "on time rate", "on-time percentage", "punctuality"],
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
            ),
            # a rate metric averages, it doesn't sum, across a team
            "team": ColumnBinding(
                model=Attendance,
                expr=Attendance.biometric_mtd_ontime * 100.0 / func.nullif(
                    Attendance.biometric_mtd_ontime + Attendance.biometric_mtd_late + Attendance.biometric_mtd_not_marked, 0
                ),
                agg="avg",
            ),
        },
    ),
}

# Longest synonym first, so e.g. "ytd cleared" matches before the bare word
# "cleared" could, and "new connects" before bare "connects".
_SYNONYM_INDEX: list[tuple[str, str]] = sorted(
    ((syn, m.key) for m in METRICS.values() for syn in m.synonyms + [m.label.lower()]),
    key=lambda pair: -len(pair[0]),
)


def resolve_metric(text: str) -> str | None:
    q = text.lower()
    for synonym, key in _SYNONYM_INDEX:
        if synonym in q:
            return key
    return None


def metric_label(metric_key: str | None) -> str:
    """Shared by response_formatter.py and narrative.py so 'what do we call
    this metric in a reply' has one answer, not two independently
    maintained copies."""
    if not metric_key:
        return "value"
    return METRICS[metric_key].label if metric_key in METRICS else metric_key


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
