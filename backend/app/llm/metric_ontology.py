# # """
# # Semantic layer: maps how people actually phrase a metric to a canonical key
# # the rest of the pipeline understands, and declares which entity levels
# # (advisor / team / company) that metric can be queried at.

# # Adding a new metric means adding one entry here and one resolver in
# # sql_generator.py — nothing else in the pipeline needs to change.
# # """

# # from dataclasses import dataclass


# # @dataclass
# # class MetricDef:
# #     key: str
# #     label: str
# #     synonyms: list[str]
# #     entity_levels: list[str]     # levels with a working resolver in sql_generator.py
# #     primary_level: str           # level used when the query doesn't specify one


# # METRICS: dict[str, MetricDef] = {
# #     "achievement_pct": MetricDef(
# #         key="achievement_pct",
# #         label="Target Achievement %",
# #         synonyms=["target achievement", "achievement", "hit rate", "on target", "achieved target", "target hit", "performance against target", "top performer", "performer"],
# #         entity_levels=["advisor", "team"],
# #         primary_level="team",   # Target Achievement is genuinely team-level source data
# #     ),
# #     "mtd_cleared": MetricDef(
# #         key="mtd_cleared",
# #         label="MTD Revenue Cleared",
# #         synonyms=["revenue", "cleared", "sales", "closed revenue", "closed"],
# #         entity_levels=["advisor"],
# #         primary_level="advisor",
# #     ),
# #     "mtd_new_connect": MetricDef(
# #         key="mtd_new_connect",
# #         label="MTD Connects",
# #         synonyms=["connect", "connects", "connections"],
# #         entity_levels=["advisor"],
# #         primary_level="advisor",
# #     ),
# #     "overdue": MetricDef(
# #         key="overdue",
# #         label="Overdue Pipeline Items",
# #         synonyms=["overdue", "past due"],
# #         entity_levels=["advisor", "team"],
# #         primary_level="advisor",
# #     ),
# #     "pipeline": MetricDef(
# #         key="pipeline",
# #         label="Open Pipeline",
# #         synonyms=["pipeline", "open deals", "open pipeline"],
# #         entity_levels=["advisor"],
# #         primary_level="advisor",
# #     ),
# #     "mtd_conversion": MetricDef(
# #         key="mtd_conversion",
# #         label="Conversion (Bookings)",
# #         synonyms=["conversion", "conversions", "booked"],
# #         entity_levels=["advisor"],
# #         primary_level="advisor",
# #     ),
# #     "portfolio_value": MetricDef(
# #         key="portfolio_value",
# #         label="Portfolio Value",
# #         synonyms=["portfolio", "portfolio value", "book size"],
# #         entity_levels=["advisor"],
# #         primary_level="advisor",
# #     ),
# # }

# # # Longest synonym first, so "target achievement" matches before the bare word "target" could.
# # _SYNONYM_INDEX: list[tuple[str, str]] = sorted(
# #     ((syn, m.key) for m in METRICS.values() for syn in m.synonyms + [m.label.lower()]),
# #     key=lambda pair: -len(pair[0]),
# # )


# # def resolve_metric(text: str) -> str | None:
# #     q = text.lower()
# #     for synonym, key in _SYNONYM_INDEX:
# #         if synonym in q:
# #             return key
# #     return None


# # def describe_available_metrics() -> str:
# #     """Used by the fallback clarification message — grounds the 'I don't
# #     understand' response in what's actually queryable instead of a canned
# #     generic apology."""
# #     return ", ".join(f"{m.label.lower()} ({'/'.join(m.entity_levels)})" for m in METRICS.values())


# # # Metric Ontology

# # # Performance
# # # ├── mtd_cleared
# # # ├── mtd_target
# # # ├── mtd_achievement_pct
# # # ├── ytd_cleared
# # # ├── ytd_target
# # # ├── ytd_achievement_pct
# # # ├── three_month_cleared
# # # ├── three_month_target
# # # └── three_month_achievement_pct

# # # Sales Funnel
# # # ├── new_connects
# # # ├── followup_connects
# # # ├── total_connects
# # # ├── new_meetings
# # # ├── followup_meetings
# # # ├── conversion
# # # ├── bookings
# # # └── todo

# # # Pipeline
# # # ├── pipeline_value
# # # ├── overdue_items
# # # └── overdue_amount

# # # Portfolio
# # # ├── portfolio_value
# # # ├── returned_value
# # # └── retention_percentage

# # # Calls
# # # ├── answered_calls
# # # ├── daily_calls
# # # └── connects

# # # Attendance
# # # ├── biometric_ontime
# # # ├── biometric_late
# # # ├── login_ontime
# # # └── login_late

# # # Organization
# # # ├── advisor
# # # ├── team
# # # ├── company
# # # ├── region
# # # ├── office
# # # ├── unit
# # # └── management hierarchy
# # # must be grown according to this ontology

# """
# Semantic Layer / Metric Ontology

# Maps business language used by employees into canonical metrics
# understood by the query planner and SQL resolver.

# Flow:

# User Query
#     |
#     v
# Metric Ontology
#     |
#     v
# Canonical Metric Key
#     |
#     v
# Query Planner
#     |
#     v
# SQL Resolver

# Adding a new metric:
# 1. Add MetricDef entry here
# 2. Add resolver in sql_generator.py

# No other pipeline changes required.
# """

# from dataclasses import dataclass


# @dataclass
# class MetricDef:
#     key: str
#     label: str
#     synonyms: list[str]
#     entity_levels: list[str]
#     primary_level: str


# # ==========================================================
# # Metric Definitions
# # ==========================================================

# METRICS: dict[str, MetricDef] = {


#     # =========================
#     # PERFORMANCE METRICS
#     # =========================

#     "mtd_cleared": MetricDef(
#         key="mtd_cleared",
#         label="MTD Revenue Cleared",
#         synonyms=[
#             "sales",
#             "revenue",
#             "cleared",
#             "closed revenue",
#             "closed sales",
#             "mtd sales",
#             "monthly sales"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),


#     "mtd_target": MetricDef(
#         key="mtd_target",
#         label="MTD Target",
#         synonyms=[
#             "target",
#             "mtd target",
#             "monthly target",
#             "sales target",
#             "goal"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),


#     "achievement_pct": MetricDef(
#         key="achievement_pct",
#         label="Target Achievement %",
#         synonyms=[
#             "target achievement",
#             "achievement",
#             "achievement percentage",
#             "achievement %",
#             "target hit",
#             "target reached",
#             "hit rate",
#             "on target"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="team",
#     ),


#     "ytd_cleared": MetricDef(
#         key="ytd_cleared",
#         label="YTD Revenue Cleared",
#         synonyms=[
#             "year sales",
#             "ytd sales",
#             "year revenue",
#             "yearly revenue"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),


#     "three_month_cleared": MetricDef(
#         key="three_month_cleared",
#         label="3 Month Revenue Cleared",
#         synonyms=[
#             "3 month sales",
#             "three month sales",
#             "quarter sales",
#             "3m revenue"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),



#     # =========================
#     # SALES FUNNEL
#     # =========================

#     "new_connects": MetricDef(
#         key="new_connects",
#         label="New Connects",
#         synonyms=[
#             "new connects",
#             "fresh connects",
#             "new calls",
#             "new customer connects"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),


#     "followup_connects": MetricDef(
#         key="followup_connects",
#         label="Followup Connects",
#         synonyms=[
#             "followup connects",
#             "follow up calls",
#             "followups"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),


#     "total_connects": MetricDef(
#         key="total_connects",
#         label="Total Connects",
#         synonyms=[
#             "connects",
#             "total connects",
#             "connections",
#             "customer connections"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),


#     "conversion": MetricDef(
#         key="conversion",
#         label="Conversion Rate",
#         synonyms=[
#             "conversion",
#             "conversion rate",
#             "booking conversion",
#             "converted"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),


#     "bookings": MetricDef(
#         key="bookings",
#         label="Bookings",
#         synonyms=[
#             "bookings",
#             "total bookings",
#             "confirmed bookings"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),



#     # =========================
#     # PIPELINE
#     # =========================

#     "pipeline_value": MetricDef(
#         key="pipeline_value",
#         label="Pipeline Value",
#         synonyms=[
#             "pipeline",
#             "open pipeline",
#             "active pipeline",
#             "deal value",
#             "pending deals"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),


#     "overdue": MetricDef(
#         key="overdue",
#         label="Overdue Pipeline Items",
#         synonyms=[
#             "overdue",
#             "past due",
#             "late pipeline",
#             "pending overdue"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),


#     "overdue_amount": MetricDef(
#         key="overdue_amount",
#         label="Overdue Amount",
#         synonyms=[
#             "overdue amount",
#             "late amount"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),



#     # =========================
#     # PORTFOLIO
#     # =========================

#     "portfolio_value": MetricDef(
#         key="portfolio_value",
#         label="Portfolio Value",
#         synonyms=[
#             "portfolio",
#             "portfolio value",
#             "book size",
#             "managed portfolio"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),


#     "returned_value": MetricDef(
#         key="returned_value",
#         label="Returned Value",
#         synonyms=[
#             "returned",
#             "returns",
#             "returned business"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),



#     # =========================
#     # CALL METRICS
#     # =========================

#     "answered_calls": MetricDef(
#         key="answered_calls",
#         label="Answered Calls",
#         synonyms=[
#             "answered calls",
#             "picked calls",
#             "received calls"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),



#     # =========================
#     # ATTENDANCE
#     # =========================

#     "late_count": MetricDef(
#         key="late_count",
#         label="Late Attendance Count",
#         synonyms=[
#             "late",
#             "late arrivals",
#             "attendance issues",
#             "late employees"
#         ],
#         entity_levels=[
#             "advisor",
#             "team",
#             "company"
#         ],
#         primary_level="advisor",
#     ),

# }


# # ==========================================================
# # Resolver Index
# # Longest synonym first to avoid partial collisions
# # ==========================================================

# _SYNONYM_INDEX: list[tuple[str, str]] = sorted(
#     (
#         (synonym.lower(), metric.key)
#         for metric in METRICS.values()
#         for synonym in metric.synonyms + [metric.label.lower()]
#     ),
#     key=lambda x: -len(x[0])
# )



# def resolve_metric(text: str) -> str | None:
#     """
#     Convert user language into canonical metric key.

#     Example:

#     "highest sales"
#         ->
#     "mtd_cleared"

#     "target achievement"
#         ->
#     "achievement_pct"
#     """

#     query = text.lower()

#     for synonym, key in _SYNONYM_INDEX:
#         if synonym in query:
#             return key

#     return None



# def describe_available_metrics() -> str:
#     """
#     Used for clarification responses.
#     """

#     return ", ".join(
#         f"{metric.label} ({'/'.join(metric.entity_levels)})"
#         for metric in METRICS.values()
#     )



"""
Semantic layer: canonical business-friendly metric names, their phrasing
synonyms, and which entity levels each one supports.

INVARIANT this file and sql_generator.py must both uphold: every
(metric.key, level) pair listed in entity_levels below MUST have a
matching @resolver(metric_key, level) in sql_generator.py. Add a metric
here and forget the resolver -> it silently returns "I don't have a way
to rank by that" instead of data. There's a test for this — see
tests/llm/test_ontology_sync.py.
"""

from dataclasses import dataclass


@dataclass
class MetricDef:
    key: str
    label: str
    synonyms: list[str]
    entity_levels: list[str]     # levels with a working resolver in sql_generator.py
    primary_level: str           # level used when the query doesn't specify one


METRICS: dict[str, MetricDef] = {

    "achievement_pct": MetricDef(
        key="achievement_pct",
        label="Target Achievement %",
        synonyms=["target achievement", "achievement", "hit rate", "on target", "achieved target",
                   "target hit", "performance against target", "top performer", "performer", "performance"],
        entity_levels=["advisor", "team"],
        primary_level="team",   # Target Achievement is genuine team-level source data
    ),

    "mtd_cleared": MetricDef(
        key="mtd_cleared",
        label="MTD Revenue Cleared",
        synonyms=["revenue", "cleared", "sales", "closed revenue", "closed", "highest sales"],
        entity_levels=["advisor"],
        primary_level="advisor",
    ),

    "ytd_cleared": MetricDef(
        key="ytd_cleared",
        label="YTD Revenue Cleared",
        synonyms=["ytd cleared", "ytd revenue", "year to date revenue", "year to date cleared", "annual cleared"],
        entity_levels=["advisor"],
        primary_level="advisor",
    ),

    "three_month_cleared": MetricDef(
        key="three_month_cleared",
        label="3-Month Revenue Cleared",
        synonyms=["3 month cleared", "three month cleared", "quarterly cleared", "3m cleared", "quarter revenue"],
        entity_levels=["advisor"],
        primary_level="advisor",
    ),

    "mtd_target": MetricDef(
        key="mtd_target",
        label="MTD Target",
        synonyms=["mtd target", "monthly target", "this month's target", "month target", "target"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
    ),

    "total_connects": MetricDef(
        key="total_connects",
        label="Total MTD Connects",
        synonyms=["connects", "connections", "connect", "total connects", "all connects", "most connects"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
    ),

    "new_connects": MetricDef(
        key="new_connects",
        label="New MTD Connects",
        synonyms=["new connects", "new connect", "fresh connects", "first connects"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
    ),

    "followup_connects": MetricDef(
        key="followup_connects",
        label="Follow-up MTD Connects",
        synonyms=["follow-up connects", "followup connects", "follow up connects", "repeat connects"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
    ),

    "conversion": MetricDef(
        key="conversion",
        label="Conversion Rate",
        synonyms=["conversion", "conversions", "conversion rate"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
    ),

    "bookings": MetricDef(
        key="bookings",
        label="Bookings Stored",
        synonyms=["booking", "bookings", "booked units", "booking stored"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
    ),

    "pipeline_value": MetricDef(
        key="pipeline_value",
        label="Open Pipeline",
        synonyms=["pipeline", "pipeline value", "open pipeline", "open deals"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
    ),

    "overdue": MetricDef(
        key="overdue",
        label="Overdue Pipeline",
        synonyms=["overdue", "past due"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
    ),

    "overdue_amount": MetricDef(
        key="overdue_amount",
        label="Overdue Amount",
        synonyms=["overdue amount", "overdue value", "amount overdue"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
    ),

    "portfolio_value": MetricDef(
        key="portfolio_value",
        label="Portfolio Value",
        synonyms=["portfolio", "portfolio value", "book size"],
        entity_levels=["advisor"],
        primary_level="advisor",
    ),

    "returned_value": MetricDef(
        key="returned_value",
        label="Portfolio Returned",
        synonyms=["returned", "returned value", "returns", "portfolio returned"],
        entity_levels=["advisor"],
        primary_level="advisor",
    ),

    "answered_calls": MetricDef(
        key="answered_calls",
        label="Answered Calls (MTD)",
        synonyms=["answered calls", "calls answered", "call answered"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
    ),

    "late_count": MetricDef(
        key="late_count",
        label="Late Attendance Count (MTD)",
        synonyms=["late count", "how many late", "number of late", "late arrivals"],
        entity_levels=["advisor", "team"],
        primary_level="advisor",
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


def describe_available_metrics() -> str:
    return ", ".join(f"{m.label.lower()} ({'/'.join(m.entity_levels)})" for m in METRICS.values())