"""
Executes a QueryPlan against the DB. Each valid (metric, level) pair
declared in metric_ontology.py's entity_levels has exactly one resolver
here — this file IS the translation layer between business-friendly
ontology names and the actual database columns. An unregistered pair
returns None rather than improvising a query.
"""

from typing import Callable
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc, func
from app.database.models import (
    Advisor, SalesFunnel, Pipeline, Performance, PerformancePeriod,
    TeamTarget, Portfolio, Calls, Attendance,
)
from app.llm.query_planner import QueryPlan

RESOLVERS: dict[tuple[str, str], Callable] = {}


def resolver(metric: str, level: str):
    def decorator(fn):
        RESOLVERS[(metric, level)] = fn
        return fn
    return decorator


def _order(column, ascending: bool):
    return asc(column) if ascending else desc(column)


# ---------------------------------------------------------------------------
# Time-period resolution
# ---------------------------------------------------------------------------

# Maps the planner's time_period string -> PerformancePeriod enum value.
_PERIOD_MAP = {
    "mtd":       PerformancePeriod.MTD,
    "ytd":       PerformancePeriod.YTD,
    "this_week": None,
    "today":    None,
}


def _resolve_period(plan: QueryPlan) -> PerformancePeriod | None:
    """
    Return the PerformancePeriod enum value for a plan, falling back to MTD.
    Returns None for non-Performance queries.
    """
    return _PERIOD_MAP.get(plan.time_period, PerformancePeriod.MTD)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_leaderboard(db: Session, plan: QueryPlan) -> list[dict] | None:
    """
    Run the resolver registered for (plan.metric, plan.level).

    Returns:
        - None   if no resolver is registered (truly unknown pair —
                 chat_service will fall through to the unknown reply)
        - list   (possibly empty) for all registered pairs
    """
    fn = RESOLVERS.get((plan.metric, plan.level))
    if not fn:
        return None
    return fn(db, plan)


# Convenience constructor used by every resolver below to keep the row
# shape consistent (advisor rows include wid/name/team/company; team rows
# only have name + value).
def _team_row(name, value):
    return {"wid": None, "name": name, "team": name, "company": None, "value": value}


# ---------------------------------------------------------------------------
# (the rest of the file is the existing resolvers, unchanged)
# ---------------------------------------------------------------------------


"""
Executes a QueryPlan against the DB. Each valid (metric, level) pair
declared in metric_ontology.py's entity_levels has exactly one resolver
here — this file IS the translation layer between business-friendly
ontology names and the actual database columns. An unregistered pair
returns None rather than improvising a query.

Section headers below match metric_ontology.py's metric order — when you
add a metric there, add its resolver(s) in the matching section here.
"""

from typing import Callable
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc, func
from app.database.models import (
    Advisor, SalesFunnel, Pipeline, Performance, PerformancePeriod,
    TeamTarget, Portfolio, Calls, Attendance,
)
from app.llm.query_planner import QueryPlan

RESOLVERS: dict[tuple[str, str], Callable] = {}


def resolver(metric: str, level: str):
    def decorator(fn):
        RESOLVERS[(metric, level)] = fn
        return fn
    return decorator


def _order(column, ascending: bool):
    return asc(column) if ascending else desc(column)


def run_leaderboard(db: Session, plan: QueryPlan) -> list[dict] | None:
    fn = RESOLVERS.get((plan.metric, plan.level))
    if not fn:
        return None
    return fn(db, plan)


def _team_row(name, value):
    return {"wid": None, "name": name, "team": name, "company": None, "value": value}


# ---- achievement_pct ----

@resolver("achievement_pct", "team")
def _(db, plan):
    rows = (
        db.query(TeamTarget.team, TeamTarget.achievement_pct.label("value"))
        .order_by(_order(TeamTarget.achievement_pct, plan.ascending))
        .limit(plan.limit).all()
    )
    return [_team_row(r.team, r.value) for r in rows]


@resolver("achievement_pct", "advisor")
def _(db, plan):
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, Performance.pct.label("value"))
        .join(Performance, Performance.wid == Advisor.wid)
        .filter(Performance.period == PerformancePeriod.MTD)
        .order_by(_order(Performance.pct, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


# ---- mtd_cleared / ytd_cleared / three_month_cleared ----

def _cleared_by_period(db, plan, period):
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, Performance.cleared.label("value"))
        .join(Performance, Performance.wid == Advisor.wid)
        .filter(Performance.period == period)
        .order_by(_order(Performance.cleared, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


@resolver("mtd_cleared", "advisor")
def _(db, plan):
    return _cleared_by_period(db, plan, PerformancePeriod.MTD)


@resolver("ytd_cleared", "advisor")
def _(db, plan):
    return _cleared_by_period(db, plan, PerformancePeriod.YTD)


@resolver("three_month_cleared", "advisor")
def _(db, plan):
    return _cleared_by_period(db, plan, PerformancePeriod.THREE_M)


# ---- mtd_target ----

@resolver("mtd_target", "advisor")
def _(db, plan):
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, Performance.target.label("value"))
        .join(Performance, Performance.wid == Advisor.wid)
        .filter(Performance.period == PerformancePeriod.MTD)
        .order_by(_order(Performance.target, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


@resolver("mtd_target", "team")
def _(db, plan):
    rows = (
        db.query(TeamTarget.team, TeamTarget.target.label("value"))
        .order_by(_order(TeamTarget.target, plan.ascending))
        .limit(plan.limit).all()
    )
    return [_team_row(r.team, r.value) for r in rows]


# ---- total_connects / new_connects / followup_connects ----

@resolver("total_connects", "advisor")
def _(db, plan):
    total = SalesFunnel.mtd_new_connect + SalesFunnel.mtd_followup_connect
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, total.label("value"))
        .join(SalesFunnel, SalesFunnel.wid == Advisor.wid)
        .order_by(_order(total, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


@resolver("total_connects", "team")
def _(db, plan):
    total = func.sum(SalesFunnel.mtd_new_connect + SalesFunnel.mtd_followup_connect)
    rows = (
        db.query(Advisor.team.label("name"), total.label("value"))
        .join(SalesFunnel, SalesFunnel.wid == Advisor.wid)
        .filter(Advisor.team.isnot(None))
        .group_by(Advisor.team)
        .order_by(_order(total, plan.ascending))
        .limit(plan.limit).all()
    )
    return [_team_row(r.name, r.value) for r in rows]


@resolver("new_connects", "advisor")
def _(db, plan):
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, SalesFunnel.mtd_new_connect.label("value"))
        .join(SalesFunnel, SalesFunnel.wid == Advisor.wid)
        .order_by(_order(SalesFunnel.mtd_new_connect, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


@resolver("new_connects", "team")
def _(db, plan):
    total = func.sum(SalesFunnel.mtd_new_connect)
    rows = (
        db.query(Advisor.team.label("name"), total.label("value"))
        .join(SalesFunnel, SalesFunnel.wid == Advisor.wid)
        .filter(Advisor.team.isnot(None))
        .group_by(Advisor.team)
        .order_by(_order(total, plan.ascending))
        .limit(plan.limit).all()
    )
    return [_team_row(r.name, r.value) for r in rows]


@resolver("followup_connects", "advisor")
def _(db, plan):
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, SalesFunnel.mtd_followup_connect.label("value"))
        .join(SalesFunnel, SalesFunnel.wid == Advisor.wid)
        .order_by(_order(SalesFunnel.mtd_followup_connect, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


@resolver("followup_connects", "team")
def _(db, plan):
    total = func.sum(SalesFunnel.mtd_followup_connect)
    rows = (
        db.query(Advisor.team.label("name"), total.label("value"))
        .join(SalesFunnel, SalesFunnel.wid == Advisor.wid)
        .filter(Advisor.team.isnot(None))
        .group_by(Advisor.team)
        .order_by(_order(total, plan.ascending))
        .limit(plan.limit).all()
    )
    return [_team_row(r.name, r.value) for r in rows]


# ---- conversion ----

@resolver("conversion", "advisor")
def _(db, plan):
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, SalesFunnel.mtd_conversion.label("value"))
        .join(SalesFunnel, SalesFunnel.wid == Advisor.wid)
        .order_by(_order(SalesFunnel.mtd_conversion, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


@resolver("conversion", "team")
def _(db, plan):
    # a rate metric averages, it doesn't sum, across a team
    avg = func.avg(SalesFunnel.mtd_conversion)
    rows = (
        db.query(Advisor.team.label("name"), avg.label("value"))
        .join(SalesFunnel, SalesFunnel.wid == Advisor.wid)
        .filter(Advisor.team.isnot(None))
        .group_by(Advisor.team)
        .order_by(_order(avg, plan.ascending))
        .limit(plan.limit).all()
    )
    return [_team_row(r.name, r.value) for r in rows]


# ---- bookings (SalesFunnel.mtd_booking_stored, per the reported mapping) ----

@resolver("bookings", "advisor")
def _(db, plan):
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, SalesFunnel.mtd_booking_stored.label("value"))
        .join(SalesFunnel, SalesFunnel.wid == Advisor.wid)
        .order_by(_order(SalesFunnel.mtd_booking_stored, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


@resolver("bookings", "team")
def _(db, plan):
    total = func.sum(SalesFunnel.mtd_booking_stored)
    rows = (
        db.query(Advisor.team.label("name"), total.label("value"))
        .join(SalesFunnel, SalesFunnel.wid == Advisor.wid)
        .filter(Advisor.team.isnot(None))
        .group_by(Advisor.team)
        .order_by(_order(total, plan.ascending))
        .limit(plan.limit).all()
    )
    return [_team_row(r.name, r.value) for r in rows]


# ---- pipeline_value ----

@resolver("pipeline_value", "advisor")
def _(db, plan):
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, Pipeline.pipeline.label("value"))
        .join(Pipeline, Pipeline.wid == Advisor.wid)
        .order_by(_order(Pipeline.pipeline, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


@resolver("pipeline_value", "team")
def _(db, plan):
    total = func.sum(Pipeline.pipeline)
    rows = (
        db.query(Advisor.team.label("name"), total.label("value"))
        .join(Pipeline, Pipeline.wid == Advisor.wid)
        .filter(Advisor.team.isnot(None))
        .group_by(Advisor.team)
        .order_by(_order(total, plan.ascending))
        .limit(plan.limit).all()
    )
    return [_team_row(r.name, r.value) for r in rows]


# ---- overdue / overdue_amount (same underlying column — the schema has
# one overdue field, not separate count-vs-amount columns; both ontology
# keys resolve to it until/unless the source sheets split that out) ----

@resolver("overdue", "advisor")
def _(db, plan):
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, Pipeline.overdue.label("value"))
        .join(Pipeline, Pipeline.wid == Advisor.wid)
        .order_by(_order(Pipeline.overdue, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


@resolver("overdue", "team")
def _(db, plan):
    total = func.sum(Pipeline.overdue)
    rows = (
        db.query(Advisor.team.label("name"), total.label("value"))
        .join(Pipeline, Pipeline.wid == Advisor.wid)
        .filter(Advisor.team.isnot(None))
        .group_by(Advisor.team)
        .order_by(_order(total, plan.ascending))
        .limit(plan.limit).all()
    )
    return [_team_row(r.name, r.value) for r in rows]


RESOLVERS[("overdue_amount", "advisor")] = RESOLVERS[("overdue", "advisor")]
RESOLVERS[("overdue_amount", "team")] = RESOLVERS[("overdue", "team")]


# ---- portfolio_value / returned_value ----

@resolver("portfolio_value", "advisor")
def _(db, plan):
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, Portfolio.value.label("value"))
        .join(Portfolio, Portfolio.wid == Advisor.wid)
        .order_by(_order(Portfolio.value, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


@resolver("returned_value", "advisor")
def _(db, plan):
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, Portfolio.returned.label("value"))
        .join(Portfolio, Portfolio.wid == Advisor.wid)
        .order_by(_order(Portfolio.returned, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


# ---- answered_calls ----

@resolver("answered_calls", "advisor")
def _(db, plan):
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, Calls.answered_calls_mtd.label("value"))
        .join(Calls, Calls.wid == Advisor.wid)
        .order_by(_order(Calls.answered_calls_mtd, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


@resolver("answered_calls", "team")
def _(db, plan):
    total = func.sum(Calls.answered_calls_mtd)
    rows = (
        db.query(Advisor.team.label("name"), total.label("value"))
        .join(Calls, Calls.wid == Advisor.wid)
        .filter(Advisor.team.isnot(None))
        .group_by(Advisor.team)
        .order_by(_order(total, plan.ascending))
        .limit(plan.limit).all()
    )
    return [_team_row(r.name, r.value) for r in rows]


# ---- late_count ----

@resolver("late_count", "advisor")
def _(db, plan):
    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, Attendance.biometric_mtd_late.label("value"))
        .join(Attendance, Attendance.wid == Advisor.wid)
        .order_by(_order(Attendance.biometric_mtd_late, plan.ascending))
        .limit(plan.limit).all()
    )
    return [dict(r._mapping) for r in rows]


@resolver("late_count", "team")
def _(db, plan):
    total = func.sum(Attendance.biometric_mtd_late)
    rows = (
        db.query(Advisor.team.label("name"), total.label("value"))
        .join(Attendance, Attendance.wid == Advisor.wid)
        .filter(Advisor.team.isnot(None))
        .group_by(Advisor.team)
        .order_by(_order(total, plan.ascending))
        .limit(plan.limit).all()
    )
    return [_team_row(r.name, r.value) for r in rows]