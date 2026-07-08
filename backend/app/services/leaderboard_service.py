from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.database.models import Advisor, SalesFunnel, Pipeline, Performance, PerformancePeriod
from app.core.exception import UnsupportedMetricError

# metric name -> (table to join, column, join condition handled below)
METRIC_MAP = {
    "mtd_new_connect": SalesFunnel.mtd_new_connect,
    "overdue": Pipeline.overdue,
}


def get_leaderboard(db: Session, metric: str, limit: int = 10):
    if metric == "mtd_cleared":
        rows = (
            db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, Performance.cleared)
            .join(Performance, Performance.wid == Advisor.wid)
            .filter(Performance.period == PerformancePeriod.MTD)
            .order_by(desc(Performance.cleared))
            .limit(limit)
            .all()
        )
        return [{"wid": r.wid, "name": r.name, "team": r.team, "company": r.company, "value": r.cleared} for r in rows]

    if metric not in METRIC_MAP:
        raise UnsupportedMetricError(metric)

    col = METRIC_MAP[metric]
    table = SalesFunnel if metric == "mtd_new_connect" else Pipeline

    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company, col.label("value"))
        .join(table, table.wid == Advisor.wid)
        .order_by(desc(col))
        .limit(limit)
        .all()
    )
    return [{"wid": r.wid, "name": r.name, "team": r.team, "company": r.company, "value": r.value} for r in rows]