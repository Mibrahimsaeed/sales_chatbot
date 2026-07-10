from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database.models import Advisor, SalesFunnel, Pipeline, Performance, PerformancePeriod
from app.core.exception import NotFoundError


def get_company_summary(db: Session, company: str) -> dict:
    activity = (
        db.query(
            func.count(Advisor.wid).label("advisors"),
            func.sum(SalesFunnel.mtd_new_connect + SalesFunnel.mtd_followup_connect).label("connects"),
            func.sum(Pipeline.overdue).label("overdue"),
            func.sum(Pipeline.pipeline).label("pipeline"),
        )
        .join(SalesFunnel, SalesFunnel.wid == Advisor.wid, isouter=True)
        .join(Pipeline, Pipeline.wid == Advisor.wid, isouter=True)
        .filter(Advisor.company.ilike(company))
        .first()
    )
    if not activity or not activity.advisors:
        raise NotFoundError(f"No company matching '{company}'")

    revenue = (
        db.query(
            func.sum(Performance.target).label("target"),
            func.sum(Performance.cleared).label("cleared"),
        )
        .join(Advisor, Advisor.wid == Performance.wid)
        .filter(Advisor.company.ilike(company), Performance.period == PerformancePeriod.MTD)
        .first()
    )

    return {
        "company": company,
        "advisors": activity.advisors or 0,
        "connects": activity.connects or 0,
        "overdue": activity.overdue or 0,
        "pipeline": activity.pipeline or 0,
        "mtd_target": (revenue.target or 0) if revenue else 0,
        "mtd_cleared": (revenue.cleared or 0) if revenue else 0,
    }