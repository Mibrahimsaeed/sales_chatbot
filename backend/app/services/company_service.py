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
        .filter(Advisor.company.ilike(company), Advisor.in_master_sheet.is_(True))
        .first()
    )
    if not activity or not activity.advisors:
        raise NotFoundError(f"No company matching '{company}'")

    def _revenue(period):
        return (
            db.query(
                func.sum(Performance.target).label("target"),
                func.sum(Performance.cleared).label("cleared"),
            )
            .join(Advisor, Advisor.wid == Performance.wid)
            .filter(
                Advisor.company.ilike(company),
                Advisor.in_master_sheet.is_(True),
                Performance.period == period,
            )
            .first()
        )

    mtd_revenue = _revenue(PerformancePeriod.MTD)
    # bug fix: this only ever queried MTD, so "YTD performance" for a
    # company always silently came back with MTD numbers (or nothing).
    ytd_revenue = _revenue(PerformancePeriod.YTD)

    return {
        "company": company,
        "advisors": activity.advisors or 0,
        "connects": activity.connects or 0,
        "overdue": activity.overdue or 0,
        "pipeline": activity.pipeline or 0,
        "mtd_target": (mtd_revenue.target or 0) if mtd_revenue else 0,
        "mtd_cleared": (mtd_revenue.cleared or 0) if mtd_revenue else 0,
        "ytd_target": (ytd_revenue.target or 0) if ytd_revenue else 0,
        "ytd_cleared": (ytd_revenue.cleared or 0) if ytd_revenue else 0,
    }