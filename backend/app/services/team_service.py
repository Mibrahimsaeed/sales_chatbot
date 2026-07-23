from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database.models import Advisor, SalesFunnel, Pipeline, Performance, PerformancePeriod, TeamTarget
from app.core.exception import NotFoundError


def get_team_summary(db: Session, team: str) -> dict:
    """Three independent pieces, all real:
    1. team_targets — genuine team-level source (Target Achievement tab)
    2. live rollup of advisor-level activity for that team
    3. live rollup of advisor-level Performance (MTD + YTD cleared/target)
    None of these are the same query — don't derive one from another."""

    target_row = (
        db.query(TeamTarget)
        .filter(TeamTarget.team.ilike(team))
        .first()
    )

    activity = (
        db.query(
            func.count(Advisor.wid).label("advisors"),
            func.sum(SalesFunnel.mtd_new_connect + SalesFunnel.mtd_followup_connect).label("connects"),
            func.sum(Pipeline.overdue).label("overdue"),
            func.sum(Pipeline.pipeline).label("pipeline"),
        )
        .join(SalesFunnel, SalesFunnel.wid == Advisor.wid, isouter=True)
        .join(Pipeline, Pipeline.wid == Advisor.wid, isouter=True)
        .filter(Advisor.team.ilike(team), Advisor.in_master_sheet.is_(True))
        .first()
    )

    if not target_row and not (activity and activity.advisors):
        raise NotFoundError(f"No team matching '{team}'")

    def _revenue(period):
        return (
            db.query(
                func.sum(Performance.target).label("target"),
                func.sum(Performance.cleared).label("cleared"),
            )
            .join(Advisor, Advisor.wid == Performance.wid)
            .filter(Advisor.team.ilike(team), Advisor.in_master_sheet.is_(True), Performance.period == period)
            .first()
        )

    # bug fix: team summaries never queried Performance at all — "YTD
    # performance" for a team had nothing to report regardless of what
    # was asked, only TeamTarget's single ambiguous-period figure.
    mtd_revenue = _revenue(PerformancePeriod.MTD)
    ytd_revenue = _revenue(PerformancePeriod.YTD)

    return {
        "team": team,
        "advisors": activity.advisors or 0 if activity else 0,
        "connects": activity.connects or 0 if activity else 0,
        "overdue": activity.overdue or 0 if activity else 0,
        "pipeline": activity.pipeline or 0 if activity else 0,
        "target": target_row.target if target_row else None,
        "achieved": target_row.achieved if target_row else None,
        "achievement_pct": target_row.achievement_pct if target_row else None,
        "mtd_target": (mtd_revenue.target or 0) if mtd_revenue else 0,
        "mtd_cleared": (mtd_revenue.cleared or 0) if mtd_revenue else 0,
        "ytd_target": (ytd_revenue.target or 0) if ytd_revenue else 0,
        "ytd_cleared": (ytd_revenue.cleared or 0) if ytd_revenue else 0,
    }