from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database.models import Advisor, SalesFunnel, Pipeline, TeamTarget
from app.core.exception import NotFoundError


def get_team_summary(db: Session, team: str) -> dict:
    """Two independent pieces, both real:
    1. team_targets — genuine team-level source (Target Achievement tab)
    2. live rollup of advisor-level activity for that team
    They are NOT the same query — don't derive one from the other."""

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
        .filter(Advisor.team.ilike(team))
        .first()
    )

    if not target_row and not (activity and activity.advisors):
        raise NotFoundError(f"No team matching '{team}'")

    return {
        "team": team,
        "advisors": activity.advisors or 0 if activity else 0,
        "connects": activity.connects or 0 if activity else 0,
        "overdue": activity.overdue or 0 if activity else 0,
        "pipeline": activity.pipeline or 0 if activity else 0,
        "target": target_row.target if target_row else None,
        "achieved": target_row.achieved if target_row else None,
        "achievement_pct": target_row.achievement_pct if target_row else None,
    }