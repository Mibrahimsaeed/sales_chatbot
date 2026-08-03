from sqlalchemy.orm import Session

from app.core.exception import NotFoundError
from app.database.models import TeamTarget
from app.llm import aggregation


def get_team_summary(db: Session, team: str) -> dict:
    """A team summary is TWO things, and they are not the same number.

    1. The ROLL-UP of the team's advisors — connects, pipeline, overdue,
       MTD/YTD target and cleared. PHASE 4: this used to be three
       hand-written `func.sum` queries here, near-identical to the ones in
       company_service and hierarchy_service. It now comes from the
       aggregation engine, so a team's connects are the same number
       whether they are asked for here, in a comparison, or in a
       leaderboard. They were not always.
    2. The team's OWN figure from the Target Achievement sheet
       (`TeamTarget`). This is a genuine independent source, not a
       derivation of (1) — it is what the business published, and it can
       legitimately disagree with the roll-up. Which one is authoritative
       is still an open business question (Phase 1, Q3), so this reads it
       explicitly and keeps it under its own keys rather than letting it
       masquerade as an aggregate.
    """
    rollup = aggregation.summary(db, "team", team)

    target_row = (
        db.query(TeamTarget)
        .filter(TeamTarget.team.ilike(team))
        .first()
    )

    if not target_row and not rollup["advisors"]:
        raise NotFoundError(f"No team matching '{team}'")

    return {
        "team": team,
        "advisors": rollup["advisors"],
        "connects": rollup["connects"],
        "overdue": rollup["overdue"],
        "pipeline": rollup["pipeline"],
        # Sheet figures — source (2) above. Kept under the exact keys the
        # API already returns.
        "target": target_row.target if target_row else None,
        "achieved": target_row.achieved if target_row else None,
        "achievement_pct": target_row.achievement_pct if target_row else None,
        "mtd_target": rollup["mtd_target"],
        "mtd_cleared": rollup["mtd_cleared"],
        "ytd_target": rollup["ytd_target"],
        "ytd_cleared": rollup["ytd_cleared"],
    }
