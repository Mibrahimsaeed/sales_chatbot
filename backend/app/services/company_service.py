from sqlalchemy.orm import Session

from app.core.exception import NotFoundError
from app.llm import aggregation


def get_company_summary(db: Session, company: str) -> dict:
    """PHASE 4: the roll-up comes from the aggregation engine.

    This was a near-copy of team_service's summary — the same three
    hand-written `func.sum` queries with `Advisor.company` swapped for
    `Advisor.team`. A company has no independent sheet source the way a
    team has `TeamTarget`, so unlike team_service this is the roll-up and
    nothing else.
    """
    rollup = aggregation.summary(db, "company", company)

    if not rollup["advisors"]:
        raise NotFoundError(f"No company matching '{company}'")

    return {
        "company": company,
        "advisors": rollup["advisors"],
        "connects": rollup["connects"],
        "overdue": rollup["overdue"],
        "pipeline": rollup["pipeline"],
        "mtd_target": rollup["mtd_target"],
        "mtd_cleared": rollup["mtd_cleared"],
        "ytd_target": rollup["ytd_target"],
        "ytd_cleared": rollup["ytd_cleared"],
    }
