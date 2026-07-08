from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.core.dependencies import get_db, get_current_user
from app.services.leaderboard_service import get_leaderboard
from app.core.exception import UnsupportedMetricError

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


@router.get("")
def leaderboard(
    metric: str = Query(..., description="mtd_cleared | mtd_new_connect | overdue"),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """get_leaderboard raises UnsupportedMetricError for anything outside its
    METRIC_MAP plus the special-cased 'mtd_cleared' — surface that as a 400,
    since it's a bad request param, not a missing resource."""
    try:
        return get_leaderboard(db, metric, limit)
    except UnsupportedMetricError as e:
        raise HTTPException(status_code=400, detail=e.message)