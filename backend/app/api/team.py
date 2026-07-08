from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.dependencies import get_db, get_current_user
from app.services.team_service import get_team_summary
from app.core.exception import NotFoundError

router = APIRouter(prefix="/team", tags=["team"])


@router.get("")
def get_team(
    name: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """get_team_summary raises NotFoundError (not a None return) when neither
    team_targets nor any advisor rows match — translate that to a 404 here."""
    try:
        return get_team_summary(db, name)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)