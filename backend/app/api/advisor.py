from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.dependencies import get_db, get_current_user
from app.services.advisor_service import find_advisor_by_name

router = APIRouter(prefix="/advisor", tags=["advisor"])


@router.get("")
def get_advisor(
    name: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """find_advisor_by_name returns a dict (from the advisor_profile view) or
    None — no ORM object to serialize, so no response_model here yet. Add
    AdvisorOut once database/schemas.py is rewritten to match the view's
    columns."""
    advisor = find_advisor_by_name(db, name)
    if not advisor:
        raise HTTPException(status_code=404, detail=f"No advisor matching '{name}'")
    return advisor