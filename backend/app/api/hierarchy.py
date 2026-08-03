from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_user
from app.core.exception import NotFoundError
from app.llm import hierarchy
from app.services import hierarchy_service

router = APIRouter(prefix="/hierarchy", tags=["hierarchy"])


@router.get("/{level}")
def get_hierarchy_entity(
    level: str,
    value: str = Query(..., description="the entity's name at this level, e.g. a unit head's name"),
    flat: bool = Query(False, description="ungrouped advisor list instead of nested-by-team"),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Generic REST access for any hierarchy level (company/unit_head/
    zonal_head/business_center/team) — same hierarchy_service functions the
    chat pipeline uses, so this and the chatbot's breakdown queries share
    one source of truth. `advisor` is deliberately not one of the accepted
    levels here — advisor lookup already has its own endpoint
    (GET /advisor?name=)."""
    if level not in hierarchy.GROUP_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported level '{level}' — must be one of: {', '.join(hierarchy.GROUP_LEVELS)}",
        )

    try:
        if flat:
            return hierarchy_service.get_level_flat_list(db, level, value)
        return hierarchy_service.get_level_breakdown(db, level, value)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
