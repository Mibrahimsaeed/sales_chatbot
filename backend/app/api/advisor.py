from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_user
from app.services.advisor_service import find_advisor_by_wid, resolve_advisor

router = APIRouter(prefix="/advisor", tags=["advisor"])


@router.get("")
def get_advisor(
    name: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Look an advisor up by name.

    Phase 2 entity resolution: a name is not an identifier here — 238
    name groups in production map to more than one real person. This
    endpoint previously returned the lowest-wid substring match, so
    ?name=Ahmed%20Ali could return "Ahmed Ali Pirzada" and 7 of the 8
    people named "Yasir Ali" were unreachable entirely.

    Now:
      200 — exactly one person matched (the profile)
      409 — several people matched; the candidates are returned so the
            caller can pick one and re-request by wid. 409 rather than
            200-with-a-list so a client that ignores the ambiguity can't
            mistake a candidate list for a resolved advisor.
      404 — nothing matched
    """
    resolution = resolve_advisor(db, name)

    if resolution.is_ambiguous:
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"'{name}' matches {len(resolution.candidates)} advisors — specify which by wid",
                "candidates": resolution.to_dict()["candidates"],
            },
        )

    if not resolution.is_resolved:
        raise HTTPException(status_code=404, detail=f"No advisor matching '{name}'")

    advisor = find_advisor_by_wid(db, resolution.wid)
    if not advisor:
        raise HTTPException(status_code=404, detail=f"No advisor matching '{name}'")
    return advisor


@router.get("/{wid}")
def get_advisor_by_wid(
    wid: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Fetch by wid — the unambiguous form, and how a client resolves the
    409 above."""
    advisor = find_advisor_by_wid(db, wid)
    if not advisor:
        raise HTTPException(status_code=404, detail=f"No advisor with wid {wid}")
    return advisor
