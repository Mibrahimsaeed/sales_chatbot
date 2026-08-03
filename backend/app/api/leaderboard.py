from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_user
from app.llm import hierarchy
from app.llm.metric_ontology import METRICS
from app.llm.query_compiler import compile_and_run, default_direction
from app.llm.query_ir import QueryIR, MetricRef, Sort

# Derived so the OpenAPI docs cannot describe a level set the system no
# longer has — the hand-written string still offered `business_center`
# and omitted bcm/office/region.
def _levels_description() -> str:
    from app.llm import hierarchy

    return " | ".join(hierarchy.HIERARCHY_LEVELS)


_LEVEL_DESCRIPTION = _levels_description()

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])

_VALID_LEVELS = hierarchy.GROUP_LEVELS + ["advisor"]


@router.get("")
def leaderboard(
    metric: str = Query(..., description="any metric key from the ontology, e.g. mtd_cleared | total_connects | overdue | achievement_pct"),
    level: str = Query("advisor", description=_LEVEL_DESCRIPTION),
    limit: int = Query(10, ge=1, le=50),
    ascending: bool | None = Query(
        None,
        description="omit to use the metric's own polarity — a lower-is-better "
                    "metric (overdue, late count) then ranks ascending",
    ),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Same generic compiler the chat pipeline uses (query_compiler.py) —
    this route and the chatbot's leaderboard queries now share one source
    of truth instead of leaderboard_service.py's separate, narrower
    METRIC_MAP (which only ever covered mtd_cleared/mtd_new_connect/overdue
    and didn't accept `level`, per the architecture review's Root Cause
    #6/#9 finding)."""
    if metric not in METRICS:
        raise HTTPException(status_code=400, detail=f"Unsupported metric: {metric}")
    # Validated up front (hierarchy rework phase 2) so an invalid level 400s
    # immediately with a clear message, instead of only surfacing further
    # down as an indirect "unsupported metric at that level" once compile_
    # and_run returns None for a level query_compiler doesn't even recognize.
    if level not in _VALID_LEVELS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported level '{level}' — must be one of: {', '.join(_VALID_LEVELS)}"
        )

    ir = QueryIR(
        intent="leaderboard",
        subject_level=level,
        metric=MetricRef(key=metric),
        sort=Sort(
            metric=metric,
            direction=default_direction(metric) if ascending is None
                      else ("asc" if ascending else "desc"),
        ),
        limit=limit,
    )

    rows = compile_and_run(db, ir)
    if rows is None:
        raise HTTPException(status_code=400, detail=f"Unsupported metric '{metric}' at level '{level}'")

    return rows
