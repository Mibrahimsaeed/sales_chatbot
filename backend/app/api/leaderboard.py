from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_user
from app.llm.metric_ontology import METRICS
from app.llm.query_compiler import compile_and_run
from app.llm.query_ir import QueryIR, MetricRef, Sort

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


@router.get("")
def leaderboard(
    metric: str = Query(..., description="any metric key from the ontology, e.g. mtd_cleared | total_connects | overdue | achievement_pct"),
    level: str = Query("advisor", description="advisor | team"),
    limit: int = Query(10, ge=1, le=50),
    ascending: bool = Query(False),
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

    ir = QueryIR(
        intent="leaderboard",
        subject_level=level,
        metric=MetricRef(key=metric),
        sort=Sort(metric=metric, direction="asc" if ascending else "desc"),
        limit=limit,
    )

    rows = compile_and_run(db, ir)
    if rows is None:
        raise HTTPException(status_code=400, detail=f"Unsupported metric '{metric}' at level '{level}'")

    return rows