"""
Query Executor

Takes a QueryPlan from query_planner.py
and routes execution to the correct service.

Flow:

QueryPlan
    |
    ↓
QueryExecutor
    |
    ├── leaderboard_service
    ├── advisor_service
    └── attendance_service

"""


from sqlalchemy.orm import Session

from app.services.leaderboard_service import get_leaderboard
from app.services.advisor_service import find_advisor_by_name
from app.services.attendance_service import get_attendance_issues

from app.services.query_planner import QueryPlan



def execute_query(
    db: Session,
    plan: QueryPlan
) -> dict:

    """
    Execute a generated query plan.

    Returns:
        {
            "query": query_name,
            "data": result
        }
    """

    query_name = plan.query_name
    params = plan.params


    # -----------------------------
    # Leaderboard
    # -----------------------------
    if query_name == "leaderboard":

        result = get_leaderboard(
            db=db,
            metric=params.get("metric"),
            limit=params.get("limit", 5)
        )

        return {
            "query": query_name,
            "data": result
        }



    # -----------------------------
    # Advisor Profile
    # -----------------------------
    if query_name == "advisor_profile":

        result = find_advisor_by_name(
            db=db,
            query=params.get("advisor_name")
        )

        return {
            "query": query_name,
            "data": result
        }



    # -----------------------------
    # Attendance
    # -----------------------------
    if query_name == "attendance_summary":

        result = get_attendance_issues(
            db=db,
            team=params.get("team"),
            limit=params.get("limit", 15)
        )

        return {
            "query": query_name,
            "data": result
        }



    # -----------------------------
    # Greeting
    # -----------------------------
    if query_name == "greeting":

        return {
            "query": query_name,
            "data": {
                "message": "Hello! How can I help you with sales analytics?"
            }
        }



    # -----------------------------
    # Unknown
    # -----------------------------
    return {
        "query": "unknown",
        "data": None
    }