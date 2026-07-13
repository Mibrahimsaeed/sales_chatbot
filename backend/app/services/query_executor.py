# """
# Query Executor

# Takes a QueryPlan from query_planner.py
# and routes execution to the correct service.

# Flow:

# QueryPlan
#     |
#     ↓
# QueryExecutor
#     |
#     ├── leaderboard_service
#     ├── advisor_service
#     └── attendance_service

# """


# from sqlalchemy.orm import Session

# from app.services.leaderboard_service import get_leaderboard
# from app.services.advisor_service import find_advisor_by_name
# from app.services.attendance_service import get_attendance_issues

# from app.services.query_planner import QueryPlan



# def execute_query(
#     db: Session,
#     plan: QueryPlan
# ) -> dict:

#     """
#     Execute a generated query plan.

#     Returns:
#         {
#             "query": query_name,
#             "data": result
#         }
#     """

#     query_name = plan.query_name
#     params = plan.params


#     # -----------------------------
#     # Leaderboard
#     # -----------------------------
#     if query_name == "leaderboard":

#         result = get_leaderboard(
#             db=db,
#             metric=params.get("metric"),
#             limit=params.get("limit", 5)
#         )

#         return {
#             "query": query_name,
#             "data": result
#         }



#     # -----------------------------
#     # Advisor Profile
#     # -----------------------------
#     if query_name == "advisor_profile":

#         result = find_advisor_by_name(
#             db=db,
#             query=params.get("advisor_name")
#         )

#         return {
#             "query": query_name,
#             "data": result
#         }



#     # -----------------------------
#     # Attendance
#     # -----------------------------
#     if query_name == "attendance_summary":

#         result = get_attendance_issues(
#             db=db,
#             team=params.get("team"),
#             limit=params.get("limit", 15)
#         )

#         return {
#             "query": query_name,
#             "data": result
#         }



#     # -----------------------------
#     # Greeting
#     # -----------------------------
#     if query_name == "greeting":

#         return {
#             "query": query_name,
#             "data": {
#                 "message": "Hello! How can I help you with sales analytics?"
#             }
#         }



#     # -----------------------------
#     # Unknown
#     # -----------------------------
#     return {
#         "query": "unknown",
#         "data": None
#     }


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

NOTE: QueryPlan (see query_planner.py) is a dataclass with fields
`action`, `level`, `entity_value`, `metric`, `limit`, `ascending`, `reason`.
There is no `query_name` and no `params` dict on this object — dispatch
below reads `plan.action` and the individual fields directly.
"""


from sqlalchemy.orm import Session

from app.services.leaderboard_service import get_leaderboard
from app.services.advisor_service import find_advisor_by_name
from app.services.attendance_service import (
    get_attendance_issues,
    get_attendance_by_status,
)

from app.llm.query_planner import QueryPlan


def execute_query(
    db: Session,
    plan: QueryPlan
) -> dict:

    """
    Execute a generated query plan.

    Returns:
        {
            "query": action,
            "data": result
        }
    """

    action = plan.action


    # -----------------------------
    # Lookup (e.g. a specific advisor by name)
    # -----------------------------
    if action == "lookup" and plan.level == "advisor":

        result = find_advisor_by_name(
            db=db,
            query=plan.entity_value
        )

        return {
            "query": action,
            "data": result
        }



    # -----------------------------
    # Attendance Filter
    # (e.g. "Give all people that are not marked in Blue Area")
    #
    # entity_value -> team   (e.g. "Blue Area", may be None if no team named)
    # reason       -> status (e.g. "Not Marked")
    # -----------------------------
    if action == "attendance_filter":

        result = get_attendance_by_status(
            db=db,
            team=plan.entity_value,
            status=plan.reason
        )

        return {
            "query": action,
            "data": result
        }



    # -----------------------------
    # Summary (team or company level)
    #
    # Only a team-level attendance summary has a wired-up service right now
    # (get_attendance_issues). Company-level summary has no backing service
    # in the code shared so far — returns a clear "not implemented" payload
    # instead of guessing at a call that doesn't exist, so this fails loud
    # and visibly rather than crashing or silently returning wrong data.
    # -----------------------------
    if action == "summary":

        if plan.level == "team":
            result = get_attendance_issues(
                db=db,
                team=plan.entity_value,
                limit=plan.limit
            )
            return {
                "query": action,
                "data": result
            }

        if plan.level == "company":
            return {
                "query": action,
                "data": None,
                "error": (
                    f"Company-level summary for '{plan.entity_value}' is not "
                    "yet implemented — no service currently handles this."
                )
            }



    # -----------------------------
    # Leaderboard
    #
    # ASSUMPTION: get_leaderboard() accepts level/ascending in addition to
    # metric/limit — the planner now generates both, but I haven't seen
    # leaderboard_service.py to confirm its signature. If it only takes
    # (db, metric, limit), drop the level/ascending kwargs below.
    # -----------------------------
    if action == "leaderboard":

        result = get_leaderboard(
            db=db,
            metric=plan.metric,
            level=plan.level,
            limit=plan.limit,
            ascending=plan.ascending
        )

        return {
            "query": action,
            "data": result
        }



    # -----------------------------
    # Greeting
    #
    # build_query_plan() never returns action == "greeting" in the version
    # of query_planner.py reviewed here. Kept in case greetings are
    # intercepted upstream and a QueryPlan(action="greeting", ...) is
    # constructed directly somewhere else — remove if that's not the case.
    # -----------------------------
    if action == "greeting":

        return {
            "query": action,
            "data": {
                "message": "Hello! How can I help you with sales analytics?"
            }
        }



    # -----------------------------
    # Unresolved / unknown
    #
    # Surface plan.reason instead of discarding it — it's the debugging
    # signal query_planner.py deliberately attached (e.g. "metric 'x' not
    # in ontology", "no metric or entity matched").
    # -----------------------------
    return {
        "query": "unresolved",
        "data": None,
        "reason": plan.reason or "no matching handler for this query plan"
    }