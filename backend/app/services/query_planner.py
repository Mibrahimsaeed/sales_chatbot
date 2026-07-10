"""
Query Planner

Converts NLU intent results into executable business query plans.

Flow:

User Question
      |
      ↓
Intent Detector
      |
      ↓
Query Planner
      |
      ↓
Service Layer
      |
      ↓
Database
"""


from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryPlan:
    """
    Represents a database operation plan.
    """

    query_name: str
    params: dict[str, Any] = field(default_factory=dict)


def build_query_plan(intent_result) -> QueryPlan:
    """
    Convert intent classification result into a query plan.

    Parameters:
        intent_result:
            Output object from intent_detector.classify_intent()

    Returns:
        QueryPlan object
    """

    intent = intent_result.intent
    entities = intent_result.entities or {}


    # ----------------------------
    # Leaderboard Queries
    # ----------------------------
    if intent == "leaderboard":

        return QueryPlan(
            query_name="leaderboard",
            params={
                "metric": entities.get("metric", "mtd_cleared"),
                "limit": entities.get("limit", 5),
                "team": entities.get("team"),
                "company": entities.get("company"),
            }
        )


    # ----------------------------
    # Advisor Profile Lookup
    # ----------------------------
    if intent == "advisor_lookup":

        return QueryPlan(
            query_name="advisor_profile",
            params={
                "advisor_name": entities.get("advisor_name"),
                "advisor_wid": entities.get("advisor_wid"),
            }
        )


    # ----------------------------
    # Attendance Queries
    # ----------------------------
    if intent == "attendance_check":

        return QueryPlan(
            query_name="attendance_summary",
            params={
                "date": entities.get("date"),
                "advisor_name": entities.get("advisor_name"),
                "team": entities.get("team"),
            }
        )


    # ----------------------------
    # Team Performance
    # ----------------------------
    if intent == "team_performance":

        return QueryPlan(
            query_name="team_performance",
            params={
                "team": entities.get("team"),
                "metric": entities.get(
                    "metric",
                    "mtd_cleared"
                ),
            }
        )


    # ----------------------------
    # Company Performance
    # ----------------------------
    if intent == "company_performance":

        return QueryPlan(
            query_name="company_performance",
            params={
                "company": entities.get("company"),
                "metric": entities.get(
                    "metric",
                    "mtd_cleared"
                ),
            }
        )


    # ----------------------------
    # Greeting / Conversation
    # ----------------------------
    if intent == "greeting":

        return QueryPlan(
            query_name="greeting",
            params={}
        )


    # ----------------------------
    # Unknown Intent
    # ----------------------------
    return QueryPlan(
        query_name="unknown",
        params={}
    )

