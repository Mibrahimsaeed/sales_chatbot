"""
Turns (text, entities) into a QueryPlan — a structured, executable
representation of what the user wants. This is the layer that replaces
rigid intent categories with something that composes: any (metric, level,
action) combination the ontology declares is automatically plannable.
"""

from dataclasses import dataclass
from html import entities
from app.llm.metric_ontology import METRICS, resolve_metric

LEVEL_KEYWORDS = {
    "advisor": ["advisor", "advisors", "agent", "agents"],
    "team": ["team", "teams"],
    "company": ["company", "companies"],
}
RANKING_KEYWORDS = ["top", "best", "worst", "highest", "lowest", "most", "least", "rank", "leaderboard"]
DESCENDING_DEFAULT_FLIPPED_BY = ["worst", "lowest", "least"]


@dataclass
class QueryPlan:
    action: str                    # "lookup" | "summary" | "leaderboard" | "unresolved"
    level: str | None = None       # "advisor" | "team" | "company"
    entity_value: str | None = None
    metric: str | None = None
    limit: int = 10
    ascending: bool = False
    reason: str = ""               # why "unresolved", for logging/debugging


def _detect_level(text: str) -> str | None:
    q = text.lower()
    for level, keywords in LEVEL_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return level
    return None


def build_query_plan(text: str, entities: dict) -> QueryPlan:
    q = text.lower()
    is_ranking = any(kw in q for kw in RANKING_KEYWORDS) or "give me" in q or "show me" in q

    # A specific advisor name matched with decent confidence, and the phrasing
    # isn't asking for a ranking -> this is a direct lookup, not a leaderboard.
    if entities.get("advisor_name") and entities.get("advisor_match_score", 1.0) >= 0.6 and not is_ranking:
        return QueryPlan(action="lookup", level="advisor", entity_value=entities["advisor_name"])

    # metric = entities.get("metric") or resolve_metric(text)

    # if entities.get("team") and not is_ranking and not metric:
    #     return QueryPlan(action="summary", level="team", entity_value=entities["team"])
    # if entities.get("company") and not is_ranking and not metric:
    #     return QueryPlan(action="summary", level="company", entity_value=entities["company"])

    
    metric = entities.get("metric") or resolve_metric(text)

    # Attendance filtering (e.g. "show not marked people in Blue Area")
    if entities.get("attendance_status"):
        return QueryPlan(
            action="attendance_filter",
            level="advisor",
            entity_value=entities.get("team"),
            reason=entities["attendance_status"],
    )

    if entities.get("team") and not is_ranking and not metric:
         return QueryPlan(action="summary", level="team", entity_value=entities["team"])

    if entities.get("company") and not is_ranking and not metric:
        return QueryPlan(action="summary", level="company", entity_value=entities["company"])

    if metric:
        metric_def = METRICS.get(metric)
        if not metric_def:
            return QueryPlan(action="unresolved", reason=f"metric '{metric}' not in ontology")

        level = _detect_level(q) or metric_def.primary_level
        if level not in metric_def.entity_levels:
            level = metric_def.primary_level  # requested level has no resolver — fall back gracefully

        ascending = any(kw in q for kw in DESCENDING_DEFAULT_FLIPPED_BY)
        limit = entities.get("limit", 10)
        return QueryPlan(action="leaderboard", level=level, metric=metric, limit=limit, ascending=ascending)

    # no metric resolved — fall back to whatever named entity we did find
    if entities.get("advisor_name"):
        return QueryPlan(action="lookup", level="advisor", entity_value=entities["advisor_name"])
    if entities.get("team"):
        return QueryPlan(action="summary", level="team", entity_value=entities["team"])
    if entities.get("company"):
        return QueryPlan(action="summary", level="company", entity_value=entities["company"])

    return QueryPlan(action="unresolved", reason="no metric or entity matched")