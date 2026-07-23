"""
Response planner (Part 8). Makes response-shape selection an explicit,
testable step instead of the implicit ir.intent -> formatter dispatch
response_formatter.py used to do directly. Same formatter functions run
either way (format_ir_leaderboard_reply etc.) — this only decides WHICH
one, plus whether within-result insights (narrative.compute_insights) are
worth attaching, from ir.intent + row count rather than intent alone:
a leaderboard that resolved to exactly one row reads better as a single
value than a "Top 1" list, and an empty result never wants insights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.llm.query_ir import QueryIR

Shape = Literal["single_value", "ranked_list", "comparison_table", "filtered_table", "empty"]


@dataclass
class ResponsePlan:
    shape: Shape
    show_insights: bool


def plan_response(ir: QueryIR, rows: list[dict]) -> ResponsePlan:
    if not rows:
        return ResponsePlan(shape="empty", show_insights=False)

    if ir.intent == "comparison":
        return ResponsePlan(shape="comparison_table", show_insights=len(rows) >= 3)

    if ir.intent == "filtered_list":
        return ResponsePlan(shape="filtered_table", show_insights=len(rows) >= 3)

    # leaderboard (and anything else defaulting to the leaderboard shape)
    if len(rows) == 1:
        return ResponsePlan(shape="single_value", show_insights=False)
    return ResponsePlan(shape="ranked_list", show_insights=len(rows) >= 3)
