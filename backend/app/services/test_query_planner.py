"""
Locks in the two failing queries from the bug report — both should now
resolve deterministically through the ontology + planner, with zero LLM
calls involved.
"""

from app.llm.query_planner import build_query_plan
from app.llm.metric_ontology import resolve_metric


def test_target_achievement_metric_resolves():
    assert resolve_metric("give me target achievement") == "achievement_pct"


def test_give_me_target_achievement_plans_team_leaderboard():
    plan = build_query_plan("give me target achievement", {})
    assert plan.action == "leaderboard"
    assert plan.metric == "achievement_pct"
    assert plan.level == "team"   # primary_level default when no level keyword present


def test_top_5_advisors_by_target_achievement():
    entities = {"limit": 5}
    plan = build_query_plan("top 5 advisors by target achievement", entities)
    assert plan.action == "leaderboard"
    assert plan.metric == "achievement_pct"
    assert plan.level == "advisor"   # "advisors" keyword overrides the metric's team default
    assert plan.limit == 5


def test_worst_overdue_flips_to_ascending_false_is_still_a_ranking():
    plan = build_query_plan("worst overdue teams", {})
    assert plan.action == "leaderboard"
    assert plan.metric == "overdue"
    assert plan.ascending is True   # "worst" flips the default order


def test_no_metric_no_entity_is_unresolved():
    plan = build_query_plan("asdkjalksjd nonsense", {})
    assert plan.action == "unresolved"