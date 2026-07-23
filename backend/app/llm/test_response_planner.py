from app.llm.query_ir import MetricRef, QueryIR, Sort
from app.llm.response_planner import plan_response


def _ir(intent="leaderboard", **overrides):
    base = dict(
        intent=intent,
        subject_level="advisor",
        metric=MetricRef(key="mtd_cleared"),
        sort=Sort(metric="mtd_cleared", direction="desc"),
        limit=10,
    )
    base.update(overrides)
    return QueryIR(**base)


ROW = {"wid": 1, "name": "A", "value": 100.0}


def test_empty_rows_are_empty_shape_regardless_of_intent():
    plan = plan_response(_ir(), [])
    assert plan.shape == "empty"
    assert plan.show_insights is False


def test_single_leaderboard_row_is_single_value():
    plan = plan_response(_ir(), [ROW])
    assert plan.shape == "single_value"
    assert plan.show_insights is False


def test_multi_row_leaderboard_is_ranked_list():
    rows = [ROW, {"wid": 2, "name": "B", "value": 50.0}]
    plan = plan_response(_ir(), rows)
    assert plan.shape == "ranked_list"


def test_comparison_intent_is_comparison_table():
    rows = [ROW, {"wid": 2, "name": "B", "value": 50.0}]
    plan = plan_response(_ir(intent="comparison"), rows)
    assert plan.shape == "comparison_table"


def test_filtered_list_intent_is_filtered_table():
    rows = [ROW, {"wid": 2, "name": "B", "value": 50.0}]
    plan = plan_response(_ir(intent="filtered_list"), rows)
    assert plan.shape == "filtered_table"


def test_insights_only_suggested_with_at_least_three_rows():
    two_rows = [ROW, {"wid": 2, "name": "B", "value": 50.0}]
    three_rows = two_rows + [{"wid": 3, "name": "C", "value": 25.0}]
    assert plan_response(_ir(), two_rows).show_insights is False
    assert plan_response(_ir(), three_rows).show_insights is True
