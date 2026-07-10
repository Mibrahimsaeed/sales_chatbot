from app.services.query_executor import execute_query
from app.services.test_query_planner import QueryPlan


class FakeDB:
    pass



def test_greeting_execution():

    plan = QueryPlan(
        query_name="greeting",
        params={}
    )


    result = execute_query(
        FakeDB(),
        plan
    )


    assert result["query"] == "greeting"
    assert "message" in result["data"]



def test_unknown_execution():

    plan = QueryPlan(
        query_name="unknown",
        params={}
    )


    result = execute_query(
        FakeDB(),
        plan
    )


    assert result["query"] == "unknown"


