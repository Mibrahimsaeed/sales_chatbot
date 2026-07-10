from app.services.query_planner import build_query_plan


class MockIntent:
    def __init__(self, intent, entities):
        self.intent = intent
        self.entities = entities



def test_leaderboard_plan():

    intent = MockIntent(
        "leaderboard",
        {
            "metric": "mtd_cleared",
            "limit": 5
        }
    )


    plan = build_query_plan(intent)


    assert plan.query_name == "leaderboard"
    assert plan.params["metric"] == "mtd_cleared"
    assert plan.params["limit"] == 5



def test_advisor_lookup_plan():

    intent = MockIntent(
        "advisor_lookup",
        {
            "advisor_name": "Waqar Haider"
        }
    )


    plan = build_query_plan(intent)


    assert plan.query_name == "advisor_profile"
    assert plan.params["advisor_name"] == "Waqar Haider"


    