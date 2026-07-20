"""No-LLM smoke tests for the benchmark harness itself: the YAML cases
are well-formed, the fixture dataset seeds, and the scoring functions
behave. Runs in the default suite (unlike test_benchmark.py, which needs
the real LLM and the explicit benchmark marker)."""

from app.database.models import Advisor
from app.llm.metric_ontology import METRICS
from app.llm.query_ir import MetricRef, QueryIR, Sort

from tests.benchmark import fixture_data
from tests.benchmark.test_benchmark import _score_entities, _score_ir, _score_sql, _set_f1, load_cases


def test_all_cases_are_well_formed():
    cases = load_cases()
    assert len(cases) >= 60
    seen_ids = set()
    for case in cases:
        assert case["id"] not in seen_ids, f"duplicate case id {case['id']}"
        seen_ids.add(case["id"])
        assert ("query" in case) != ("turns" in case), f"{case['id']}: exactly one of query/turns"
        assert "expect" in case and case["expect"], f"{case['id']}: missing expect"
        expected_metric = case["expect"].get("ir", {}).get("metric")
        if expected_metric:
            assert expected_metric in METRICS, f"{case['id']}: unknown metric {expected_metric}"
        for f in case["expect"].get("ir", {}).get("filters", []):
            field = f.get("field")
            if field and field not in ("team", "company", "advisor", "attendance_status"):
                assert field in METRICS, f"{case['id']}: unknown filter field {field}"


def test_fixture_seed_loads(db_session):
    fixture_data.seed(db_session)
    assert db_session.query(Advisor).count() == 6


def test_set_f1():
    assert _set_f1({"a"}, {"a"}) == 1.0
    assert _set_f1({"a"}, {"b"}) == 0.0
    assert _set_f1(set(), set()) == 1.0
    assert 0 < _set_f1({"a", "b"}, {"a"}) < 1


def test_score_ir_fragment_subset():
    ir = QueryIR(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="mtd_cleared"),
        sort=Sort(metric="mtd_cleared", direction="desc"),
        limit=5,
    )
    assert _score_ir({"metric": "mtd_cleared", "limit": 5}, ir) == 1.0
    assert _score_ir({"metric": "overdue"}, ir) == 0.0
    assert _score_ir({}, ir) == 1.0
    assert _score_ir({"metric": "mtd_cleared"}, None) == 0.0


def test_score_sql_first_row_tolerance():
    rows = [{"name": "A", "value": 90.0}]
    assert _score_sql({"first_row": {"name": "A", "value": 90}}, rows) == 1.0
    assert _score_sql({"first_row": {"name": "A", "value": 91}}, rows) == 0.0
    assert _score_sql({"row_count": 1}, rows) == 1.0
    assert _score_sql({"row_count": 1}, None) == 0.0


def test_score_entities_dimensions():
    entities = {"teams": ["Blue Area"], "companies": ["Graana"], "advisor_name": "Waqar Haider"}
    expect = {"teams": ["Blue Area"], "companies": ["Graana"], "advisor": "Waqar Haider"}
    assert _score_entities(expect, entities) == 1.0
    assert _score_entities({"teams": ["Downtown"]}, entities) == 0.0
