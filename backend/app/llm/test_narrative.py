from app.llm import narrative
from app.llm.narrative import compute_facts, compute_insights, polish_reply, _numbers_in
from app.llm.query_ir import Filter, MetricRef, QueryIR, Sort


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


ROWS = [
    {"wid": 1, "name": "Waqar Haider", "team": "Blue Area", "company": "Graana", "value": 900.0},
    {"wid": 2, "name": "Ali Raza", "team": "Downtown", "company": "IMARAT", "value": 500.0},
    {"wid": 3, "name": "Sana Khan", "team": "Downtown", "company": "IMARAT", "value": 100.0},
]


def test_leaderboard_facts_are_deterministic():
    facts = compute_facts(_ir(), ROWS)
    assert facts["top"] == {"name": "Waqar Haider", "value": 900.0}
    assert facts["bottom"] == {"name": "Sana Khan", "value": 100.0}
    assert facts["average"] == 500.0
    assert facts["spread"] == 800.0
    assert facts["result_count"] == 3
    assert facts["metric"] == "MTD Revenue Cleared"


def test_comparison_facts_include_winner_and_lead():
    facts = compute_facts(_ir(intent="comparison"), ROWS[:2])
    assert facts["winner"] == "Waqar Haider"
    assert facts["lead"] == 400.0
    assert facts["lead_pct"] == 80.0


def test_empty_rows_produce_count_only():
    facts = compute_facts(_ir(), [])
    assert facts["result_count"] == 0
    assert "top" not in facts


def test_filters_are_restated_in_facts():
    ir = _ir(filters=[Filter(field="attendance_rate", operator=">", value=90)])
    facts = compute_facts(ir, ROWS)
    assert facts["filters"] == [{"field": "attendance_rate", "operator": ">", "value": 90}]


def test_polish_accepts_summary_using_only_fact_numbers(monkeypatch):
    monkeypatch.setattr(narrative.settings, "nlu_narrative", True)
    monkeypatch.setattr(
        narrative, "call_llm_json",
        lambda prompt: {"summary": "Waqar Haider leads with 900, 800 ahead of the lowest."},
    )
    facts = compute_facts(_ir(), ROWS)
    reply = polish_reply(facts, "TEMPLATE")
    assert reply.startswith("Waqar Haider leads")
    assert reply.endswith("TEMPLATE")


def test_polish_rejects_invented_numbers(monkeypatch):
    monkeypatch.setattr(narrative.settings, "nlu_narrative", True)
    monkeypatch.setattr(
        narrative, "call_llm_json",
        lambda prompt: {"summary": "Waqar Haider leads with a stunning 9999."},
    )
    facts = compute_facts(_ir(), ROWS)
    assert polish_reply(facts, "TEMPLATE") == "TEMPLATE"


def test_polish_fails_soft_on_llm_failure(monkeypatch):
    monkeypatch.setattr(narrative.settings, "nlu_narrative", True)
    monkeypatch.setattr(narrative, "call_llm_json", lambda prompt: None)
    facts = compute_facts(_ir(), ROWS)
    assert polish_reply(facts, "TEMPLATE") == "TEMPLATE"


def test_polish_disabled_by_flag(monkeypatch):
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    called = []
    monkeypatch.setattr(narrative, "call_llm_json", lambda prompt: called.append(1))
    assert polish_reply({"a": 1}, "TEMPLATE") == "TEMPLATE"
    assert called == []


def test_number_normalization_treats_90_and_90_point_0_as_equal():
    assert _numbers_in({"v": 90.0}) == _numbers_in("attendance is 90")


# ---- compute_insights (Part 8) ----

def test_no_insights_when_values_are_reasonably_uniform():
    assert compute_insights(_ir(), ROWS) == []


def test_flags_outlier_far_above_group_average():
    skewed_rows = [
        {"wid": 1, "name": "A", "value": 10.0},
        {"wid": 2, "name": "B", "value": 10.0},
        {"wid": 3, "name": "C", "value": 10.0},
        {"wid": 4, "name": "D", "value": 100.0},
    ]
    insights = compute_insights(_ir(), skewed_rows)
    assert len(insights) == 1
    assert "D" in insights[0]
    assert "above" in insights[0]


def test_requires_at_least_three_values():
    assert compute_insights(_ir(), ROWS[:2]) == []


def test_zero_spread_produces_no_insights():
    flat_rows = [{"wid": i, "name": str(i), "value": 50.0} for i in range(4)]
    assert compute_insights(_ir(), flat_rows) == []


def test_insight_quotes_the_actual_row_values():
    skewed_rows = [
        {"wid": 1, "name": "A", "value": 10.0},
        {"wid": 2, "name": "B", "value": 10.0},
        {"wid": 3, "name": "C", "value": 10.0},
        {"wid": 4, "name": "D", "value": 100.0},
    ]
    insights = compute_insights(_ir(), skewed_rows)
    used = _numbers_in(insights)
    # the outlier's own quoted value must be traceable back to real row data
    assert "100" in used
