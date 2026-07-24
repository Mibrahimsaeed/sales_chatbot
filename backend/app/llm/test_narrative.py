from app.llm import narrative
from app.llm.narrative import (
    build_explanation,
    compute_facts,
    compute_insights,
    compute_trends,
    explain_comparison,
    explain_subject,
    polish_explanation,
    _numbers_in,
)
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


def test_polish_accepts_rewrite_using_only_existing_numbers(monkeypatch):
    monkeypatch.setattr(narrative.settings, "nlu_narrative", True)
    monkeypatch.setattr(
        narrative, "call_llm_json",
        lambda prompt: {"summary": "Waqar Haider leads with 900, 800 ahead of the lowest."},
    )
    facts = compute_facts(_ir(), ROWS)
    explanation = "Waqar Haider has 900 MTD Revenue Cleared, ranking 1st of 3 advisors shown."
    polished = polish_explanation(explanation, facts)
    assert polished == "Waqar Haider leads with 900, 800 ahead of the lowest."


def test_polish_rejects_invented_numbers(monkeypatch):
    monkeypatch.setattr(narrative.settings, "nlu_narrative", True)
    monkeypatch.setattr(
        narrative, "call_llm_json",
        lambda prompt: {"summary": "Waqar Haider leads with a stunning 9999."},
    )
    facts = compute_facts(_ir(), ROWS)
    explanation = "Waqar Haider has 900 MTD Revenue Cleared, ranking 1st of 3 advisors shown."
    assert polish_explanation(explanation, facts) == explanation


def test_polish_fails_soft_on_llm_failure(monkeypatch):
    monkeypatch.setattr(narrative.settings, "nlu_narrative", True)
    monkeypatch.setattr(narrative, "call_llm_json", lambda prompt: None)
    explanation = "Waqar Haider has 900 MTD Revenue Cleared."
    assert polish_explanation(explanation, {}) == explanation


def test_polish_disabled_by_flag(monkeypatch):
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    called = []
    monkeypatch.setattr(narrative, "call_llm_json", lambda prompt: called.append(1))
    assert polish_explanation("Waqar Haider has 900.", {}) == "Waqar Haider has 900."
    assert called == []


def test_polish_of_empty_explanation_is_a_noop(monkeypatch):
    called = []
    monkeypatch.setattr(narrative, "call_llm_json", lambda prompt: called.append(1))
    assert polish_explanation("", {}) == ""
    assert called == []


# ---- build_explanation / explain_subject / explain_comparison (Part 11) ----

def test_leaderboard_explanation_includes_ranking_justification():
    explanation = build_explanation(_ir(), ROWS)
    assert "Waqar Haider" in explanation
    assert "900" in explanation
    assert "1st of 3" in explanation


def test_leaderboard_explanation_uses_true_total_when_given():
    explanation = build_explanation(_ir(), ROWS, total_count=47)
    assert "1st of 47" in explanation


def test_percentage_metric_explains_gap_to_goal():
    row = {"wid": 1, "name": "Ali", "value": 75.0}
    explanation = explain_subject(row, "achievement_pct", "advisor", _ir(), rank=3, total=8)
    assert "achieved 75% of the assigned target" in explanation
    assert "ranking 3rd of 8 advisors shown" in explanation
    assert "remaining 25% short of the monthly goal" in explanation


def test_percentage_metric_above_target_explains_surplus():
    row = {"wid": 1, "name": "Ali", "value": 120.0}
    explanation = explain_subject(row, "achievement_pct", "advisor", _ir())
    assert "exceeding the monthly goal by 20%" in explanation


def test_single_result_has_no_ranking_clause():
    row = {"wid": 1, "name": "Ali", "value": 75.0}
    explanation = explain_subject(row, "achievement_pct", "advisor", _ir(), rank=1, total=1)
    assert "ranking" not in explanation


def test_filtered_list_explanation_has_no_ranking_language():
    ir = _ir(intent="filtered_list")
    explanation = build_explanation(ir, ROWS, total_count=10)
    assert "ranking" not in explanation
    assert "Waqar Haider" in explanation


def test_filters_are_related_to_the_ranking_metric():
    ir = _ir(filters=[Filter(field="attendance_rate", operator=">", value=90)])
    explanation = build_explanation(ir, ROWS, total_count=3)
    assert "filtered by" in explanation
    assert "Attendance Rate" in explanation


def test_comparison_explanation_covers_every_subject_and_the_lead():
    explanation = explain_comparison(ROWS[:2], "mtd_cleared", "advisor", _ir(intent="comparison"))
    assert "Waqar Haider" in explanation
    assert "Ali Raza" in explanation
    assert "leads Ali Raza by 400" in explanation
    assert "80%" in explanation


def test_empty_rows_produce_no_explanation():
    assert build_explanation(_ir(), []) == ""


# ---- compute_trends (Part 11) ----

def test_trend_is_empty_without_history(db_session):
    ir = _ir()
    assert compute_trends(ir, ROWS, db_session) == []


def test_trend_reports_delta_against_latest_snapshot(db_session):
    from app.database.models import AdvisorHistory
    import datetime

    db_session.add(AdvisorHistory(wid=1, snapshot_at=datetime.datetime(2026, 1, 1), mtd_cleared=800.0))
    db_session.commit()

    trends = compute_trends(_ir(), ROWS, db_session)
    assert len(trends) == 1
    assert "Waqar Haider" in trends[0]
    assert "up 100" in trends[0]
    assert "800" in trends[0] and "900" in trends[0]


def test_trend_skips_unsupported_metric(db_session):
    ir = _ir(metric=MetricRef(key="conversion"), sort=Sort(metric="conversion", direction="desc"))
    assert compute_trends(ir, ROWS, db_session) == []


def test_trend_skips_team_level(db_session):
    from app.database.models import AdvisorHistory
    import datetime

    db_session.add(AdvisorHistory(wid=1, snapshot_at=datetime.datetime(2026, 1, 1), mtd_cleared=800.0))
    db_session.commit()
    ir = _ir(subject_level="team")
    assert compute_trends(ir, ROWS, db_session) == []


def test_trend_skips_unchanged_values(db_session):
    from app.database.models import AdvisorHistory
    import datetime

    db_session.add(AdvisorHistory(wid=1, snapshot_at=datetime.datetime(2026, 1, 1), mtd_cleared=900.0))
    db_session.commit()
    assert compute_trends(_ir(), ROWS, db_session) == []


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
