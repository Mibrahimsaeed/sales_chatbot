"""Model-level tests for the confidence-aware QueryIR fields (Part 10) —
intent_confidence, time_range.confidence, ambiguity_reasons, and
confidence_level are additive; an IR built the OLD way (no keyword args
for any of them) must behave exactly as it did before these fields
existed. Validator-driven population of ambiguity_reasons/confidence_level
is covered in test_ir_validator.py."""

from app.llm.query_ir import MetricRef, QueryIR, Sort, TimeRange


def test_new_fields_default_to_backward_compatible_values():
    ir = QueryIR(
        intent="leaderboard",
        metric=MetricRef(key="mtd_cleared"),
        sort=Sort(metric="mtd_cleared", direction="desc"),
    )
    assert ir.intent_confidence == 1.0
    assert ir.time_range.confidence == 1.0
    assert ir.ambiguity_reasons == []
    assert ir.confidence_level is None


def test_time_range_confidence_is_independently_settable():
    tr = TimeRange(period="YTD", confidence=0.5)
    assert tr.confidence == 0.5
    assert tr.period == "YTD"


def test_confidence_level_accepts_all_three_tiers():
    for level in ("high", "medium", "low"):
        ir = QueryIR(intent="leaderboard", confidence_level=level)
        assert ir.confidence_level == level


def test_breakdown_is_a_valid_intent():
    ir = QueryIR(intent="breakdown", subject_level="unit_head")
    assert ir.intent == "breakdown"


def test_flat_defaults_false_for_backward_compatibility():
    ir = QueryIR(intent="leaderboard", metric=MetricRef(key="mtd_cleared"), sort=Sort(metric="mtd_cleared"))
    assert ir.flat is False


def test_flat_is_independently_settable():
    ir = QueryIR(intent="breakdown", subject_level="unit_head", flat=True)
    assert ir.flat is True
