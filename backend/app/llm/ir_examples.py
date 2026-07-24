"""
Few-shot examples for the LLM semantic parser (P2 of the NLU rework).
Kept in their own module — not inline strings in prompt_builder — so
test_ir_examples.py can validate every example against the real QueryIR
model and ir_validator, meaning an example can never silently drift from
the schema the model is being asked to produce.

Entity names used here (Blue Area, Downtown, Graana) are fictional
few-shot scaffolding — the prompt separately grounds the model in the
REAL team/company gazetteer, and ir_validator re-grounds every subject
regardless of what the model saw here.

Each example: utterance, optional prior_ir (for the follow-up-patch
example), the expected IR dict (exactly the keys of the strict output
schema in llm_client.QUERY_IR_JSON_SCHEMA), and expect_valid=False for
the deliberately-ambiguous example whose low-confidence filter is
SUPPOSED to trip the validator into a clarification.
"""

EXAMPLES: list[dict] = [
    {
        # the canonical multi-filter + explicit-sort query
        "utterance": "Show Graana advisors with attendance above 90% and achievement above 80%, sorted by meetings",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {"key": "total_meetings", "confidence": 0.95},
            "filters": [
                {"field": "company", "operator": "=", "value": "Graana", "confidence": 0.95},
                {"field": "attendance_rate", "operator": ">", "value": 90, "confidence": 0.9},
                {"field": "achievement_pct", "operator": ">", "value": 80, "confidence": 0.9},
            ],
            # no period mentioned — MTD is a default guess, not a stated fact
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "total_meetings", "direction": "desc"},
            "limit": None,
            "group_by": None,
            "overall_confidence": 0.9,
            "intent_confidence": 0.95,
        },
    },
    {
        "utterance": "compare Blue Area with Downtown on revenue",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "comparison",
            "subject_level": "team",
            "subjects": [
                {"type": "team", "value": "Blue Area", "match_confidence": 1.0},
                {"type": "team", "value": "Downtown", "match_confidence": 1.0},
            ],
            "metric": {"key": "mtd_cleared", "confidence": 0.95},
            "filters": [],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "mtd_cleared", "direction": "desc"},
            "limit": None,
            "group_by": None,
            "overall_confidence": 0.9,
            "intent_confidence": 0.95,
        },
    },
    {
        "utterance": "who is the best performer",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {"key": "achievement_pct", "confidence": 0.85},
            "filters": [],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "achievement_pct", "direction": "desc"},
            "limit": 1,
            "group_by": None,
            "overall_confidence": 0.85,
            "intent_confidence": 0.9,
        },
    },
    {
        # "underperforming" = bottom of the achievement ranking, ascending
        "utterance": "show me the underperforming advisors",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {"key": "achievement_pct", "confidence": 0.8},
            "filters": [],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "achievement_pct", "direction": "asc"},
            "limit": 10,
            "group_by": None,
            "overall_confidence": 0.8,
            "intent_confidence": 0.9,
        },
    },
    {
        # "almost achieved" = a band, expressed as two AND-combined filters
        "utterance": "advisors who almost achieved their target",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {"key": "achievement_pct", "confidence": 0.85},
            "filters": [
                {"field": "achievement_pct", "operator": ">=", "value": 80, "confidence": 0.75},
                {"field": "achievement_pct", "operator": "<", "value": 100, "confidence": 0.75},
            ],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "achievement_pct", "direction": "desc"},
            "limit": None,
            "group_by": None,
            "overall_confidence": 0.8,
            "intent_confidence": 0.85,
        },
    },
    {
        # typo'd team name — emit the CORRECTED gazetteer value with a
        # slightly reduced confidence, don't echo the typo
        "utterance": "top revenue in blue aera",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {"key": "mtd_cleared", "confidence": 0.95},
            "filters": [
                {"field": "team", "operator": "=", "value": "Blue Area", "confidence": 0.85},
            ],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "mtd_cleared", "direction": "desc"},
            "limit": 10,
            "group_by": None,
            "overall_confidence": 0.85,
            "intent_confidence": 0.95,
        },
    },
    {
        "utterance": "top 5 advisors by ytd revenue",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {"key": "ytd_cleared", "confidence": 0.95},
            "filters": [],
            # "ytd" is stated explicitly — high time confidence, unlike the
            # defaulted-MTD examples above
            "time_range": {"mode": "snapshot", "period": "YTD", "compare_to": None, "confidence": 0.95},
            "sort": {"metric": "ytd_cleared", "direction": "desc"},
            "limit": 5,
            "group_by": None,
            "overall_confidence": 0.95,
            "intent_confidence": 0.95,
        },
    },
    {
        "utterance": "which teams have the best attendance rate",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "subject_level": "team",
            "subjects": [],
            "metric": {"key": "attendance_rate", "confidence": 0.95},
            "filters": [],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "attendance_rate", "direction": "desc"},
            "limit": 10,
            "group_by": None,
            "overall_confidence": 0.95,
            "intent_confidence": 0.95,
        },
    },
    {
        # follow-up patch: keep everything from the prior IR, add the filter
        "utterance": "only Graana",
        "prior_ir": {
            "intent": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {"key": "mtd_cleared", "confidence": 0.95},
            "filters": [],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "mtd_cleared", "direction": "desc"},
            "limit": 10,
            "group_by": None,
            "overall_confidence": 0.95,
            "intent_confidence": 0.95,
        },
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {"key": "mtd_cleared", "confidence": 0.95},
            "filters": [
                {"field": "company", "operator": "=", "value": "Graana", "confidence": 0.95},
            ],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "mtd_cleared", "direction": "desc"},
            "limit": 10,
            "group_by": None,
            "overall_confidence": 0.9,
            "intent_confidence": 0.95,
        },
    },
    {
        # ambiguous business language — clarify with a low-confidence guess,
        # never an invented metric key. The low confidence intentionally
        # trips the validator into asking a targeted question.
        "utterance": "who is struggling this month",
        "prior_ir": None,
        "expect_valid": False,
        "ir": {
            "intent": "clarify",
            "subject_level": "advisor",
            "subjects": [],
            "metric": None,
            "filters": [
                {"field": "achievement_pct", "operator": "<", "value": 50, "confidence": 0.4},
            ],
            # "this month" IS explicit — the ambiguity here is entirely
            # about the metric/intent, not the period, which is why
            # time_range.confidence stays high even though overall_confidence
            # (and intent_confidence, tracking it) are both low
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.9},
            "sort": {"metric": None, "direction": "desc"},
            "limit": 10,
            "group_by": None,
            "overall_confidence": 0.4,
            "intent_confidence": 0.4,
        },
    },
]


def render_examples() -> str:
    """The examples block injected into the parser prompt."""
    import json

    lines = ["Examples (follow these exactly in style and strictness):"]
    for i, ex in enumerate(EXAMPLES, 1):
        lines.append(f'Example {i} — User: "{ex["utterance"]}"')
        if ex["prior_ir"]:
            lines.append(f"  (Previous turn's query was: {json.dumps(ex['prior_ir'])})")
        lines.append(f"  -> {json.dumps(ex['ir'])}")
    return "\n".join(lines)
