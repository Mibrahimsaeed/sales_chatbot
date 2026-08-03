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
        # new hierarchy level (Part: hierarchy rework) — subject_level is
        # "unit_head", NOT "team"; "unit head" must never be inferred from
        # a bare team mention, only from the literal phrase.
        "utterance": "top 5 unit heads by connects",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "subject_level": "unit_head",
            "subjects": [],
            "metric": {"key": "total_connects", "confidence": 0.95},
            "filters": [],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "total_connects", "direction": "desc"},
            "limit": 5,
            "group_by": None,
            "overall_confidence": 0.9,
            "intent_confidence": 0.9,
        },
    },
    {
        # new hierarchy level comparison — same "comparison" intent shape
        # as the team example above, just at subject_level "zonal_head".
        "utterance": "compare zonal head Ahmed Ali with zonal head Bilal Khan on revenue",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "comparison",
            "subject_level": "zonal_head",
            "subjects": [
                {"type": "zonal_head", "value": "Ahmed Ali", "match_confidence": 1.0},
                {"type": "zonal_head", "value": "Bilal Khan", "match_confidence": 1.0},
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
        # "breakdown" intent (Part: hierarchy rework phase 2) — a question
        # about ONE named entity, nested by team. The utterance deliberately
        # includes "performance" (a metric synonym) — this is exactly the
        # phrasing that used to get mis-parsed as an unfiltered leaderboard
        # (or with the subject dropped) before "breakdown" existed as its
        # own intent; it is NOT a ranking, so no metric/sort is needed.
        "utterance": "give me a breakdown of unit head Zeeshan Tariq's performance",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "breakdown",
            "subject_level": "unit_head",
            "subjects": [
                {"type": "unit_head", "value": "Zeeshan Tariq", "match_confidence": 1.0},
            ],
            "metric": None,
            "filters": [],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": None, "direction": "desc"},
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.9,
            "intent_confidence": 0.9,
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
    # ------------------------------------------------------------------
    # Phase 5.2: the levels no example demonstrated. The schema accepted
    # bcm/office/region, the prompt described them, and every worked
    # example still showed only advisor/team/unit_head/zonal_head — so
    # the shapes the model actually imitates never included them.
    # ------------------------------------------------------------------
    {
        # BCM — a level of the verified chain, previously undemonstrated.
        "utterance": "top 5 bcms by connects",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "subject_level": "bcm",
            "subjects": [],
            "metric": {"key": "total_connects", "confidence": 0.95},
            "filters": [],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "total_connects", "direction": "desc"},
            "limit": 5,
            "group_by": None,
            "overall_confidence": 0.94,
            "intent_confidence": 0.95,
        },
    },
    {
        # office — an ATTRIBUTE level. Rankable as a subject_level even
        # though it does not nest in the chain.
        "utterance": "top business centers by revenue",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "subject_level": "office",
            "subjects": [],
            "metric": {"key": "mtd_cleared", "confidence": 0.95},
            "filters": [],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "mtd_cleared", "direction": "desc"},
            "limit": 10,
            "group_by": None,
            "overall_confidence": 0.93,
            "intent_confidence": 0.94,
        },
    },
    {
        # region as a FILTER rather than a subject — the shape "advisors
        # in North Region" takes.
        "utterance": "top advisors in North/KPK region by revenue",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {"key": "mtd_cleared", "confidence": 0.95},
            "filters": [
                {"field": "region", "operator": "=", "value": "North/KPK", "confidence": 0.9},
            ],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "mtd_cleared", "direction": "desc"},
            "limit": 10,
            "group_by": None,
            "overall_confidence": 0.92,
            "intent_confidence": 0.93,
        },
    },
    {
        # company as a subject_level — the schema forbade this entirely
        # before Phase 5.2's predecessor widened HIERARCHY_LEVELS.
        "utterance": "compare Graana and IMARAT by revenue",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "comparison",
            "subject_level": "company",
            "subjects": [
                {"type": "company", "value": "Graana", "match_confidence": 0.98},
                {"type": "company", "value": "IMARAT", "match_confidence": 0.98},
            ],
            "metric": {"key": "mtd_cleared", "confidence": 0.95},
            "filters": [],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None, "confidence": 0.6},
            "sort": {"metric": "mtd_cleared", "direction": "desc"},
            "limit": 10,
            "group_by": None,
            "overall_confidence": 0.94,
            "intent_confidence": 0.96,
        },
    },
    {
        # A DAILY question (Phase 5.1 vocabulary): name the period the
        # user used, do not substitute MTD.
        "utterance": "top advisors by connects today",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {"key": "total_connects", "confidence": 0.95},
            "filters": [],
            "time_range": {"mode": "snapshot", "period": "DAILY", "compare_to": None, "confidence": 0.95},
            "sort": {"metric": "total_connects", "direction": "desc"},
            "limit": 10,
            "group_by": None,
            "overall_confidence": 0.93,
            "intent_confidence": 0.95,
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
