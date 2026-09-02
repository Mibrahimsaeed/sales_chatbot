"""
Few-shot examples for the LLM semantic parser (P2 of the NLU rework).

Kept in their own module — not inline strings in prompt_builder — so
test_ir_examples.py can validate every example against the real QueryIR
model and ir_validator, meaning an example can never silently drift from
the schema the model is being asked to produce.

Entity names used here (Blue Area, Downtown, Graana, etc.) are fictional
few-shot scaffolding. The prompt separately grounds the model in the REAL
team/company/person gazetteers, and ir_validator re-grounds every subject
regardless of what the model saw here.

The examples deliberately cover the compositional QueryIR shapes that the
LLM must learn to produce. They are semantic templates, NOT query-specific
answer mappings.

IMPORTANT:
- Do not add production/test queries merely because they fail.
- Examples must teach reusable semantic structures.
- Never encode query -> answer mappings here.
"""

EXAMPLES: list[dict] = [
    {
        # ---------------------------------------------------------------
        # MULTI-FILTER + EXPLICIT SORT
        # ---------------------------------------------------------------
        "utterance": (
            "Show Graana advisors with attendance above 90% and "
            "achievement above 80%, sorted by meetings"
        ),
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {
                "key": "total_meetings",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "attendance_rate",
                    "confidence": 0.9,
                },
                {
                    "key": "achievement_pct",
                    "confidence": 0.9,
                },
                {
                    "key": "total_meetings",
                    "confidence": 0.95,
                },
            ],
            "filters": [
                {
                    "field": "company",
                    "operator": "=",
                    "value": "Graana",
                    "confidence": 0.95,
                },
                {
                    "field": "attendance_rate",
                    "operator": ">",
                    "value": 90,
                    "confidence": 0.9,
                },
                {
                    "field": "achievement_pct",
                    "operator": ">",
                    "value": 80,
                    "confidence": 0.9,
                },
            ],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "total_meetings",
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.9,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # COMPARISON
        # ---------------------------------------------------------------
        "utterance": "compare Blue Area with Downtown on revenue",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "comparison",
            "operation": "comparison",
            "subject_level": "team",
            "subjects": [
                {
                    "type": "team",
                    "value": "Blue Area",
                    "match_confidence": 1.0,
                },
                {
                    "type": "team",
                    "value": "Downtown",
                    "match_confidence": 1.0,
                },
            ],
            "metric": {
                "key": "mtd_cleared",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "mtd_cleared",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "mtd_cleared",
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.9,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # BEST PERFORMER
        # ---------------------------------------------------------------
        "utterance": "who is the best performer",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {
                "key": "achievement_pct",
                "confidence": 0.85,
            },
            "metrics": [
                {
                    "key": "achievement_pct",
                    "confidence": 0.85,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "achievement_pct",
                "direction": "desc",
            },
            "limit": 1,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.85,
            "intent_confidence": 0.9,
        },
    },

    {
        # ---------------------------------------------------------------
        # UNDERPERFORMING
        # ---------------------------------------------------------------
        "utterance": "show me the underperforming advisors",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {
                "key": "achievement_pct",
                "confidence": 0.8,
            },
            "metrics": [
                {
                    "key": "achievement_pct",
                    "confidence": 0.8,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "achievement_pct",
                "direction": "asc",
            },
            "limit": 10,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.8,
            "intent_confidence": 0.9,
        },
    },

    {
        # ---------------------------------------------------------------
        # ALMOST ACHIEVED TARGET
        # ---------------------------------------------------------------
        "utterance": "advisors who almost achieved their target",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "filtered_list",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {
                "key": "achievement_pct",
                "confidence": 0.85,
            },
            "metrics": [
                {
                    "key": "achievement_pct",
                    "confidence": 0.85,
                },
            ],
            "filters": [
                {
                    "field": "achievement_pct",
                    "operator": ">=",
                    "value": 80,
                    "confidence": 0.75,
                },
                {
                    "field": "achievement_pct",
                    "operator": "<",
                    "value": 100,
                    "confidence": 0.75,
                },
            ],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "achievement_pct",
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.8,
            "intent_confidence": 0.85,
        },
    },

    {
        # ---------------------------------------------------------------
        # TYPO / GAZETTEER CORRECTION
        # ---------------------------------------------------------------
        "utterance": "top revenue in blue aera",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {
                "key": "mtd_cleared",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "mtd_cleared",
                    "confidence": 0.95,
                },
            ],
            "filters": [
                {
                    "field": "team",
                    "operator": "=",
                    "value": "Blue Area",
                    "confidence": 0.85,
                },
            ],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "mtd_cleared",
                "direction": "desc",
            },
            "limit": 10,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.85,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # YTD
        # ---------------------------------------------------------------
        "utterance": "top 5 advisors by ytd revenue",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {
                "key": "ytd_cleared",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "ytd_cleared",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "YTD",
                "compare_to": None,
                "confidence": 0.95,
            },
            "sort": {
                "metric": "ytd_cleared",
                "direction": "desc",
            },
            "limit": 5,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.95,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # TEAM RANKING
        # ---------------------------------------------------------------
        "utterance": "which teams have the best attendance rate",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "team",
            "subjects": [],
            "metric": {
                "key": "attendance_rate",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "attendance_rate",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "attendance_rate",
                "direction": "desc",
            },
            "limit": 10,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.95,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # FOLLOW-UP PATCH
        # ---------------------------------------------------------------
        "utterance": "only Graana",
        "prior_ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {
                "key": "mtd_cleared",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "mtd_cleared",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "mtd_cleared",
                "direction": "desc",
            },
            "limit": 10,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.95,
            "intent_confidence": 0.95,
        },
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {
                "key": "mtd_cleared",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "mtd_cleared",
                    "confidence": 0.95,
                },
            ],
            "filters": [
                {
                    "field": "company",
                    "operator": "=",
                    "value": "Graana",
                    "confidence": 0.95,
                },
            ],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "mtd_cleared",
                "direction": "desc",
            },
            "limit": 10,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.9,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # UNIT HEAD RANKING
        # ---------------------------------------------------------------
        "utterance": "top 5 unit heads by connects",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "unit_head",
            "subjects": [],
            "metric": {
                "key": "total_connects",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "total_connects",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "total_connects",
                "direction": "desc",
            },
            "limit": 5,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.9,
            "intent_confidence": 0.9,
        },
    },

    {
        # ---------------------------------------------------------------
        # ZONAL HEAD COMPARISON
        # ---------------------------------------------------------------
        "utterance": (
            "compare zonal head Ahmed Ali with zonal head Bilal Khan "
            "on revenue"
        ),
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "comparison",
            "operation": "comparison",
            "subject_level": "zonal_head",
            "subjects": [
                {
                    "type": "zonal_head",
                    "value": "Ahmed Ali",
                    "match_confidence": 1.0,
                },
                {
                    "type": "zonal_head",
                    "value": "Bilal Khan",
                    "match_confidence": 1.0,
                },
            ],
            "metric": {
                "key": "mtd_cleared",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "mtd_cleared",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "mtd_cleared",
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.9,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # GROUP_METRIC — ONE NAMED ENTITY'S OWN FIGURE
        #
        # The single most ordinary analytical question there is, and the
        # corpus had NO example of it. `group_metric` is one of only three
        # plan actions that reach the model at all, so the shape it sees
        # most often was the one shape it was never shown.
        #
        # This slot held a `breakdown` example. `breakdown` is declared
        # expressible_in_ir=False and is served entirely by the rule
        # planner, so it is absent from the operation enum and the model
        # could not emit it however well it imitated the example — and
        # nlu_pipeline never routes a breakdown-shaped query here anyway.
        #
        # WHAT IT TEACHES: the answer is about the entity that was NAMED,
        # so subject_level equals the subject's own type, and there is
        # nothing to rank — one row, no sort metric.
        # ---------------------------------------------------------------
        "utterance": "what is unit head Zeeshan Tariq's revenue",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "group_metric",
            "subject_level": "unit_head",
            "subjects": [
                {
                    "type": "unit_head",
                    "value": "Zeeshan Tariq",
                    "match_confidence": 1.0,
                },
            ],
            "metric": {
                "key": "mtd_cleared",
                "confidence": 0.95,
            },
            "metrics": [],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.95,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # GROUP_METRIC — A TEAM, NOT A RANKING OF ITS MEMBERS
        #
        # The distinction group_metric exists for. "Blue Area's connects"
        # is ONE number for the team; it is not a leaderboard of the
        # people in it, and it is not a filtered_list of anything.
        # ---------------------------------------------------------------
        "utterance": "how many connects does Blue Area have",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "group_metric",
            "subject_level": "team",
            "subjects": [
                {
                    "type": "team",
                    "value": "Blue Area",
                    "match_confidence": 1.0,
                },
            ],
            "metric": {
                "key": "total_connects",
                "confidence": 0.95,
            },
            "metrics": [],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.95,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # GROUP_METRIC — AN ATTRIBUTE-LEVEL ENTITY
        #
        # `company` is an ATTRIBUTE, not a step in the chain, and this is
        # the shape that proves the two facts are compatible: an attribute
        # cannot be traversed through, but it is a perfectly good thing to
        # ask about. Answer it at the level the user named.
        # ---------------------------------------------------------------
        "utterance": "what is Graana's revenue this month",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "group_metric",
            "subject_level": "company",
            "subjects": [
                {
                    "type": "company",
                    "value": "Graana",
                    "match_confidence": 1.0,
                },
            ],
            "metric": {
                "key": "mtd_cleared",
                "confidence": 0.95,
            },
            "metrics": [],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.9,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.95,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # AMBIGUOUS BUSINESS LANGUAGE
        # ---------------------------------------------------------------
        "utterance": "who is struggling this month",
        "prior_ir": None,
        "expect_valid": False,
        "ir": {
            "intent": "clarify",
            # THE OPERATION IS "clarify_metric". `clarify` alone is the
            # legacy INTENT name and is not an operation the registry
            # knows, so this example asked the model to emit a value the
            # grammar rejects — on the one example whose whole purpose is
            # to show how to decline.
            "operation": "clarify_metric",
            "subject_level": "advisor",
            "subjects": [],
            "metric": None,
            "metrics": [],
            "filters": [
                {
                    "field": "achievement_pct",
                    "operator": "<",
                    "value": 50,
                    "confidence": 0.4,
                },
            ],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.9,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": 10,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.4,
            "intent_confidence": 0.4,
        },
    },

    # =================================================================
    # LEVEL COVERAGE
    # =================================================================

    {
        # ---------------------------------------------------------------
        # BCM
        # ---------------------------------------------------------------
        "utterance": "top 5 bcms by connects",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "bcm",
            "subjects": [],
            "metric": {
                "key": "total_connects",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "total_connects",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "total_connects",
                "direction": "desc",
            },
            "limit": 5,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.94,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # OFFICE
        # ---------------------------------------------------------------
        "utterance": "top business centers by revenue",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "office",
            "subjects": [],
            "metric": {
                "key": "mtd_cleared",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "mtd_cleared",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "mtd_cleared",
                "direction": "desc",
            },
            "limit": 10,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.93,
            "intent_confidence": 0.94,
        },
    },

    {
        # ---------------------------------------------------------------
        # REGION ATTRIBUTE AS FILTER
        # ---------------------------------------------------------------
        "utterance": "top advisors in North/KPK region by revenue",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {
                "key": "mtd_cleared",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "mtd_cleared",
                    "confidence": 0.95,
                },
            ],
            "filters": [
                {
                    "field": "region",
                    "operator": "=",
                    "value": "North/KPK",
                    "confidence": 0.9,
                },
            ],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "mtd_cleared",
                "direction": "desc",
            },
            "limit": 10,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.92,
            "intent_confidence": 0.93,
        },
    },

    {
        # ---------------------------------------------------------------
        # COMPANY AS SUBJECT
        # ---------------------------------------------------------------
        "utterance": "compare Graana and IMARAT by revenue",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "comparison",
            "operation": "comparison",
            "subject_level": "company",
            "subjects": [
                {
                    "type": "company",
                    "value": "Graana",
                    "match_confidence": 0.98,
                },
                {
                    "type": "company",
                    "value": "IMARAT",
                    "match_confidence": 0.98,
                },
            ],
            "metric": {
                "key": "mtd_cleared",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "mtd_cleared",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "mtd_cleared",
                "direction": "desc",
            },
            "limit": 10,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.94,
            "intent_confidence": 0.96,
        },
    },

    {
        # ---------------------------------------------------------------
        # DAILY
        # ---------------------------------------------------------------
        "utterance": "top advisors by connects today",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {
                "key": "total_connects",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "total_connects",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "DAILY",
                "compare_to": None,
                "confidence": 0.95,
            },
            "sort": {
                "metric": "total_connects",
                "direction": "desc",
            },
            "limit": 10,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.93,
            "intent_confidence": 0.95,
        },
    },

    # =================================================================
    # COMPOSITIONAL SHAPES
    # =================================================================

    {
        # ---------------------------------------------------------------
        # OR FILTER TREE
        # ---------------------------------------------------------------
        "utterance": "Show advisors in Blue Area or DownTown",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "population",
            "subject_level": "advisor",
            "subjects": [],
            "metric": None,
            "metrics": [],
            "filters": [],
            "filter_tree": {
                "op": "or",
                "children": [
                    {
                        "field": "team",
                        "operator": "=",
                        "value": "Blue Area",
                        "confidence": 0.95,
                    },
                    {
                        "field": "team",
                        "operator": "=",
                        "value": "Downtown",
                        "confidence": 0.95,
                    },
                ],
            },
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.9,
            "intent_confidence": 0.9,
        },
    },

    {
        # ---------------------------------------------------------------
        # NOT FILTER TREE
        # ---------------------------------------------------------------
        "utterance": "List all advisors excluding Blue Area",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "population",
            "subject_level": "advisor",
            "subjects": [],
            "metric": None,
            "metrics": [],
            "filters": [],
            "filter_tree": {
                "op": "not",
                "children": [
                    {
                        "field": "team",
                        "operator": "=",
                        "value": "Blue Area",
                        "confidence": 0.95,
                    },
                ],
            },
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.9,
            "intent_confidence": 0.9,
        },
    },

    {
        # ---------------------------------------------------------------
        # MULTIPLE METRICS
        # ---------------------------------------------------------------
        "utterance": "Show connects and answered calls for all BCMs",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "bcm",
            "subjects": [],
            "metric": {
                "key": "total_connects",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "total_connects",
                    "confidence": 0.95,
                },
                {
                    "key": "answered_calls",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "total_connects",
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.92,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # METRIC PER SUBJECT
        # ---------------------------------------------------------------
        "utterance": "Blue Area's connects and Downtown's revenue",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "comparison",
            "operation": "comparison",
            "subject_level": "team",
            "subjects": [
                {
                    "type": "team",
                    "value": "Blue Area",
                    "match_confidence": 1.0,
                    "metric": {
                        "key": "total_connects",
                        "confidence": 0.95,
                    },
                },
                {
                    "type": "team",
                    "value": "Downtown",
                    "match_confidence": 1.0,
                    "metric": {
                        "key": "mtd_cleared",
                        "confidence": 0.95,
                    },
                },
            ],
            "metric": {
                "key": "total_connects",
                "confidence": 0.8,
            },
            "metrics": [
                {
                    "key": "total_connects",
                    "confidence": 0.95,
                },
                {
                    "key": "mtd_cleared",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "total_connects",
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.88,
            "intent_confidence": 0.9,
        },
    },

    {
        # ---------------------------------------------------------------
        # TWO DIFFERENT METRIC CONDITIONS
        # ---------------------------------------------------------------
        "utterance": (
            "Show BCMs with connects above 1200 and "
            "achievement below 50%"
        ),
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "filtered_list",
            "subject_level": "bcm",
            "subjects": [],
            "metric": None,
            "metrics": [
                {
                    "key": "total_connects",
                    "confidence": 0.95,
                },
                {
                    "key": "achievement_pct",
                    "confidence": 0.95,
                },
            ],
            "filters": [
                {
                    "field": "total_connects",
                    "operator": ">",
                    "value": 1200,
                    "confidence": 0.95,
                },
                {
                    "field": "achievement_pct",
                    "operator": "<",
                    "value": 50,
                    "confidence": 0.95,
                },
            ],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.93,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # TEAM SIZE CONDITION
        # ---------------------------------------------------------------
        "utterance": "Which BCMs have more than five advisors in their teams?",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "filtered_list",
            "subject_level": "bcm",
            "subjects": [],
            "metric": None,
            "metrics": [
                {
                    "key": "team_size",
                    "confidence": 0.9,
                },
            ],
            "filters": [
                {
                    "field": "team_size",
                    "operator": ">",
                    "value": 5,
                    "confidence": 0.9,
                },
            ],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.9,
            "intent_confidence": 0.9,
        },
    },

    {
        # ---------------------------------------------------------------
        # PURE METRIC FILTER
        # ---------------------------------------------------------------
        "utterance": "advisors with connects above 1000",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "filtered_list",
            "subject_level": "advisor",
            "subjects": [],
            "metric": None,
            "metrics": [
                {
                    "key": "total_connects",
                    "confidence": 0.95,
                },
            ],
            "filters": [
                {
                    "field": "total_connects",
                    "operator": ">",
                    "value": 1000,
                    "confidence": 0.95,
                },
            ],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.93,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # TWO CONDITIONS WITHOUT RANKING
        # ---------------------------------------------------------------
        "utterance": (
            "advisors with achievement below 50% and "
            "answered calls % below 50%"
        ),
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "filtered_list",
            "subject_level": "advisor",
            "subjects": [],
            "metric": None,
            "metrics": [
                {
                    "key": "achievement_pct",
                    "confidence": 0.95,
                },
                {
                    "key": "answered_calls_rate",
                    "confidence": 0.95,
                },
            ],
            "filters": [
                {
                    "field": "achievement_pct",
                    "operator": "<",
                    "value": 50,
                    "confidence": 0.95,
                },
                {
                    "field": "answered_calls_rate",
                    "operator": "<",
                    "value": 50,
                    "confidence": 0.95,
                },
            ],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.93,
            "intent_confidence": 0.95,
        },
    },

    {
        # ---------------------------------------------------------------
        # PARAPHRASE
        # ---------------------------------------------------------------
        "utterance": (
            "Which business centers have the highest number of connects?"
        ),
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "office",
            "subjects": [],
            "metric": {
                "key": "total_connects",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "total_connects",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "total_connects",
                "direction": "desc",
            },
            "limit": 10,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.92,
            "intent_confidence": 0.95,
        },
    },

    # =================================================================
    # HIERARCHY READS
    #
    # THESE ARE THE IMPORTANT NEW EXAMPLES.
    #
    # The previous examples taught leaderboard/filter/comparison shapes
    # but did not demonstrate:
    #
    #   subject_of
    #   target_level
    #   relation
    #
    # Those fields encode the meaning of:
    #
    #   "reports to"
    #   "directly reports to"
    #   "under"
    #   "beneath"
    #   "within their organisation"
    #
    # These examples deliberately use fictional names and different
    # hierarchy levels so the model learns the semantic structure rather
    # than memorizing one failing query.
    # =================================================================

    {
        # ---------------------------------------------------------------
        # HIERARCHY READ — DIRECT REPORTS
        #
        # Explicitly says "directly", therefore relation = direct.
        # The question asks for advisors, so target_level = advisor.
        # The named manager is the scope/subject_of.
        # ---------------------------------------------------------------
        "utterance": "How many advisors directly report to Unit Head Ahmed?",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "population",
            "subject_level": "advisor",
            "subjects": [
                {
                    "type": "unit_head",
                    "value": "Ahmed",
                    "match_confidence": 1.0,
                },
            ],
            "metric": None,
            "metrics": [],
            "filters": [],
            "filter_tree": None,
            "target_level": "advisor",
            "subject_of": "unit_head",
            "relation": "direct",
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.96,
            "intent_confidence": 0.98,
        },
    },

    {
        # ---------------------------------------------------------------
        # HIERARCHY READ — SUBTREE
        #
        # "under" does NOT mean direct. It means the complete subtree.
        # ---------------------------------------------------------------
        "utterance": "How many advisors are under Unit Head Ahmed?",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "population",
            "subject_level": "advisor",
            "subjects": [
                {
                    "type": "unit_head",
                    "value": "Ahmed",
                    "match_confidence": 1.0,
                },
            ],
            "metric": None,
            "metrics": [],
            "filters": [],
            "filter_tree": None,
            "target_level": "advisor",
            "subject_of": "unit_head",
            "relation": "subtree",
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.96,
            "intent_confidence": 0.98,
        },
    },

    {
        # ---------------------------------------------------------------
        # A MANAGEMENT LEVEL WITHIN A NAMED GROUP
        #
        # THE SHAPE WITH NO COVERAGE. Every other hierarchy example reads
        # a level beneath a PERSON ("which BCMs are under Zonal Head
        # Bilal") or advisors within a TEAM. Nobody showed a MANAGEMENT
        # level within a TEAM, and measured against the real model five
        # equivalent phrasings of it produced four different
        # interpretations — two of them asking which metric was meant,
        # for a question that names no measure at all.
        #
        # Deliberately a different level and a different team from the
        # phrasings this was written for: what has to be learned is the
        # SHAPE — group is the scope, level is the target, no measure
        # required — not any particular sentence.
        # ---------------------------------------------------------------
        "utterance": "zonal heads in Downtown",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "population",
            "subject_level": "team",
            "subjects": [
                {"type": "team", "value": "Downtown", "match_confidence": 1.0},
            ],
            # NO MEASURE, and none is needed: the question asks WHO holds
            # a level, not how they performed. Asking which metric was
            # meant is a wrong answer to a complete question.
            "metric": None,
            "metrics": [],
            "filters": [],
            "filter_tree": None,
            "target_level": "zonal_head",
            "subject_of": "team",
            "relation": "subtree",
            "time_range": {"mode": "snapshot", "period": "MTD",
                           "compare_to": None, "confidence": 0.6},
            "sort": {"metric": None, "direction": "desc"},
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.95,
            "intent_confidence": 0.96,
        },
    },

    {
        # ---------------------------------------------------------------
        # A ROLE INSIDE A GROUP, AND A DIRECT RELATIONSHIP
        #
        # "the Unit Head in AMD" names no person — it names a ROLE and the
        # group to find it in. So the TEAM is the subject and the role is
        # `subject_of`; putting the role in `subjects` invents an entity
        # nobody named, and the scope is then lost entirely.
        #
        # Measured before this example existed: the model set
        # subject_of="unit_head" correctly and dropped the team, so eight
        # equivalent phrasings all lost the scope.
        #
        # Also the reporting reading: "reports to" IS the direct
        # relationship. "Directly" makes it explicit, it does not create
        # it — "under" would be subtree.
        # ---------------------------------------------------------------
        "utterance": "who reports to the unit head in Downtown",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "population",
            "subject_level": "advisor",
            "subjects": [
                {"type": "team", "value": "Downtown", "match_confidence": 1.0},
            ],
            "metric": None,
            "metrics": [],
            "filters": [],
            "filter_tree": None,
            "target_level": "advisor",
            "subject_of": "unit_head",
            "relation": "direct",
            "time_range": {"mode": "snapshot", "period": "MTD",
                           "compare_to": None, "confidence": 0.6},
            "sort": {"metric": None, "direction": "desc"},
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.95,
            "intent_confidence": 0.96,
        },
    },

    {
        # ---------------------------------------------------------------
        # HIERARCHY READ — DIRECT REPORTS FROM A LOWER MANAGER
        #
        # The same shape as the Unit Head example, one rung down, so the
        # scope level is learned as a variable rather than as the single
        # value every hierarchy example happened to use.
        #
        # This example previously read "Which teams directly report to BCM
        # Ahmed?" with target_level="team". `team` is the ROOT of the
        # chain (team -> unit_head -> zonal_head -> bcm -> advisor), so
        # nothing is beneath a BCM at team level: it described a traversal
        # running upwards. The validator refuses that pairing outright,
        # so the example could never have been imitated successfully — it
        # taught an org chart this business does not have.
        # ---------------------------------------------------------------
        "utterance": "Which advisors directly report to BCM Ahmed?",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "population",
            "subject_level": "advisor",
            "subjects": [
                {
                    "type": "bcm",
                    "value": "Ahmed",
                    "match_confidence": 1.0,
                },
            ],
            "metric": None,
            "metrics": [],
            "filters": [],
            "filter_tree": None,
            "target_level": "advisor",
            "subject_of": "bcm",
            "relation": "direct",
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.96,
            "intent_confidence": 0.98,
        },
    },

    {
        # ---------------------------------------------------------------
        # HIERARCHY READ — SUBTREE AT AN INTERMEDIATE LEVEL
        #
        # "teams under" means all teams in the subtree, not only immediate
        # reports.
        # ---------------------------------------------------------------
        "utterance": "Show all advisors under BCM Ahmed",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "population",
            "subject_level": "advisor",
            "subjects": [
                {
                    "type": "bcm",
                    "value": "Ahmed",
                    "match_confidence": 1.0,
                },
            ],
            "metric": None,
            "metrics": [],
            "filters": [],
            "filter_tree": None,
            "target_level": "advisor",
            "subject_of": "bcm",
            "relation": "subtree",
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.96,
            "intent_confidence": 0.98,
        },
    },

    {
        # ---------------------------------------------------------------
        # HIERARCHY READ + METRIC
        #
        # A hierarchy relation determines WHO is in scope.
        # The metric determines WHAT is returned for those entities.
        #
        # This prevents the model from treating "under" as a filter and
        # dropping the hierarchy relationship.
        # ---------------------------------------------------------------
        "utterance": "Show connects of advisors directly reporting to Unit Head Ahmed",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "advisor",
            "subjects": [
                {
                    "type": "unit_head",
                    "value": "Ahmed",
                    "match_confidence": 1.0,
                },
            ],
            "metric": {
                "key": "total_connects",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "total_connects",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "target_level": "advisor",
            "subject_of": "unit_head",
            "relation": "direct",
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "total_connects",
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.96,
            "intent_confidence": 0.98,
        },
    },

    {
        # ---------------------------------------------------------------
        # HIERARCHY READ + METRIC + SUBTREE
        # ---------------------------------------------------------------
        "utterance": "What is the revenue of advisors under Unit Head Ahmed?",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "leaderboard",
            "operation": "leaderboard",
            "subject_level": "advisor",
            "subjects": [
                {
                    "type": "unit_head",
                    "value": "Ahmed",
                    "match_confidence": 1.0,
                },
            ],
            "metric": {
                "key": "mtd_cleared",
                "confidence": 0.95,
            },
            "metrics": [
                {
                    "key": "mtd_cleared",
                    "confidence": 0.95,
                },
            ],
            "filters": [],
            "filter_tree": None,
            "target_level": "advisor",
            "subject_of": "unit_head",
            "relation": "subtree",
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": "mtd_cleared",
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.95,
            "intent_confidence": 0.97,
        },
    },

    {
        # ---------------------------------------------------------------
        # HIERARCHY READ — TARGET LEVEL INFERRED FROM GENERIC PEOPLE WORD
        #
        # "people" means the leaf population according to the semantic
        # rules. This example teaches that generic population language
        # can still require target_level resolution.
        # ---------------------------------------------------------------
        "utterance": "How many people directly report to Unit Head Ahmed?",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "population",
            "subject_level": "advisor",
            "subjects": [
                {
                    "type": "unit_head",
                    "value": "Ahmed",
                    "match_confidence": 1.0,
                },
            ],
            "metric": None,
            "metrics": [],
            "filters": [],
            "filter_tree": None,
            "target_level": "advisor",
            "subject_of": "unit_head",
            "relation": "direct",
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.95,
            "intent_confidence": 0.97,
        },
    },

    {
        # ---------------------------------------------------------------
        # HIERARCHY READ — NO DEPTH WORD
        #
        # This teaches the intended default: when the user asks for the
        # entities beneath a manager without specifying "directly", the
        # relation is subtree.
        # ---------------------------------------------------------------
        "utterance": "Which BCMs are under Zonal Head Bilal?",
        "prior_ir": None,
        "expect_valid": True,
        "ir": {
            "intent": "filtered_list",
            "operation": "population",
            "subject_level": "bcm",
            "subjects": [
                {
                    "type": "zonal_head",
                    "value": "Bilal",
                    "match_confidence": 1.0,
                },
            ],
            "metric": None,
            "metrics": [],
            "filters": [],
            "filter_tree": None,
            "target_level": "bcm",
            "subject_of": "zonal_head",
            "relation": "subtree",
            "time_range": {
                "mode": "snapshot",
                "period": "MTD",
                "compare_to": None,
                "confidence": 0.6,
            },
            "sort": {
                "metric": None,
                "direction": "desc",
            },
            "limit": None,
            "group_by": None,
            "flat": False,
            "overall_confidence": 0.96,
            "intent_confidence": 0.98,
        },
    },
]


def render_examples() -> str:
    """
    Render the few-shot examples compactly.

    A field whose value equals the schema default is omitted. The grammar
    constrained output schema still requires the model to emit all fields.

    The examples intentionally preserve the semantic fields that carry
    teaching signal, especially the hierarchy-read fields:
        target_level
        subject_of
        relation
    """
    import json

    defaults = {
        "operation": None,
        "subjects": [],
        "metric": None,
        "metrics": [],
        "filters": [],
        "filter_tree": None,
        "target_level": None,
        "subject_of": None,
        "relation": "subtree",
        "limit": None,
        "group_by": None,
        "flat": False,
        "sort": {
            "metric": None,
            "direction": "desc",
        },
    }

    def compact(ir: dict) -> dict:
        return {
            key: value
            for key, value in ir.items()
            if key not in defaults or value != defaults[key]
        }

    lines = [
        "Examples (follow these exactly — they are semantic structures to "
        "imitate, never query-to-answer mappings).",
        (
            "Fields omitted from an example are defaults — "
            "you must still emit every field required by the output schema."
        ),
        (
            "IMPORTANT HIERARCHY RULE: when an example contains "
            "target_level, subject_of, or relation, preserve all three "
            "together. They form one hierarchy-read relationship."
        ),
    ]

    for index, example in enumerate(EXAMPLES, 1):
        lines.append(
            f'Example {index} — User: "{example["utterance"]}"'
        )

        if example["prior_ir"]:
            lines.append(
                "  (Previous turn's query was: "
                f"{json.dumps(compact(example['prior_ir']))})"
            )

        lines.append(
            f"  -> {json.dumps(compact(example['ir']))}"
        )

    return "\n".join(lines)