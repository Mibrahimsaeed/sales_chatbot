"""
Locks in the two failing queries from the bug report — both should now
resolve deterministically through the ontology + planner, with zero LLM
calls involved.
"""

import pytest

from app.llm.query_planner import build_query_plan
from app.llm.metric_ontology import resolve_metric


def test_target_achievement_metric_resolves():
    assert resolve_metric("give me target achievement") == "achievement_pct"


def test_give_me_target_achievement_plans_team_leaderboard():
    plan = build_query_plan("give me target achievement", {})
    assert plan.action == "leaderboard"
    assert plan.metric == "achievement_pct"
    assert plan.level == "team"   # primary_level default when no level keyword present


def test_top_5_advisors_by_target_achievement():
    entities = {"limit": 5}
    plan = build_query_plan("top 5 advisors by target achievement", entities)
    assert plan.action == "leaderboard"
    assert plan.metric == "achievement_pct"
    assert plan.level == "advisor"   # "advisors" keyword overrides the metric's team default
    assert plan.limit == 5


def test_worst_overdue_flips_to_ascending_false_is_still_a_ranking():
    """UPDATED BY PHASE 2 — and note this test's own NAME already said
    "ascending false"; only the assertion disagreed.

    "Worst" is a QUALITY word, not a numeric one. The worst teams on
    overdue are the ones with the MOST overdue, so the ranking descends.
    The pre-Phase-2 vocabulary treated "worst" as a synonym for "lowest",
    which is right for revenue and backwards for a metric where less is
    better — it asked for the worst offenders and returned the cleanest
    teams."""
    plan = build_query_plan("worst overdue teams", {})
    assert plan.action == "leaderboard"
    assert plan.metric == "overdue"
    assert plan.ascending is False   # worst overdue = MOST overdue


def test_no_metric_no_entity_is_unresolved():
    plan = build_query_plan("asdkjalksjd nonsense", {})
    assert plan.action == "unresolved"


# ---- Hierarchy rework: new levels (unit_head / zonal_head / business_center) ----

def test_bare_unit_head_mention_plans_a_breakdown_not_a_summary():
    entities = {"unit_head": "Zeeshan Tariq"}
    plan = build_query_plan("tell me about unit head Zeeshan Tariq", entities)
    assert plan.action == "breakdown"
    assert plan.level == "unit_head"
    assert plan.entity_value == "Zeeshan Tariq"


def test_bare_zonal_head_mention_plans_a_breakdown():
    entities = {"zonal_head": "Ahmed Ali"}
    plan = build_query_plan("how is zonal head Ahmed Ali doing", entities)
    assert plan.action == "breakdown"
    assert plan.level == "zonal_head"


def test_bare_business_center_mention_plans_a_breakdown():
    entities = {"office": "F-11 Business Center"}
    plan = build_query_plan("show F-11 Business Center", entities)
    assert plan.action == "breakdown"
    assert plan.level == "office"


def test_top_unit_heads_by_connects_is_a_ranking_not_a_breakdown():
    entities = {"limit": 5}
    plan = build_query_plan("top 5 unit heads by connects", entities)
    assert plan.action == "leaderboard"
    assert plan.level == "unit_head"
    assert plan.metric == "total_connects"
    assert plan.limit == 5


def test_team_summary_still_wins_over_new_levels_when_both_absent():
    # existing team/company behavior is untouched — a bare team mention
    # still plans "summary", not "breakdown"
    entities = {"team": "Blue Area"}
    plan = build_query_plan("tell me about Blue Area", entities)
    assert plan.action == "summary"
    assert plan.level == "team"


# ---- Phase 2: ambiguity + flat opt-in ----

def test_ambiguous_entity_short_circuits_before_any_other_branch():
    entities = {
        "ambiguous_entity": {"value": "Zeeshan Tariq", "levels": ["advisor", "unit_head"]},
        "advisor_name": "Zeeshan Tariq",
        "unit_head": "Zeeshan Tariq",
    }
    plan = build_query_plan("tell me about Zeeshan Tariq", entities)
    assert plan.action == "clarify_ambiguous"
    assert plan.ambiguous == {"value": "Zeeshan Tariq", "levels": ["advisor", "unit_head"]}


def test_explicit_level_wording_resolves_ambiguity_without_asking():
    """Live-verified real case: 'Noman Ziafat' is both a real advisor and a
    unit head (Advisor.bm) for other advisors — the user saying "unit
    head" explicitly IS the disambiguation and must not be asked again."""
    entities = {
        "ambiguous_entity": {"value": "Noman Ziafat", "levels": ["advisor", "unit_head"]},
        "advisor_name": "Noman Ziafat",
        "unit_head": "Noman Ziafat",
    }
    plan = build_query_plan("tell me about unit head Noman Ziafat", entities)
    assert plan.action == "breakdown"
    assert plan.level == "unit_head"
    assert plan.entity_value == "Noman Ziafat"


def test_ambiguity_still_asked_when_text_names_neither_level_explicitly():
    entities = {
        "ambiguous_entity": {"value": "Noman Ziafat", "levels": ["advisor", "unit_head"]},
        "advisor_name": "Noman Ziafat",
        "unit_head": "Noman Ziafat",
    }
    plan = build_query_plan("tell me about Noman Ziafat", entities)
    assert plan.action == "clarify_ambiguous"


def test_flat_phrase_sets_flat_on_breakdown_plan():
    entities = {"unit_head": "Zeeshan Tariq"}
    plan = build_query_plan("give me a flat list of unit head Zeeshan Tariq's advisors", entities)
    assert plan.action == "breakdown"
    assert plan.flat is True


def test_breakdown_defaults_to_nested_without_a_flat_phrase():
    entities = {"unit_head": "Zeeshan Tariq"}
    plan = build_query_plan("tell me about unit head Zeeshan Tariq", entities)
    assert plan.action == "breakdown"
    assert plan.flat is False


# ---- Bug fix (live-reported): explicit level wording ignored when an
# UNRELATED level keyword also appears elsewhere in the sentence ----

def test_zonal_head_wording_not_defeated_by_unrelated_team_word_elsewhere():
    """The exact reported reproduction: 'zonal head X' is explicit, but the
    sentence also contains the bare word 'team' ("...and his team") — which
    used to make the old single-global-_detect_level check return "team"
    instead of "zonal_head" (since "team" is checked earlier in the fixed
    priority order), even though "team" isn't one of the ambiguous
    candidates (advisor/zonal_head) at all."""
    entities = {
        "ambiguous_entity": {"value": "Muhammad Ayaz", "levels": ["advisor", "zonal_head"]},
        "advisor_name": "Muhammad Ayaz",
        "zonal_head": "Muhammad Ayaz",
    }
    plan = build_query_plan("show me the connects of zonal head Muhammad Ayaz and his team", entities)
    assert plan.action != "clarify_ambiguous"


@pytest.mark.parametrize("level,phrase", [
    ("advisor", "tell me about advisor Ali Raza"),
    ("team", "tell me about team Ali Raza"),
    ("unit_head", "tell me about unit head Ali Raza"),
    ("zonal_head", "tell me about zonal head Ali Raza"),
    ("office", "tell me about business center Ali Raza"),
])
def test_explicit_wording_resolves_ambiguity_for_every_level(level, phrase):
    # "company" as the other candidate for every case — never one of the
    # 5 levels under test, so it's a safe stand-in "some other level".
    entities = {
        "ambiguous_entity": {"value": "Ali Raza", "levels": ["company", level]},
        "company": "Ali Raza",
        ("advisor_name" if level == "advisor" else level): "Ali Raza",
    }
    plan = build_query_plan(phrase, entities)
    assert plan.action != "clarify_ambiguous"

# ---- Phase 4: explicit intent priority ----

def _ent(**kw):
    """Entities as the extractor would emit them for a resolved person."""
    base = {}
    base.update(kw)
    return base


def test_priority_hierarchy_beats_advisor_profile():
    """"Show Ali's team" names a person but ASKS about a group. Advisor
    lookup used to be the first branch and claimed it, returning one
    individual's profile."""
    entities = _ent(advisor_name="Ali Murtaza", advisor_wid=288, unit_head="Ali Murtaza")
    plan = build_query_plan("show ali murtaza's team", entities)
    assert plan.action == "breakdown"
    assert plan.level == "unit_head"


def test_priority_who_reports_to_is_forward_hierarchy():
    entities = _ent(advisor_name="Ali Murtaza", advisor_wid=288, unit_head="Ali Murtaza")
    plan = build_query_plan("who reports to ali murtaza", entities)
    assert plan.action == "breakdown"


def test_priority_reverse_hierarchy_for_possessive_manager_role():
    entities = _ent(advisor_name="Kainat Khalid", advisor_wid=7)
    plan = build_query_plan("who is kainat khalid's bm", entities)
    assert plan.action == "reverse_hierarchy"
    assert plan.level == "bm"
    assert plan.entity_wid == 7


def test_reverse_hierarchy_detects_the_level_asked_about():
    entities = _ent(advisor_name="Kainat Khalid", advisor_wid=7)
    assert build_query_plan("who is kainat khalid's zonal head", entities).level == "zonal_head"
    assert build_query_plan("who is kainat khalid's unit head", entities).level == "unit_head"
    # a generic manager word means the immediate manager
    assert build_query_plan("who is kainat khalid's manager", entities).level == "unit_head"


def test_who_does_x_report_to_is_reverse_not_forward():
    """"who reports to X" (X's reports) and "who does X report to" (X's
    manager) are near-identical strings asking opposite questions."""
    entities = _ent(advisor_name="Kainat Khalid", advisor_wid=7)
    assert build_query_plan("who does kainat khalid report to", entities).action == "reverse_hierarchy"


def test_priority_advisor_profile_when_nothing_relational():
    entities = _ent(advisor_name="Kainat Khalid", advisor_wid=7)
    plan = build_query_plan("tell me about kainat khalid", entities)
    assert plan.action == "lookup"
    assert plan.entity_wid == 7


def test_priority_clarification_outranks_everything():
    entities = _ent(
        advisor_name="Ali Murtaza", advisor_wid=288,
        ambiguous_entity={"value": "Ali Murtaza", "levels": ["advisor", "zonal_head"]},
        zonal_head="Ali Murtaza",
    )
    # no explicit level word, not relational -> must ask, not guess
    plan = build_query_plan("tell me about ali murtaza", entities)
    assert plan.action == "clarify_ambiguous"


def test_leaderboard_still_wins_for_ranking_phrasing():
    plan = build_query_plan("top 5 advisors by revenue", {"limit": 5})
    assert plan.action == "leaderboard"
    assert plan.metric == "mtd_cleared"


def test_relational_phrasing_does_not_hijack_a_ranking():
    """"top advisors under unit head X" is still a ranking — the
    relational cue must not turn every mention into a breakdown."""
    entities = _ent(unit_head="Fraz Khalid", limit=5)
    plan = build_query_plan("top 5 advisors by revenue under fraz khalid", entities)
    assert plan.action == "leaderboard"
