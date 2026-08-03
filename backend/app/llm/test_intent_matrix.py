"""
THE INTENT MATRIX — executable documentation of planner routing.

Every row is (query, expected intent). Together they cover every
supported intent, common paraphrases users actually type, the overlap
cases where a query matches several intents' triggers, and every
previously reported failure.

This file is the regression net for intent selection. Because selection
is now SCORED rather than ordered (see query_planner.py), a change to a
weight in intent_catalog.py shows up here as a specific row flipping —
which is the point. Under the old first-match design the same change was
invisible until a user reported a wrong answer.

Entities are supplied directly rather than extracted from a database:
this suite tests ROUTING, not entity resolution (which has its own tests
in test_advisor_resolver.py / test_entity_extractor.py). Keeping them
separate means a routing failure here can't be a grounding failure in
disguise.
"""

import pytest

from app.llm import intent_catalog as cat
from app.llm.query_planner import build_query_plan, score_intents

# Entity fixtures reused across rows.
TEAM = {"team": "Blue Area"}
COMPANY = {"company": "Graana"}
UNIT = {"unit_head": "Adeel Dogar"}
ZONE = {"zonal_head": "Salman Arshad"}
CENTER = {"office": "F-11 Center"}
PERSON = {"advisor_name": "Waqar Haider", "advisor_wid": 41, "advisor_match_score": 1.0}
PERSON_IN_TEAM = {**PERSON, **TEAM}
TWO_COMPANIES = {"companies": ["Graana", "Agency21"], "company": "Graana"}
TWO_TEAMS = {"teams": ["Blue Area", "DHA"], "team": "Blue Area"}
TWO_UNITS = {"unit_heads": ["Kaleem Ahmed", "Adeel Dogar"], "unit_head": "Kaleem Ahmed"}
AMBIGUOUS_PERSON = {
    "advisor_name": "Yasir Ali", "advisor_match_score": 1.0, "advisor_ambiguous": True,
}
AMBIGUOUS_LEVELS = {
    "advisor_name": "Ali Murtaza", "advisor_wid": 288, "advisor_match_score": 1.0,
    "unit_head": "Ali Murtaza",
    "ambiguous_entity": {"value": "Ali Murtaza", "levels": ["advisor", "unit_head"]},
}


# =====================================================================
# (query, entities, expected plan action)
# =====================================================================
MATRIX: list[tuple[str, dict, str]] = [
    # ---------- ROSTER: enumerate people in a group ----------
    ("all advisors in Blue Area", TEAM, "roster"),
    ("list advisors in Blue Area", TEAM, "roster"),
    ("show advisors in Blue Area", TEAM, "roster"),
    ("advisors in Blue Area", TEAM, "roster"),
    ("advisors from Blue Area", TEAM, "roster"),
    ("employees in Blue Area", TEAM, "roster"),
    ("staff in Blue Area", TEAM, "roster"),
    ("agents in Blue Area", TEAM, "roster"),
    ("who works in Blue Area", TEAM, "roster"),
    ("who works under Adeel Dogar", UNIT, "roster"),
    ("give me the advisors in Blue Area", TEAM, "roster"),
    ("name the advisors in Blue Area", TEAM, "roster"),
    ("list all advisors in Graana", COMPANY, "roster"),
    ("all advisors under Adeel Dogar", UNIT, "roster"),
    ("show advisors under Adeel Dogar", UNIT, "roster"),
    ("advisors assigned to F-11 Center", CENTER, "roster"),
    ("all people in Blue Area", TEAM, "roster"),
    ("advisors in Salman Arshad", ZONE, "roster"),

    # ---------- HIERARCHY: the group under someone, nested ----------
    ("show Adeel Dogar's team", UNIT, "breakdown"),
    ("Adeel Dogar's team", UNIT, "breakdown"),
    ("who reports to Adeel Dogar", UNIT, "breakdown"),
    ("who is in Adeel Dogar's team", UNIT, "breakdown"),
    ("who are in Adeel Dogar's team", UNIT, "breakdown"),
    ("team of Adeel Dogar", UNIT, "breakdown"),
    ("members of Adeel Dogar's team", UNIT, "breakdown"),
    ("Salman Arshad's team", ZONE, "breakdown"),
    ("show me Adeel Dogar's team", UNIT, "breakdown"),
    ("Adeel Dogar's reports", UNIT, "breakdown"),
    ("Adeel Dogar's people", UNIT, "breakdown"),

    # ---------- REVERSE HIERARCHY: the person above someone ----------
    ("who is Waqar Haider's BM", PERSON, "reverse_hierarchy"),
    # Phase 5.4 — every level above someone, not just one.
    ("who is above Waqar Haider", PERSON, "ancestry"),
    ("show me the full hierarchy above Waqar Haider", PERSON, "ancestry"),
    ("who is Waqar Haider's ZM", PERSON, "reverse_hierarchy"),
    ("who is Waqar Haider's RM", PERSON, "reverse_hierarchy"),
    ("who is Waqar Haider's unit head", PERSON, "reverse_hierarchy"),
    ("who is Waqar Haider's zonal head", PERSON, "reverse_hierarchy"),
    ("who is Waqar Haider's manager", PERSON, "reverse_hierarchy"),
    ("who is Waqar Haider's boss", PERSON, "reverse_hierarchy"),
    ("who is Waqar Haider's supervisor", PERSON, "reverse_hierarchy"),
    ("who does Waqar Haider report to", PERSON, "reverse_hierarchy"),
    ("who is BM of Waqar Haider", PERSON, "reverse_hierarchy"),
    ("Waqar Haider's manager", PERSON, "reverse_hierarchy"),
    ("who is his BM", PERSON, "reverse_hierarchy"),
    ("what about her zonal head", PERSON, "reverse_hierarchy"),
    ("who is Waqar Haider managed by", PERSON, "reverse_hierarchy"),

    # ---------- ADVISOR PROFILE ----------
    ("tell me about Waqar Haider", PERSON, "lookup"),
    ("Waqar Haider", PERSON, "lookup"),
    ("how is Waqar Haider doing", PERSON, "lookup"),
    ("what about Waqar Haider", PERSON, "lookup"),
    # "performance" resolves achievement_pct in the ontology, but on its
    # own it asks how somebody is DOING — the profile answers that, one
    # percentage does not. See cat.GENERAL_INTEREST_SYNONYMS.
    ("performance of Waqar Haider", PERSON, "lookup"),
    ("Waqar Haider details", PERSON, "lookup"),
    ("info on Waqar Haider", PERSON, "lookup"),
    ("show Waqar Haider profile", PERSON, "lookup"),
    ("who is Waqar Haider", PERSON, "lookup"),

    # ---------- ADVISOR METRIC: ONE measure for one person ----------
    # The profile CONTAINS these numbers, which is exactly why they need
    # their own intent: containing an answer is not giving one.
    ("connects of Waqar Haider", PERSON, "advisor_metric"),
    ("meetings of Waqar Haider", PERSON, "advisor_metric"),
    ("pipeline of Waqar Haider", PERSON, "advisor_metric"),
    ("overdue of Waqar Haider", PERSON, "advisor_metric"),
    ("target of Waqar Haider", PERSON, "advisor_metric"),
    ("cleared of Waqar Haider", PERSON, "advisor_metric"),
    ("cr booked of Waqar Haider", PERSON, "advisor_metric"),
    ("Waqar Haider's revenue", PERSON, "advisor_metric"),
    ("how many meetings did Waqar Haider have", PERSON, "advisor_metric"),
    # A specific phrasing containing a general word is still specific.
    ("performance against target of Waqar Haider", PERSON, "advisor_metric"),

    # ---------- ENTITY SUMMARY: a group's aggregates ----------
    ("how is Blue Area doing", TEAM, "summary"),
    ("Blue Area", TEAM, "summary"),
    ("tell me about Blue Area", TEAM, "summary"),
    ("how is Graana performing", COMPANY, "summary"),
    ("Graana", COMPANY, "summary"),
    ("tell me about unit head Adeel Dogar", UNIT, "breakdown"),
    ("how is Adeel Dogar doing", UNIT, "breakdown"),
    ("F-11 Center", CENTER, "breakdown"),

    # ---------- LEADERBOARD: metric rankings ----------
    ("top 5 advisors by revenue", {"limit": 5}, "leaderboard"),
    ("best advisors by revenue", {}, "leaderboard"),
    ("worst advisors by overdue", {}, "leaderboard"),
    ("highest connects", {}, "leaderboard"),
    ("lowest achievement", {}, "leaderboard"),
    ("rank teams by revenue", {}, "leaderboard"),
    ("leaderboard by connects", {}, "leaderboard"),
    ("top 10 teams by achievement", {"limit": 10}, "leaderboard"),
    ("who has the most connects", {}, "leaderboard"),
    ("give me target achievement", {}, "leaderboard"),
    ("top unit heads by connects", {}, "leaderboard"),
    ("top 5 advisors by ytd revenue", {"limit": 5}, "leaderboard"),
    ("show me revenue", {}, "leaderboard"),
    ("bookings leaderboard", {}, "leaderboard"),
    ("top advisors by conversion", {}, "leaderboard"),
    ("worst overdue teams", {}, "leaderboard"),

    # ---------- ATTENDANCE ----------
    ("who was late today", {"attendance_status": "Late"}, "attendance_filter"),
    ("show not marked people in Blue Area",
     {"attendance_status": "Not Marked", **TEAM}, "attendance_filter"),
    ("who was absent", {"attendance_status": "Absent"}, "attendance_filter"),
    ("late advisors in Blue Area", {"attendance_status": "Late", **TEAM}, "attendance_filter"),

    # ---------- CLARIFICATION ----------
    ("tell me about Yasir Ali", AMBIGUOUS_PERSON, "clarify_person"),
    ("how is Yasir Ali doing", AMBIGUOUS_PERSON, "clarify_person"),
    ("tell me about Ali Murtaza", AMBIGUOUS_LEVELS, "clarify_ambiguous"),
    ("Ali Murtaza", AMBIGUOUS_LEVELS, "clarify_ambiguous"),

    # ---------- OVERLAP: roster phrasing + ranking ----------
    # The entity SCOPES a ranking rather than being enumerated.
    ("top 5 advisors in Blue Area by revenue", {**TEAM, "limit": 5}, "leaderboard"),
    ("best advisors in Blue Area by connects", TEAM, "leaderboard"),
    ("top advisors under Adeel Dogar by revenue", UNIT, "leaderboard"),
    # ...but with no metric named, it stays a roster.
    ("all advisors in Blue Area", TEAM, "roster"),

    # ---------- OVERLAP: relational + roster ----------
    # "advisors" means enumerate; "team" means describe the team.
    ("show advisors under Adeel Dogar", UNIT, "roster"),
    ("show Adeel Dogar's team", UNIT, "breakdown"),
    ("list all advisors under Adeel Dogar", UNIT, "roster"),

    # ---------- OVERLAP: forward vs reverse hierarchy ----------
    # Near-identical strings, opposite questions.
    ("who reports to Adeel Dogar", UNIT, "breakdown"),
    ("who does Waqar Haider report to", PERSON, "reverse_hierarchy"),

    # ---------- OVERLAP: person named, but a group asked about ----------
    ("show Ali Murtaza's team", AMBIGUOUS_LEVELS, "breakdown"),
    ("who is Ali Murtaza's unit head", AMBIGUOUS_LEVELS, "reverse_hierarchy"),

    # ---------- OVERLAP: person + team both present ----------
    ("tell me about Waqar Haider", PERSON_IN_TEAM, "lookup"),
    ("all advisors in Blue Area", PERSON_IN_TEAM, "roster"),

    # ---------- Previously reported failures ----------
    ("Show Adeel Dogar's team", UNIT, "breakdown"),
    ("Who reports to Adeel Dogar?", UNIT, "breakdown"),
    ("Who is BM of Ali Murtaza?", AMBIGUOUS_LEVELS, "reverse_hierarchy"),
    ("Show Ali Murtaza's team", AMBIGUOUS_LEVELS, "breakdown"),
    ("All advisors in Blue Area", TEAM, "roster"),
    ("Advisors in North/KPK Region", {"team": "North/KPK Region"}, "roster"),
    ("show me the connects of zonal head Salman Arshad and his team",
     {**ZONE, "advisor_name": "Salman Arshad", "advisor_match_score": 0.9}, "leaderboard"),

    # ---------- COMPARISON: two entities side by side ----------
    ("compare Graana and Agency21", TWO_COMPANIES, "comparison"),
    ("compare Graana vs Agency21", TWO_COMPANIES, "comparison"),
    ("compare Graana versus Agency21", TWO_COMPANIES, "comparison"),
    ("difference between Graana and Agency21", TWO_COMPANIES, "comparison"),
    ("Graana vs Agency21", TWO_COMPANIES, "comparison"),
    ("which is performing better, Graana or Agency21?", TWO_COMPANIES, "comparison"),
    ("who is doing better, Graana or Agency21", TWO_COMPANIES, "comparison"),
    ("how does Graana compare to Agency21", TWO_COMPANIES, "comparison"),
    ("comparison of Graana and Agency21", TWO_COMPANIES, "comparison"),
    ("compare Blue Area and DHA", TWO_TEAMS, "comparison"),
    ("compare Blue Area with DHA", TWO_TEAMS, "comparison"),
    ("compare Kaleem's team with Adeel Dogar's team", TWO_UNITS, "comparison"),
    ("compare Kaleem and Adeel Dogar", TWO_UNITS, "comparison"),
    # a comparison spanning LEVELS keeps each entity's own type
    ("compare Blue Area and Graana", {**TEAM, **COMPANY, "teams": ["Blue Area"],
                                      "companies": ["Graana"]}, "comparison"),

    # ---------- COMPARISON with an explicit metric ----------
    ("compare Graana and Agency21 by revenue", TWO_COMPANIES, "comparison"),
    ("compare Blue Area and DHA attendance", TWO_TEAMS, "comparison"),
    ("compare Kaleem's team and Adeel Dogar's team by meetings", TWO_UNITS, "comparison"),
    ("compare Graana and Agency21 on connects", TWO_COMPANIES, "comparison"),

    # ---------- OVERLAP: comparison beats leaderboard when both fire ----------
    # "compare A and B by revenue" is two-sided; a ranking answers with
    # one list that drops the pairing entirely.
    ("compare Graana and Agency21 by revenue", TWO_COMPANIES, "comparison"),
    # ...but a bare "compare" with nothing to compare must NOT claim it
    ("compare revenue", {}, "leaderboard"),
    # one side named, the other missing -> say so, never silently answer
    # about the side that did resolve
    ("compare Graana", COMPANY, "comparison_incomplete"),
    ("compare Blue Area and DHA", TEAM, "comparison_incomplete"),

    # ---------- Unresolvable ----------
    ("asdkjalksjd nonsense", {}, "unresolved"),
    ("hello there friend", {}, "unresolved"),
    ("", {}, "unresolved"),
]


@pytest.mark.parametrize("query,entities,expected", MATRIX, ids=[f"{q[:52]}" for q, _e, _x in MATRIX])
def test_intent_matrix(query, entities, expected):
    plan = build_query_plan(query, dict(entities))
    assert plan.action == expected, (
        f"{query!r} routed to {plan.action!r} (score {plan.intent_score}, "
        f"evidence {plan.intent_evidence}, runner-up {plan.runner_up})"
    )


def test_matrix_covers_every_documented_intent():
    """Documentation can't drift: every intent in the catalog must appear
    in the matrix, and every matrix expectation must be a real action."""
    produced = {build_query_plan(q, dict(e)).action for q, e, _x in MATRIX}
    documented_actions = {
        "clarify_ambiguous", "clarify_person", "roster", "breakdown",
        "reverse_hierarchy", "lookup", "attendance_filter", "summary",
        "leaderboard", "comparison", "comparison_incomplete", "unresolved",
        # M7 — one metric for one person.
        "advisor_metric",
        # Phase 5.4 — the whole reporting line, not one level of it.
        "ancestry",
    }
    assert produced <= documented_actions, f"undocumented action: {produced - documented_actions}"
    # every catalog intent is exercised
    assert len(cat.INTENT_DOCS) == 12
    for action in ("roster", "breakdown", "reverse_hierarchy", "lookup",
                   "summary", "leaderboard", "attendance_filter",
                   "clarify_person", "clarify_ambiguous", "comparison",
                   "advisor_metric", "ancestry"):
        assert action in produced, f"matrix never produces {action}"


def test_matrix_is_large_enough_to_be_a_net():
    assert len(MATRIX) >= 100, f"matrix has only {len(MATRIX)} rows"


# =====================================================================
# The scoring mechanism itself
# =====================================================================

def test_every_plan_carries_its_evidence():
    """A routing decision must be inspectable without re-running the
    planner — this is what lands in the request trace."""
    plan = build_query_plan("all advisors in Blue Area", dict(TEAM))
    assert plan.intent_score > 0
    assert "roster_phrase" in plan.intent_evidence
    assert plan.runner_up is not None


def test_overlapping_query_records_the_runner_up():
    """The whole point of scoring: a conflict is a comparison, and the
    loser is visible."""
    _ctx, candidates = score_intents("top 5 advisors in Blue Area by revenue", {**TEAM, "limit": 5})
    intents = [c.intent for c in candidates]
    assert intents[0] == "leaderboard"
    assert "roster" in intents, "the roster reading should be a scored candidate, not silently absent"


def test_ranking_plus_metric_outscores_a_roster_phrase():
    """Replaces the hand-coded 'decline if metric and ranking' guards that
    used to live inside the roster and hierarchy branches."""
    _ctx, candidates = score_intents("top 5 advisors in Blue Area by revenue", {**TEAM, "limit": 5})
    by_intent = {c.intent: c.score for c in candidates}
    assert by_intent["leaderboard"] > by_intent["roster"]


def test_a_bare_roster_phrase_outscores_the_summary_reading():
    _ctx, candidates = score_intents("all advisors in Blue Area", dict(TEAM))
    by_intent = {c.intent: c.score for c in candidates}
    assert by_intent["roster"] > by_intent.get("entity_summary", 0)


def test_clarification_is_a_hard_gate():
    """An unresolvable ambiguity must beat every other reading — answering
    any of them would be a guess."""
    _ctx, candidates = score_intents("tell me about Ali Murtaza", dict(AMBIGUOUS_LEVELS))
    assert candidates[0].intent == "clarify_ambiguous"
    assert candidates[0].score >= cat.W_HARD_GATE


def test_scores_are_deterministic():
    """Same input, same decision — every time."""
    runs = {build_query_plan("show advisors under Adeel Dogar", dict(UNIT)).action for _ in range(25)}
    assert runs == {"roster"}


def test_identity_resolution_strengthens_a_person_intent():
    """A name resolved to a specific wid is stronger evidence than a bare
    name match."""
    _c1, with_wid = score_intents("who is Waqar Haider's BM", dict(PERSON))
    _c2, without = score_intents(
        "who is Waqar Haider's BM",
        {"advisor_name": "Waqar Haider", "advisor_match_score": 1.0},
    )
    assert with_wid[0].score > without[0].score


def test_weak_ranking_cue_does_not_by_itself_make_a_ranking():
    """"show me" opens any request and is barely evidence — treating it
    as equal to "top" is what let "show me X's team" look like a
    ranking."""
    plan = build_query_plan("show me Adeel Dogar's team", dict(UNIT))
    assert plan.action == "breakdown"
