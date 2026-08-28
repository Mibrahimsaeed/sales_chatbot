""""Which BCMs work under Unit Head X" asks for BCMs, not for advisors.

THE DEFECT. The question names two levels — one identifies the manager,
one is the population wanted — and the planner read only the first. With
"directly" in the sentence `_score_direct_reports` picked both up, via
`_named_target_level`, and answered correctly. Without it the identical
sentence fell to `roster` or `hierarchy`, neither of which carries a
target level at all, so:

  - "Which BCMs work under Unit Head X"  answered with a 129-ADVISOR
    unit summary;
  - "Give me the BCMs who work under X"  listed 129 ADVISORS under a
    heading that said BCMs, which is the dangerous one because it looks
    like an answer;
  - "How many BCMs are under X" and "How many advisors are under X"
    returned the SAME number.

The target level was understood and then dropped between the planner and
the answer. Nothing new was needed to fix it: `QueryPlan.target_level`,
`_named_target_level` and the whole direct-reports dispatch already
existed, and this reads them transitively.

The fixture is test_direct_reports' org, deliberately: the two readings
of one question must be tested against the same tree, and that tree
already models the awkward part of production — a manager who is his own
sub-level, so self-exclusion is exercised rather than assumed.
"""

import pytest

from app.database.models import Advisor
from app.llm import entity_extractor
from app.llm.nlu_pipeline import resolve
from app.services import chat_service, hierarchy_service

# team -> unit_head(rm) -> zonal_head(portfolio_lead) -> bcm(management_lead)
#
# Uma heads AMD. Beneath her: Zed is a Zonal Head, Bee is a BCM under Zed,
# and Uma is her own Zonal Head and BCM for the two "Direct" advisors.
_ORG = [
    (1, "Uma",        "AMD", "Uma",  "Uma", "Uma"),
    (2, "Direct One", "AMD", "Uma",  "Uma", "Uma"),
    (3, "Direct Two", "AMD", "Uma",  "Uma", "Uma"),
    (4, "Zed",        "AMD", "Uma",  "Zed", "Zed"),
    (5, "Under Zed",  "AMD", "Uma",  "Zed", "Zed"),
    (6, "Bee",        "AMD", "Uma",  "Zed", "Bee"),
    (7, "Under Bee",  "AMD", "Uma",  "Zed", "Bee"),
    (8, "Outsider",   "OTH", "Otto", "Otto", "Otto"),
]


@pytest.fixture()
def org(db_session):
    for wid, name, team, rm, pl, ml in _ORG:
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                               rm=rm, portfolio_lead=pl, management_lead=ml,
                               in_master_sheet=True))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    return db_session


def _answer(db, text):
    """The WHOLE pipeline. A plan built correctly and then routed away by
    nlu_pipeline answers "I'm not tracking that one", so an assertion made
    against build_query_plan alone can pass while the app is broken."""
    resolution = resolve(text, db, session_id=None)
    response = chat_service._dispatch(db, resolution)
    return resolution, response, str(response.get("reply") or "")


def _members(response):
    return sorted(m["name"] for m in (response.get("data") or {}).get("members", []))


# --------------------------------------------------- the reported query
def test_bcms_under_a_unit_head_are_bcms_not_advisors(org):
    """THE bug. Uma's subtree holds 7 advisors and 2 BCMs beneath her."""
    resolution, response, reply = _answer(org, "Which BCMs work under Unit Head Uma?")
    assert resolution.plan.action == "scoped_reports"
    assert resolution.plan.target_level == "bcm"
    assert response["data"]["count"] == 2
    assert _members(response) == ["Bee", "Zed"]
    assert "BCM" in reply


def test_the_same_question_as_a_count_gives_the_same_population(org):
    """"How many BCMs" and "which BCMs" are one question asked twice —
    they used to return 7 and 7, both wrong, from the advisor scope."""
    _, listed, _ = _answer(org, "Which BCMs work under Unit Head Uma?")
    _, counted, _ = _answer(org, "How many BCMs work under Unit Head Uma?")
    assert counted["data"]["count"] == listed["data"]["count"] == 2


def test_a_bcm_count_and_an_advisor_count_no_longer_agree(org):
    """The clearest symptom of the bug: two different questions, one
    answer. Both returned the advisor population because the BCM question
    had nowhere to put its target level.

    Compared on the REPLY, because the two are now served by different
    readings — the BCM count by scoped_reports, the advisor count by the
    pre-existing team-size path this change deliberately leaves alone."""
    _, bcms, bcm_reply = _answer(org, "How many BCMs are under Unit Head Uma?")
    _, _, advisor_reply = _answer(org, "How many advisors are under Unit Head Uma?")
    assert bcms["data"]["count"] == 2
    assert "2 BCMs" in bcm_reply
    assert bcm_reply != advisor_reply
    assert "2 BCMs" not in advisor_reply


# ------------------------------------------------------ every phrasing
@pytest.mark.parametrize("text", [
    "Which BCMs work under Unit Head Uma?",
    "Give me the BCMs under Unit Head Uma.",
    "Which BCMs are below Unit Head Uma?",
    "Who are the BCMs reporting to Unit Head Uma?",
    "Which BCMs are working under Unit Head Uma?",
    "Show the BCMs under Uma",
])
def test_every_transitive_phrasing_reaches_the_same_answer(org, text):
    """Phrasings of one question must not disagree — "which BCMs" planned
    as a breakdown and "give me the BCMs" as a roster, so the same
    question returned a summary or a list of the wrong people depending
    on the verb."""
    resolution, response, _ = _answer(org, text)
    assert resolution.plan.target_level == "bcm", text
    assert _members(response) == ["Bee", "Zed"], text


# ------------------------------------------------------- other levels
def test_zonal_heads_under_a_unit_head(org):
    _, response, _ = _answer(org, "Which zonal heads work under Unit Head Uma?")
    assert _members(response) == ["Zed"]        # Uma is her own, and excluded


def test_the_leaf_population_is_left_to_the_roster_reading(org):
    """"advisors under X" already resolved correctly and has a settled
    definition (test_unit_head_roster, test_team_size). This change adds
    the MANAGER levels, which had none; it must not restate the leaf."""
    for text in ("Which advisors work under BCM Bee?",
                 "How many advisors are under Zonal Head Zed?",
                 "all advisors under Uma"):
        resolution, _, _ = _answer(org, text)
        assert resolution.plan.action != "scoped_reports", text


def test_self_is_never_one_of_its_own_reports(org):
    """Uma is her own BCM for two advisors; counting her among the BCMs
    beneath her makes a manager one of their own reports."""
    _, response, _ = _answer(org, "Which BCMs work under Unit Head Uma?")
    assert "Uma" not in _members(response)


# ------------------------------- what must NOT change (regression guards)
def test_direct_reports_still_means_immediate(org):
    """The word "directly" must keep its meaning: Uma's immediate reports
    are Zonal Heads, not the two BCMs beneath her."""
    resolution, response, _ = _answer(org, "Who reports directly to Uma?")
    assert resolution.plan.action == "direct_reports"
    assert response["data"]["target_level"] == "zonal_head"


def test_directly_with_a_named_target_is_still_the_direct_reading(org):
    """"how many advisors DIRECTLY report to Uma" is 2 (the two Directs),
    not the 6 in her subtree — the transitive scorer must decline on
    "directly" so the immediate reading keeps the sentence.

    (Phrasing taken from test_direct_reports, which pins which wordings
    reach this route; "report directly to" resolves to `lookup` and did
    so before this change too.)"""
    resolution, response, _ = _answer(org, "how many advisors directly report to Uma")
    assert resolution.plan.action == "direct_reports"
    assert response["data"]["count"] == 2


def test_a_question_with_no_target_level_keeps_its_old_reading(org):
    """"Who is under Uma" names no level to enumerate, so it must keep
    whatever it resolved to before — this change adds a reading, it does
    not take one away."""
    resolution, _, _ = _answer(org, "Who is under Uma?")
    assert resolution.plan.action != "scoped_reports"


def test_a_teams_own_reading_is_untouched(org):
    """"Uma's team" names `team`, which is ABOVE unit_head, so it is not a
    descent and must not be read as one."""
    resolution, _, _ = _answer(org, "Give me Uma's team")
    assert resolution.plan.action != "scoped_reports"


def test_a_ranking_under_a_manager_stays_a_ranking(org):
    """"top advisors under Uma by connects" is a leaderboard with a scope
    filter and already worked — naming a measure must keep it there."""
    resolution, _, _ = _answer(org, "top advisors under Uma by connects")
    assert getattr(resolution.plan, "action", None) != "scoped_reports"


def test_a_plain_roster_is_untouched(org):
    resolution, _, _ = _answer(org, "all advisors in AMD")
    assert resolution.plan.action == "roster"


# ------------------------------------------------------- service layer
def test_the_service_refuses_a_target_that_is_not_beneath(org):
    """`team` sits above `unit_head`; asking for it as a descent is not a
    question about this level and must not answer as though it were."""
    assert hierarchy_service.get_scoped_reports(org, "unit_head", "Uma", "team") is None


def test_count_and_members_come_from_one_population(org):
    """The count is len(members) by construction, so the number and the
    list cannot be served from two different scopes."""
    reports = hierarchy_service.get_scoped_reports(org, "unit_head", "Uma", "bcm")
    assert reports["count"] == len(reports["members"]) == 2
