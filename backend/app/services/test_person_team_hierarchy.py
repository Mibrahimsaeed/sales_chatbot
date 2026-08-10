"""Phase 20 — "<person>'s team" means the people under that person.

THREE DIFFERENT QUESTIONS share the word "team", and the system has to
tell them apart:

    "Ali Asghar's pipeline"          his own hierarchy-level figure
    "Ali Asghar's team's pipeline"   the people who report to him
    "Beverly Center's pipeline"      a named team, no person involved

TWO DEFECTS made the middle one wrong, and they failed in opposite
directions.

1. THE MEASURE WAS DROPPED. `_score_hierarchy` built
   `QueryPlan(action="breakdown", level=..., entity_value=...)` and never
   set `metric`. The SCOPE was already right — the plan named the manager
   and the level, and every metric path filters on exactly that pair — so
   "X's team pipeline" and "X's team's connects" returned the same canned
   breakdown card (advisor count, connects, cleared) and neither answered
   what was asked. The relationship was understood; the question was not.

2. THE PERSON WAS DROPPED. `routing.unresolved_subject` skipped its
   refusal for any text containing a relation ("X's team"), on the
   documented assumption that the traversal is grounded downstream. When
   the source grounded to NOTHING there was no downstream: the person
   vanished, the word "team" remained, and the planner built a perfectly
   valid UNFILTERED team leaderboard. "Omer Sandhu (Virtual)'s team
   pipeline" answered with a ranking of all nine teams.

The second is the dangerous one and is why the safety tests below assert
on the ABSENCE of a ranking, not just on the presence of a clarification:
a wrong confident answer and a refusal are both "not the number", and
only one of them is acceptable.

Every value here is asserted against a SUM computed in the test from the
fixture, so a test cannot pass by reading the wrong scope.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, conversation_memory, entity_extractor, narrative,
    semantic_parser,
)
from app.services.chat_service import handle_chat_message

# Managers are deliberately NOT advisor rows, so each name grounds at
# exactly ONE level and the queries below need no disambiguation turn.
# That mirrors the real sheet, where most managers appear only in the
# manager columns.
#
#   ZH "Zara Iqbal"  -> portfolio_lead
#     BCM "Bilal Khan" -> management_lead
#       UH "Umar Ahmed"   -> rm
#
# wid: (name, team, rm, management_lead, portfolio_lead, pipeline, connects)
PEOPLE = [
    (1, "Ayesha One",   "Alpha", "Umar Ahmed",  "Bilal Khan", "Zara Iqbal", 100, 10),
    (2, "Bilal Two",    "Alpha", "Umar Ahmed",  "Bilal Khan", "Zara Iqbal", 200, 20),
    (3, "Chand Three",  "Beta",  "Other Ahmed", "Bilal Khan", "Zara Iqbal", 400, 40),
    # under the zonal head only — not under Bilal Khan or Umar Ahmed
    (4, "Dania Four",   "Beta",  "Other Ahmed", "Other Khan", "Zara Iqbal", 800, 80),
    # outside the whole subtree; must never be counted
    (5, "Emaan Five",   "Gamma", "Other Ahmed", "Other Khan", "Other Iqbal", 1600, 160),
]

UNIT_HEAD_PIPELINE = 100 + 200            # wids 1,2
BCM_PIPELINE = 100 + 200 + 400            # wids 1,2,3
ZONAL_PIPELINE = 100 + 200 + 400 + 800    # wids 1,2,3,4
UNIT_HEAD_CONNECTS = 10 + 20


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid, name, team, rm, ml, pl, pipe, connects in PEOPLE:
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                               rm=rm, management_lead=ml, portfolio_lead=pl,
                               in_master_sheet=True))
        db_session.add(Pipeline(wid=wid, pipeline=pipe, overdue=0))
        db_session.add(Calls(wid=wid, connects_mtd=connects, connects_daily=0,
                             answered_calls_mtd=connects, answered_calls_daily=0))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=connects,
                                   mtd_followup_connect=0, mtd_cr=1))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=1000, cleared=pipe))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


def _reply(db, text):
    return handle_chat_message(db, text, session_id=None)


def _numbers(text):
    import re
    out = set()
    for token in re.findall(r"\d[\d,]*", text):
        try:
            out.add(int(token.replace(",", "")))
        except ValueError:
            pass
    return out


def _is_ranking(reply):
    """A global leaderboard, which a person's-team question must never
    produce — that is the wrong-confident-answer failure."""
    return "🏆" in reply or "ranking" in reply.lower()


# ---------------------------------------------------------------------
# The relationship, at each manager level
# ---------------------------------------------------------------------


def test_a_unit_heads_team_aggregates_the_advisors_under_them(db):
    reply = _reply(db, "What is Umar Ahmed's team pipeline?")["reply"]
    assert UNIT_HEAD_PIPELINE in _numbers(reply)
    assert 1600 not in _numbers(reply), "an advisor outside the subtree was counted"


def test_a_bcms_team_aggregates_the_advisors_under_them(db):
    reply = _reply(db, "What is Bilal Khan's team pipeline?")["reply"]
    assert BCM_PIPELINE in _numbers(reply)
    assert 1600 not in _numbers(reply)


def test_a_zonal_heads_team_aggregates_the_advisors_under_them(db):
    reply = _reply(db, "What is Zara Iqbal's team pipeline?")["reply"]
    assert ZONAL_PIPELINE in _numbers(reply)
    assert 1600 not in _numbers(reply)


def test_the_three_levels_give_three_different_scopes(db):
    """The nesting is real: each level up adds people. Equal answers
    would mean the level was ignored and one scope served all three."""
    unit = _numbers(_reply(db, "What is Umar Ahmed's team pipeline?")["reply"])
    bcm = _numbers(_reply(db, "What is Bilal Khan's team pipeline?")["reply"])
    zonal = _numbers(_reply(db, "What is Zara Iqbal's team pipeline?")["reply"])
    assert UNIT_HEAD_PIPELINE in unit
    assert BCM_PIPELINE in bcm and BCM_PIPELINE not in unit
    assert ZONAL_PIPELINE in zonal and ZONAL_PIPELINE not in bcm


# ---------------------------------------------------------------------
# The measure is carried, whatever the wording
# ---------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "What is Umar Ahmed's team pipeline?",
    "What is the pipeline of Umar Ahmed's team?",
    "What is Umar Ahmed's team's pipeline?",
])
def test_every_wording_of_the_relationship_answers_the_same(db, text):
    assert UNIT_HEAD_PIPELINE in _numbers(_reply(db, text)["reply"])


def test_a_different_measure_gives_a_different_answer(db):
    """THE dropped-metric defect. Both of these used to return the same
    breakdown card, so the measure the user named changed nothing."""
    pipeline = _numbers(_reply(db, "What is Umar Ahmed's team pipeline?")["reply"])
    connects = _numbers(_reply(db, "What are Umar Ahmed's team's connects?")["reply"])
    assert UNIT_HEAD_PIPELINE in pipeline
    assert UNIT_HEAD_CONNECTS in connects
    assert pipeline != connects


def test_the_reply_discloses_which_manager_it_scoped_to(db):
    """A group figure under a person's name is unreadable without the
    scope: 300 could be his own or his team's."""
    reply = _reply(db, "What is Umar Ahmed's team pipeline?")["reply"]
    assert "Umar Ahmed" in reply


# ---------------------------------------------------------------------
# Safety — a named person must never become a global ranking
# ---------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "What is Nobody Here's team pipeline?",
    "What are Nobody Here's team's connects?",
    "What is the pipeline of Nobody Here's team?",
    "What is Nobody Here's pipeline?",
])
def test_an_ungroundable_person_refuses_instead_of_ranking(db, text):
    """THE safety invariant. Asserted on the ABSENCE of a ranking as well
    as the presence of a clarification — a global leaderboard is a
    confident answer to a question nobody asked, which is worse than
    saying no."""
    response = _reply(db, text)
    reply = response["reply"]
    assert not _is_ranking(reply), "a named person's query became a global ranking"
    assert response["type"] == "clarification"
    assert "Nobody Here" in reply


def test_a_name_with_parentheses_is_still_recognised_as_a_person(db):
    """Real master-sheet names carry qualifiers — "Omer Sandhu
    (Virtual)". The possessive pattern stopped at the bracket, so the
    refusal was skipped and these queries fell through to a ranking."""
    response = _reply(db, "What is Omer Sandhu (Virtual)'s team pipeline?")
    assert not _is_ranking(response["reply"])
    assert response["type"] == "clarification"
    assert "Omer Sandhu (Virtual)" in response["reply"]


def test_no_named_person_query_can_reach_a_team_leaderboard(db):
    """The invariant across every wording at once."""
    for name in ("Nobody Here", "Omer Sandhu (Virtual)", "Zzz Qqq"):
        for form in ("{}'s team pipeline", "the pipeline of {}'s team",
                     "{}'s team's connects"):
            reply = _reply(db, f"What is {form.format(name)}?")["reply"]
            assert not _is_ranking(reply), f"{name} / {form}"


# ---------------------------------------------------------------------
# Everything else must be unchanged
# ---------------------------------------------------------------------


def test_a_named_team_still_resolves_to_that_team(db):
    """Shape C — no person involved, and this path was already correct."""
    reply = _reply(db, "What is Alpha's pipeline?")["reply"]
    assert (100 + 200) in _numbers(reply)


def test_a_named_team_with_a_relation_word_is_not_a_manager_query(db):
    reply = _reply(db, "What is Alpha's team pipeline?")["reply"]
    assert (100 + 200) in _numbers(reply)


@pytest.mark.parametrize("text,expected", [
    ("What is Ayesha One's pipeline?", 100),
    ("What is Chand Three's pipeline?", 400),
])
def test_an_ordinary_persons_own_metric_is_unchanged(db, text, expected):
    """Shape A. The person is an advisor, so this is their own row —
    never a group roll-up."""
    assert expected in _numbers(_reply(db, text)["reply"])


def test_an_advisors_own_metric_is_not_their_teams_total(db):
    """The distinction that makes shape A different from shape B."""
    own = _numbers(_reply(db, "What is Ayesha One's pipeline?")["reply"])
    assert 100 in own
    assert (100 + 200) not in own


def test_a_manager_with_no_measure_still_gets_the_breakdown(db):
    """"X's team" naming no measure keeps the roster/breakdown answer —
    the metric branch is additive, not a replacement."""
    response = _reply(db, "Show me Umar Ahmed's team")
    assert not _is_ranking(str(response.get("reply")))
    assert "Umar Ahmed" in str(response.get("reply"))
