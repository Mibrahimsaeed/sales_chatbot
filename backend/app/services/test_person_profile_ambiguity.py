"""Phase 38 — being a BCM is not a second person to disambiguate.

    "who is Aamir Ali Qureshi?"
    -> "'Aamir Ali Qureshi' could mean the Unit Head or the Zonal Head or
        the BCM or the Advisor — which did you mean?"

Four readings of one man, offered for a sentence that plainly means the
man. It bit a BCM who is nothing else too: "details of Abdul Qadir"
asked "BCM or Advisor?" of someone whose highest role is not in doubt.

THE WORDING DECIDED IT, NOT THE PERSON. The ambiguity branch had exactly
two ways not to ask, and each required the sentence to name something:
`_asks_about_the_person` wanted a MEASURE, the group branch wanted a
RELATION or a group word. A question naming neither — "details of X",
"who is X?", a bare name — matched no branch and fell through to the
question. Same person, six phrasings, three answers and three
clarifications:

    connects of X     -> his own figure      (measure named)
    performance of X  -> his profile         (measure named)
    team size of X    -> Unit Head, 14       (group named)
    team of X         -> Unit Head, 14       (group named)
    details of X      -> CLARIFY
    who is X?         -> CLARIFY
    X                 -> CLARIFY

A SINGLE-ROLE ADVISOR never saw any of it, because no ambiguity is
created for them — which is what isolates the cause to ambiguity
handling rather than to anyone's data. `_highest_role` was correct
throughout and was simply never consulted on this path.

`advisor` is the answer rather than the senior role because these words
ask who someone IS, and that is exactly what the same sentence returns
for a single-role advisor.

THE CLARIFICATION IS NOT GONE. A name that is also a TEAM or a COMPANY
is two different entities sharing a spelling, no ranking settles it, and
the question is still the honest reply — pinned below and in the golden
corpus. Two different PEOPLE sharing a name is a different mechanism
again (`set_pending_person`), and also still asks.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, conversation_memory, entity_extractor, narrative,
    nlu_pipeline, semantic_parser,
)
from app.services.chat_service import handle_chat_message

# wid, name,          rm (unit),      portfolio_lead, management_lead, connects
PEOPLE = [
    # Wears all three hats.
    (1, "Owais Tariq",   "Owais Tariq",  "Owais Tariq",  "Owais Tariq",   10),
    (2, "Rida Kamal",    "Owais Tariq",  "Owais Tariq",  "Owais Tariq",   20),
    # BCM + Zonal Head.
    (3, "Saira Bhatti",  "Owais Tariq",  "Saira Bhatti", "Saira Bhatti",  30),
    (4, "Junaid Aziz",   "Owais Tariq",  "Saira Bhatti", "Other Bcm",     40),
    # BCM only.
    (5, "Faiz Ahmed",    "Owais Tariq",  "Other Zonal",  "Faiz Ahmed",    50),
    (6, "Noor Zahra",    "Owais Tariq",  "Other Zonal",  "Faiz Ahmed",    60),
    # No role at all — the control.
    (7, "Plain Person",  "Owais Tariq",  "Other Zonal",  "Other Bcm",     70),
]

MULTI_ROLE = [
    ("Owais Tariq", "unit_head"),     # BCM + Zonal Head + Unit Head
    ("Saira Bhatti", "zonal_head"),   # BCM + Zonal Head
    ("Faiz Ahmed", "bcm"),            # BCM only
]

PROFILE_QUESTIONS = ["details of {n}", "who is {n}?", "tell me about {n}", "{n}"]


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid, name, rm, pl, ml, connects in PEOPLE:
        db_session.add(Advisor(wid=wid, name=name, team="Alpha", company="Graana",
                               rm=rm, portfolio_lead=pl, management_lead=ml,
                               in_master_sheet=True))
        db_session.add(Calls(wid=wid, connects_mtd=connects, connects_daily=0,
                             answered_calls_mtd=connects, answered_calls_daily=0))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=connects,
                                   mtd_followup_connect=0, mtd_cr=1))
        db_session.add(Pipeline(wid=wid, pipeline=connects, overdue=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=connects))
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


def _ask(db, text):
    conversation_memory._store.clear()
    return handle_chat_message(db, text, session_id=None)


def _asks_which_role(response):
    return response["type"] == "clarification" and "could mean" in str(response["reply"])


# ---------------------------------------------------------------------
# A profile question about a person is answered, not questioned
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,_highest", MULTI_ROLE)
@pytest.mark.parametrize("question", PROFILE_QUESTIONS)
def test_a_profile_question_never_asks_which_role(db, question, name, _highest):
    assert not _asks_which_role(_ask(db, question.format(n=name)))


@pytest.mark.parametrize("name,_highest", MULTI_ROLE)
@pytest.mark.parametrize("question", PROFILE_QUESTIONS)
def test_a_profile_question_answers_about_the_person(db, question, name, _highest):
    """The person, not the group they lead — the same answer a single-role
    advisor gets for these words."""
    response = _ask(db, question.format(n=name))
    assert response["type"] == "advisor"
    assert name in response["reply"]


@pytest.mark.parametrize("name,_highest", MULTI_ROLE)
def test_the_profile_is_their_own_figures(db, name, _highest):
    """Owais Tariq's own 10, not his unit's 280 — the person reading has
    to stay reachable (Phase 22 RULE 1)."""
    own = {n: c for _w, n, _r, _p, _m, c in PEOPLE}[name]
    assert str(own) in _ask(db, f"details of {name}")["reply"]


def test_a_single_role_person_is_unchanged(db):
    """No ambiguity is created for them, so this path is not consulted —
    their answers were already right and must stay byte-identical."""
    response = _ask(db, "who is Plain Person?")
    assert response["type"] == "advisor"
    assert "70" in response["reply"]


def test_nothing_is_left_pending(db):
    """A stored question with none on screen would eat the next turn."""
    conversation_memory._store.pop("pend", None)
    handle_chat_message(db, "who is Owais Tariq?", session_id="pend")
    assert conversation_memory.get_pending_level("pend") is None


# ---------------------------------------------------------------------
# Measures and groups keep their own answers
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,_highest", MULTI_ROLE)
def test_a_measure_question_still_returns_their_own_metric(db, name, _highest):
    own = {n: c for _w, n, _r, _p, _m, c in PEOPLE}[name]
    response = _ask(db, f"connects of {name}")
    assert response["type"] == "advisor_metric"
    assert str(own) in response["reply"]


@pytest.mark.parametrize("name,highest,expected", [
    ("Owais Tariq", "unit_head", 7),
    ("Saira Bhatti", "zonal_head", 2),
    ("Faiz Ahmed", "bcm", 2),
])
def test_a_team_question_still_uses_the_highest_role(db, name, highest, expected):
    """Phases 28/37, untouched: "team size of X" and "X's team" resolve to
    the senior-most role the person holds."""
    from app.llm import aggregation

    assert aggregation.headcount(db, highest, name) == expected
    for question in (f"team size of {name}", f"{name}'s team size"):
        assert f"{expected} advisors" in _ask(db, question)["reply"]


def test_the_person_and_the_team_remain_different_answers(db):
    """The distinction the whole branch exists to keep."""
    person = _ask(db, "details of Owais Tariq")["reply"]
    team = _ask(db, "team size of Owais Tariq")["reply"]
    assert "7 advisors" not in person
    assert "7 advisors" in team


# ---------------------------------------------------------------------
# Genuine ambiguity still asks
# ---------------------------------------------------------------------


def test_a_name_that_is_also_a_team_still_asks(db, db_session):
    """A BCM who is also a TEAM NAME is two different entities sharing a
    spelling. No role ranking settles that, so the question stands — this
    is what keeps the resolutions above honest."""
    db_session.add(Advisor(wid=8, name="Extra Person", team="Faiz Ahmed",
                           company="Graana", rm="Owais Tariq",
                           portfolio_lead="Other Zonal", management_lead="Faiz Ahmed",
                           in_master_sheet=True))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()

    assert _asks_which_role(_ask(db, "who is Faiz Ahmed?"))


def test_the_guard_is_the_highest_role_returning_nothing():
    """Stated directly: a team/company reading makes the ranking silent,
    and that silence is what preserves the clarification."""
    assert nlu_pipeline._highest_role(["bcm", "advisor"]) == "bcm"
    assert nlu_pipeline._highest_role(["team", "bcm", "advisor"]) is None
    assert nlu_pipeline._highest_role(["company", "bcm"]) is None


def test_two_different_people_sharing_a_name_still_ask(db, db_session):
    """A different mechanism (set_pending_person) and a genuinely
    unresolvable ambiguity — hierarchy cannot say which person was meant."""
    db_session.add(Advisor(wid=9, name="Rida Kamal", team="Beta", company="Graana",
                           rm="Other Unit", portfolio_lead="Other Zonal",
                           management_lead="Other Bcm", in_master_sheet=True))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()

    response = _ask(db, "who is Rida Kamal?")
    assert response["type"] == "clarification"
    assert "multiple advisors" in response["reply"].lower()
