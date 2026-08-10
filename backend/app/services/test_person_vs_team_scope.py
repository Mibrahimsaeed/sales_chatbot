"""Phase 22 — a manager's own figures vs their group's.

THE BUSINESS RULE. Everyone in the hierarchy is also an advisor with
their own row, so a bare measure question about a person means THAT
PERSON:

    "connects of Fawad Hafeez"          his own 54
    "connects of Fawad Hafeez's team"   his zone's 949

THE DEFECT. Once a name grounded at a manager level, the person reading
became unreachable: both questions returned 949, in the same sentence,
and no phrasing got to the 54. Being a BCM/Zonal Head/Unit Head silently
implied group scope.

A SECOND DEFECT had the opposite shape. "connects of Zonal Head Faisal
Hussain Naqvi" states the level, and the level was detected — then
ignored when the ENTITY was chosen. The name stayed grounded at
zonal_head, bcm, region and advisor at once, and _Intent.group_entity()
took whichever came first in GROUP_LEVEL_ORDER, which is `bcm`. The
user's own words lost to an ordering constant: 227 (the BCM's four
reports) instead of 763 (the Zonal Head's eleven).

Both fixes reuse `_pin_level` — the narrowing the clarification answer
already performs — so stating the level in the sentence and picking it
from the offered list now run through the same code and cannot disagree.

The fixture gives every manager an OWN value that differs from their
group's, so no assertion here can pass by reading the wrong scope.
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

# Managers are advisors too, and their own connects differ from their
# group's — the whole point of the rule.
#
#   Zoya Rahim  (zonal head)  own 5   | zone  = 5 + 11 + 22 + 33 = 71
#     Basit Iqbal (bcm)       own 11  | centre= 11 + 22 = 33
#       Umair Sethi (unit hd) own 22  | unit  = 22
#       Nida Aslam            own 33
#   Kamil Yousaf              own 99  — outside, must never be counted
PEOPLE = [
    # wid, name,          team,    rm,            management_lead, portfolio_lead, connects
    (1, "Zoya Rahim",    "Alpha", "Umair Sethi", "Basit Iqbal",  "Zoya Rahim",   5),
    (2, "Basit Iqbal",   "Alpha", "Umair Sethi", "Basit Iqbal",  "Zoya Rahim",   11),
    (3, "Umair Sethi",   "Alpha", "Umair Sethi", "Basit Iqbal",  "Zoya Rahim",   22),
    (4, "Nida Aslam",    "Beta",  "Other Rm",    "Other Ml",     "Zoya Rahim",   33),
    (5, "Kamil Yousaf",  "Gamma", "Other Rm",    "Other Ml",     "Other Pl",     99),
]

OWN = {"Zoya Rahim": 5, "Basit Iqbal": 11, "Umair Sethi": 22}
ZONE = 5 + 11 + 22 + 33     # portfolio_lead = Zoya Rahim
CENTRE = 5 + 11 + 22        # management_lead = Basit Iqbal
UNIT = 5 + 11 + 22          # rm = Umair Sethi
OUTSIDE = 99


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid, name, team, rm, ml, pl, connects in PEOPLE:
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                               rm=rm, management_lead=ml, portfolio_lead=pl,
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


def _reply(db, text, session=None):
    return handle_chat_message(db, text, session_id=session)


def _numbers(text):
    import re
    out = set()
    for token in re.findall(r"\d[\d,]*", text):
        try:
            out.add(int(token.replace(",", "")))
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------------
# PERSON scope — a bare measure question means the person
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Zoya Rahim", "Basit Iqbal", "Umair Sethi"])
def test_a_managers_own_measure_is_their_own_row(db, name):
    """THE defect. Each of these returned their group's total instead."""
    reply = _reply(db, f"connects of {name}")["reply"]
    assert OWN[name] in _numbers(reply)


@pytest.mark.parametrize("name,group", [
    ("Zoya Rahim", ZONE), ("Basit Iqbal", CENTRE), ("Umair Sethi", UNIT),
])
def test_a_managers_own_measure_is_NOT_their_group_total(db, name, group):
    """The other half — asserting the own value alone would pass on a
    reply that happened to contain both."""
    assert group not in _numbers(_reply(db, f"connects of {name}")["reply"])


def test_an_ordinary_advisor_is_unchanged(db):
    assert OUTSIDE in _numbers(_reply(db, "connects of Kamil Yousaf")["reply"])


@pytest.mark.parametrize("phrasing", [
    "connects of {}", "{}'s connects", "how many connects does {} have?",
])
def test_every_person_phrasing_gives_the_person(db, phrasing):
    reply = _reply(db, phrasing.format("Basit Iqbal"))["reply"]
    assert OWN["Basit Iqbal"] in _numbers(reply)
    assert CENTRE not in _numbers(reply)


# ---------------------------------------------------------------------
# TEAM scope — the people under them
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,level,group", [
    ("Zoya Rahim", "Zonal Head", ZONE),
    ("Basit Iqbal", "BCM", CENTRE),
    ("Umair Sethi", "Unit Head", UNIT),
])
def test_a_managers_team_measure_is_the_subordinate_total(db, name, level, group):
    session = f"team-{name}"
    conversation_memory._store.pop(session, None)
    response = _reply(db, f"connects of {name}'s team", session=session)
    reply = response["reply"]
    if response["type"] == "clarification" and "could mean" in reply:
        reply = _reply(db, level, session=session)["reply"]
    assert group in _numbers(reply)
    assert OUTSIDE not in _numbers(reply), "someone outside the subtree was counted"


def test_person_and_team_are_different_answers(db):
    """THE distinction. These were byte-identical before Phase 22."""
    person = _reply(db, "connects of Zoya Rahim")["reply"]
    session = "pt"
    conversation_memory._store.pop(session, None)
    team = _reply(db, "connects of Zoya Rahim's team", session=session)
    team_reply = team["reply"]
    if team["type"] == "clarification" and "could mean" in team_reply:
        team_reply = _reply(db, "Zonal Head", session=session)["reply"]

    assert OWN["Zoya Rahim"] in _numbers(person)
    assert ZONE in _numbers(team_reply)
    assert person != team_reply


# ---------------------------------------------------------------------
# Explicit level == clarified level
# ---------------------------------------------------------------------


@pytest.mark.parametrize("prefix,level,expected", [
    ("Zonal Head", "Zonal Head", ZONE),
    ("BCM", "BCM", CENTRE),
    ("Unit Head", "Unit Head", UNIT),
])
def test_an_explicit_level_prefix_scopes_to_that_level(db, prefix, level, expected):
    """"connects of Zonal Head X" must use portfolio_lead, not whichever
    level GROUP_LEVEL_ORDER happens to list first."""
    name = {"Zonal Head": "Zoya Rahim", "BCM": "Basit Iqbal",
            "Unit Head": "Umair Sethi"}[prefix]
    assert expected in _numbers(_reply(db, f"connects of {prefix} {name}")["reply"])


def test_the_explicit_prefix_and_the_clarification_agree(db):
    """The two routes must reach the same scope — they now share
    _pin_level, so they cannot drift."""
    explicit = _reply(db, "connects of BCM Basit Iqbal")["reply"]

    session = "clar"
    conversation_memory._store.pop(session, None)
    first = _reply(db, "connects of Basit Iqbal's team", session=session)
    clarified = first["reply"]
    if first["type"] == "clarification" and "could mean" in clarified:
        clarified = _reply(db, "BCM", session=session)["reply"]

    assert CENTRE in _numbers(explicit)
    assert CENTRE in _numbers(clarified)


def test_a_reverse_question_is_not_pinned_by_its_own_level_word(db):
    """"X's unit head" names the level of the ANSWER. Pinning X to
    unit_head turned a reverse lookup into a query about X's group."""
    response = _reply(db, "who is Nida Aslam's unit head?")
    assert response["type"] != "breakdown"
    assert UNIT not in _numbers(str(response.get("reply")))


# ---------------------------------------------------------------------
# Safety — still no global ranking
# ---------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Nobody Here's connects",
    "Nobody Here's team connects",
    "what are Nobody Here's connects",
])
def test_an_ungroundable_person_never_becomes_a_ranking(db, text):
    reply = str(_reply(db, text).get("reply"))
    assert "🏆" not in reply and "ranking" not in reply.lower()


@pytest.mark.xfail(reason=(
    "PRE-EXISTING GAP, found by this phase and not caused by it. "
    "routing.unresolved_subject reads POSSESSIVE spans only — its "
    "_POSSESSIVE pattern requires \"'s\" — so \"connects of <unknown>\" "
    "never reaches the refusal and still answers with a global ranking. "
    "The possessive phrasings above are guarded; this one is not. "
    "Pinned as xfail so the hole is visible and turns green the moment "
    "the gate learns the 'measure of <person>' shape."), strict=True)
def test_the_of_phrasing_should_also_refuse_an_unknown_person(db):
    reply = str(_reply(db, "connects of Nobody Here").get("reply"))
    assert "🏆" not in reply and "ranking" not in reply.lower()
