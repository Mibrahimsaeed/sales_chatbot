"""Phase 37 — four ways to ask how big someone's team is, one answer.

    "team size of Haseeb Arslan"        -> "could mean the Unit Head or
                                            the Zonal Head or the BCM or
                                            the Advisor?"
    "how many advisors are under him"   -> his own profile, "0 MTD connects"
    "Haseeb Arslan's team size"         -> 75

THE COUNTING WAS NEVER WRONG. For fifteen real managers across three
levels, the `advisors` column count, `aggregation.headcount` and
`hierarchy_service.get_level_roster` agreed exactly. Not one of those was
reached by three of the four phrasings.

ONE WORD DID IT. `detect_level("team size of X")` returns `team`,
because "team" is a level keyword and "team size" contains it. `team` is
not one of the person's ROLE levels, so `_pin_stated_level` declined
outright and the pipeline asked which of four readings of one man was
meant. The Chairman — who holds a single role and so has no ambiguity to
resolve — answered correctly all along, which is what isolated the cause
to ambiguity resolution rather than to counting.

So "team" in "team size" names WHAT IS MEASURED, and "advisors" in
"advisors under X" names WHAT IS RETURNED. Neither describes the
person's own level, and when no word does, their own hierarchy answers
it — `_highest_role`, the same ranking the possessive form has used
since Phase 28, which is why all four phrasings now agree.

A GENUINE TEAM IS SAFE because the guard only fires when the value is
NOT itself grounded at `team`: a person who shares a name with a real
team keeps both readings and keeps the clarification, and "connects of
Blue Area" grounds at team alone, raises no ambiguity, and never reaches
this code.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, aggregation, conversation_memory, entity_extractor,
    narrative, nlu_pipeline, semantic_parser,
)
from app.services import hierarchy_service
from app.services.chat_service import handle_chat_message

# Uzair Malik is Unit Head, Zonal Head and BCM; Sadia Noor is Zonal Head
# and BCM; Imran Latif is a BCM only. Each level's scope is a different
# size, so a wrong level shows up as a wrong number rather than passing
# by coincidence.
#
# wid, name,            rm (unit),      portfolio_lead (zonal), management_lead (bcm)
PEOPLE = [
    (1, "Uzair Malik",   "Uzair Malik",  "Uzair Malik",  "Uzair Malik"),
    (2, "Sadia Noor",    "Uzair Malik",  "Uzair Malik",  "Uzair Malik"),
    (3, "Kamil Raza",    "Uzair Malik",  "Uzair Malik",  "Imran Latif"),
    (4, "Nida Perveen",  "Uzair Malik",  "Sadia Noor",   "Imran Latif"),
    (5, "Imran Latif",   "Uzair Malik",  "Sadia Noor",   "Sadia Noor"),
    (6, "Hamza Iqbal",   "Other Unit",   "Sadia Noor",   "Sadia Noor"),
    (7, "Rida Aslam",    "Other Unit",   "Other Zonal",  "Imran Latif"),
    (8, "Plain Advisor", "Other Unit",   "Other Zonal",  "Other Bcm"),
]

# Uzair Malik : unit 5 (wids 1-5) | zonal 3 (1,2,3) | bcm 2 (1,2)
UZAIR_UNIT = 5
# Sadia Noor  : zonal 3 (4,5,6)   | bcm 2 (5,6)
SADIA_ZONAL = 3
# Imran Latif : bcm 3 (3,4,7)
IMRAN_BCM = 3

PHRASINGS = [
    "team size of {name}",
    "{name}'s team size",
    "how many people are under {name}",
    "how many advisors are under {name}",
]

MANAGERS = [
    ("Uzair Malik", "unit_head", UZAIR_UNIT),
    ("Sadia Noor", "zonal_head", SADIA_ZONAL),
    ("Imran Latif", "bcm", IMRAN_BCM),
]


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid, name, rm, pl, ml in PEOPLE:
        db_session.add(Advisor(wid=wid, name=name, team="Alpha", company="Graana",
                               rm=rm, portfolio_lead=pl, management_lead=ml,
                               in_master_sheet=True))
        db_session.add(Calls(wid=wid, connects_mtd=wid * 10, connects_daily=0,
                             answered_calls_mtd=wid, answered_calls_daily=0))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=wid * 10,
                                   mtd_followup_connect=0, mtd_cr=1))
        db_session.add(Pipeline(wid=wid, pipeline=wid, overdue=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=wid))
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


def _size(db, text):
    """The team size a reply reports, or the response type when it
    reported none — so a clarification fails loudly rather than as 0.

    TWO SHAPES, ONE NUMBER. A single-subject team-size question answers
    with a sentence of its own ("Ahmed has a team size of 74."), while
    "team of X" still produces the breakdown wording. Both are read here
    so these tests keep measuring the NUMBER and the level it was counted
    at — what they were written to pin — rather than the phrasing.
    """
    import re

    response = _ask(db, text)
    reply = str(response.get("reply") or "").strip()
    for pattern in (r"has a team size of (\d[\d,]*)", r"has (\d[\d,]*) advisors"):
        match = re.search(pattern, reply)
        if match:
            return int(match.group(1).replace(",", ""))
    return response["type"]


# ---------------------------------------------------------------------
# A/B. Every phrasing, every level, one number
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,_level,expected", MANAGERS)
@pytest.mark.parametrize("phrasing", PHRASINGS)
def test_every_phrasing_returns_the_team_size(db, phrasing, name, _level, expected):
    assert _size(db, phrasing.format(name=name)) == expected


@pytest.mark.parametrize("name,_level,_expected", MANAGERS)
def test_the_four_phrasings_agree_with_each_other(db, name, _level, _expected):
    """Stated as one assertion so a future change cannot fix one wording
    and quietly leave another asking."""
    answers = {p: _size(db, p.format(name=name)) for p in PHRASINGS}
    assert len(set(answers.values())) == 1, answers


@pytest.mark.parametrize("name,level,expected", MANAGERS)
def test_the_count_is_the_hierarchys_own(db, name, level, expected):
    """No second definition of "under this person": the number a team-size
    question reports is the one headcount and the roster already agree
    on."""
    assert aggregation.headcount(db, level, name) == expected
    assert hierarchy_service.get_level_roster(db, level, name)["count"] == expected


# ---------------------------------------------------------------------
# C. Multi-role people answer at their highest level
# ---------------------------------------------------------------------


def test_a_bcm_who_is_also_a_zonal_head_answers_at_zonal_scope(db):
    """Sadia Noor leads 2 as a BCM and 3 as a Zonal Head."""
    assert _size(db, "team size of Sadia Noor") == SADIA_ZONAL
    assert aggregation.headcount(db, "bcm", "Sadia Noor") == 2


def test_a_zonal_head_who_is_also_a_unit_head_answers_at_unit_scope(db):
    assert _size(db, "team size of Uzair Malik") == UZAIR_UNIT
    assert aggregation.headcount(db, "zonal_head", "Uzair Malik") == 3


@pytest.mark.parametrize("role", ["BCM", "Zonal Head"])
def test_a_stated_junior_role_is_still_corrected_upward(db, role):
    """Phase 31: naming a role points at the person; their own hierarchy
    decides the scope."""
    assert _size(db, f"team size of {role} Uzair Malik") == UZAIR_UNIT


def test_a_single_role_manager_is_unchanged(db):
    """Imran Latif is a BCM and nothing more — no ambiguity to resolve,
    and this path is not even consulted for him."""
    assert _size(db, "team size of Imran Latif") == IMRAN_BCM


# ---------------------------------------------------------------------
# D. A person who leads nobody
# ---------------------------------------------------------------------


def test_someone_with_no_subordinates_does_not_ask_a_question(db):
    """A plain advisor grounds at one level, so there is nothing to
    clarify. Note a manager cannot have zero subordinates by
    construction: a name reaches `bcm` only because some advisor's row
    names them there."""
    response = _ask(db, "team size of Plain Advisor")
    assert response["type"] != "clarification"


def test_a_manager_scope_with_nobody_in_it_counts_zero(db):
    """The counting layer's own answer for an empty scope, pinned so a
    genuine zero stays distinguishable from a misroute."""
    assert aggregation.headcount(db, "bcm", "Nobody At All") == 0


# ---------------------------------------------------------------------
# E. Everything that already worked
# ---------------------------------------------------------------------


def test_a_persons_own_metric_is_unchanged(db):
    """Phase 22 RULE 1 — "connects of X" is about X, not about his team."""
    response = _ask(db, "connects of Uzair Malik")
    assert response["type"] == "advisor_metric"
    assert "10" in response["reply"]


def test_a_team_metric_aggregates_exactly_that_team(db):
    """The distinction this phase turns on: team SIZE counts the people,
    a team METRIC sums over the same people."""
    response = _ask(db, "connects of Uzair Malik's team")
    assert sum(m["value"] or 0 for m in response["members"]) == sum(
        wid * 10 for wid in range(1, UZAIR_UNIT + 1))
    assert len(response["members"]) == UZAIR_UNIT


def test_a_genuine_team_entity_is_still_a_team(db):
    """"Alpha" is a real team. The guard cannot touch it — it grounds at
    `team` alone, so no ambiguity reaches this code at all."""
    response = _ask(db, "connects of Alpha")
    assert response["type"] != "clarification"
    assert "Alpha" in response["reply"]


def test_a_name_that_is_also_a_team_still_asks(db, db_session):
    """The safety property in its own right: when the value IS grounded
    at `team`, `team` is one of the readings on offer and choosing for
    the user would be a guess."""
    assert nlu_pipeline._measures_the_group(
        "team size of ambiguous", "team", ["team", "bcm", "advisor"]) is False
    assert nlu_pipeline._measures_the_group(
        "team size of a person", "team", ["bcm", "advisor"]) is True


def test_a_threshold_is_not_read_as_a_relation():
    """`under` is also a comparator ("less than", "below", "under"), so a
    following number disqualifies the relation reading."""
    assert nlu_pipeline._measures_the_group(
        "advisors under 50 connects", None, ["bcm", "advisor"]) is False
    assert nlu_pipeline._measures_the_group(
        "people under Uzair Malik", None, ["bcm", "advisor"]) is True


def test_the_roster_phrasing_still_returns_a_roster(db):
    """Phase 30, unchanged."""
    response = _ask(db, "show all advisors under Unit Head Uzair Malik")
    assert response["type"] == "roster"
    assert response["data"]["count"] == UZAIR_UNIT


def test_team_size_and_the_roster_describe_the_same_people(db):
    """The whole point: one membership definition behind both."""
    assert _size(db, "team size of Uzair Malik") == \
        _ask(db, "show all advisors under Unit Head Uzair Malik")["data"]["count"]
