"""Phase 28 — one person, four hats, no question to ask.

"connects of Haseeb Arslan's team" answered `'Haseeb Arslan' could mean
the Unit Head or the Zonal Head or the BCM or the Advisor or the Region —
which did you mean?`. Every option was the same man. He grounds at four
levels BECAUSE he is a Unit Head: his 89 advisors name him in `rm`, and
the ones directly under him name him in `portfolio_lead` and
`management_lead` too. The clarification offered four readings of one
person, so no answer was wrong and none was informative.

THE ROLES ARE ALREADY IN HAND. A name reaches `unit_head` only via some
advisor's `rm`, `zonal_head` only via `portfolio_lead`, `bcm` only via
`management_lead` — grounding IS the hierarchy relationship. So the
levels the ambiguity already carries say which roles the person holds,
and picking the senior one is a read of `hierarchy.CHAIN`, whose order
(unit_head > zonal_head > bcm > advisor) is exactly the required
priority. No traversal, no second resolver, no reordering.

WHICH ROLE WINS DEPENDS ON WHAT WAS ASKED, and only two answers exist.
A turn asking for the GROUP takes the senior role — that is the scope
the person leads. A turn asking about the PERSON keeps their own figure,
which is Phase 22's rule and is left strictly alone here.

The fixture gives every reading a DIFFERENT total on purpose. Adeel Raza
is the sharp case: his BCM scope (34) is LARGER than his Zonal Head
scope (28), so a test asserting 28 fails both if the fix does nothing
and if it picks by size rather than by rank — and 34 is precisely what
the old GROUP_LEVEL_ORDER-first behaviour would have returned.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, conversation_memory, entity_extractor, hierarchy,
    narrative, nlu_pipeline, semantic_parser,
)
from app.llm.nlu_pipeline import _highest_role
from app.services.chat_service import handle_chat_message

# wid, name,        rm (unit head), portfolio_lead (zonal), management_lead (bcm), connects
PEOPLE = [
    # Tahir Malik wears all four hats, each over a different population.
    (1,  "Tahir Malik", "Tahir Malik", "Tahir Malik", "Tahir Malik",  7),
    (2,  "Sana Riaz",   "Tahir Malik", "Tahir Malik", "Tahir Malik", 11),
    (3,  "Bilal Khan",  "Tahir Malik", "Tahir Malik", "Other Bcm",   13),
    (4,  "Nida Aslam",  "Tahir Malik", "Other Zh",    "Other Bcm",   17),
    # Adeel Raza tops out at Zonal Head — and his BCM scope is bigger.
    (5,  "Adeel Raza",  "Other Uh",    "Adeel Raza",  "Adeel Raza",   5),
    (6,  "Kiran Shah",  "Other Uh",    "Adeel Raza",  "Other Bcm",   23),
    (7,  "Omar Faruq",  "Other Uh",    "Other Zh",    "Adeel Raza",  29),
    # Hina Sethi tops out at BCM.
    (8,  "Hina Sethi",  "Other Uh",    "Other Zh",    "Hina Sethi",   3),
    (9,  "Zaid Anwar",  "Other Uh",    "Other Zh",    "Hina Sethi",  19),
    # Rabia Noor is an advisor and nothing else.
    (10, "Rabia Noor",  "Other Uh",    "Other Zh",    "Other Bcm",   31),
]

TAHIR_UNIT = 7 + 11 + 13 + 17    # 48
TAHIR_ZONE = 7 + 11 + 13         # 31
TAHIR_CENTRE = 7 + 11            # 18
TAHIR_OWN = 7

ADEEL_ZONE = 5 + 23              # 28  <- the highest ROLE
ADEEL_CENTRE = 5 + 29            # 34  <- the bigger NUMBER
ADEEL_OWN = 5

HINA_CENTRE = 3 + 19             # 22
HINA_OWN = 3
RABIA_OWN = 31


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
    session = f"p28-{text}"
    conversation_memory._store.pop(session, None)
    return handle_chat_message(db, text, session_id=session)


def _numbers(text):
    import re
    return {int(t.replace(",", "")) for t in re.findall(r"\d[\d,]*", text)}


def _is_clarification(response):
    return response["type"] == "clarification" and "could mean" in response["reply"]


# ---------------------------------------------------------------------
# The rank itself
# ---------------------------------------------------------------------


@pytest.mark.parametrize("levels,expected", [
    (["advisor"], "advisor"),
    (["advisor", "bcm"], "bcm"),
    (["advisor", "bcm", "zonal_head"], "zonal_head"),
    (["advisor", "bcm", "zonal_head", "unit_head"], "unit_head"),
])
def test_the_senior_role_is_chosen(levels, expected):
    assert _highest_role(levels) == expected


def test_the_order_the_levels_arrive_in_does_not_matter(db):
    """Ambiguity levels come out of a dict iteration; rank must come from
    the hierarchy, not from whichever grounding was seen first."""
    assert _highest_role(["unit_head", "advisor", "bcm"]) == "unit_head"
    assert _highest_role(["bcm", "advisor", "unit_head"]) == "unit_head"


def test_the_priority_is_read_from_the_chain_not_redeclared():
    """If CHAIN is ever rebound, this ranking follows it. Pinned so the
    priority cannot quietly fork into a second list."""
    assert list(nlu_pipeline._ROLE_LEVELS) == [
        lvl for lvl in hierarchy.CHAIN if lvl != "team"]
    assert list(nlu_pipeline._ROLE_LEVELS) == [
        "unit_head", "zonal_head", "bcm", "advisor"]


@pytest.mark.parametrize("levels", [
    ["advisor", "team"], ["bcm", "team"], ["advisor", "company"],
])
def test_a_name_that_is_also_a_group_still_has_no_answer(levels):
    """A person sharing a spelling with a TEAM is two different entities.
    No role ordering settles that, so the question is still asked."""
    assert _highest_role(levels) is None


def test_region_neither_blocks_nor_wins(db):
    """`region` still mixes places with people (hierarchy.AMBIGUOUS_LEVELS),
    which is why every Unit Head also grounds there. It is the same person
    under a stale column, not a fifth role."""
    assert _highest_role(["advisor", "bcm", "zonal_head", "unit_head", "region"]) == "unit_head"
    assert _highest_role(["advisor", "region"]) == "advisor"


# ---------------------------------------------------------------------
# Team queries use the highest role's hierarchy
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,total", [
    ("Tahir Malik", TAHIR_UNIT),
    ("Adeel Raza", ADEEL_ZONE),
    ("Hina Sethi", HINA_CENTRE),
])
def test_a_team_query_uses_the_highest_roles_scope(db, name, total):
    reply = _ask(db, f"connects of {name}'s team")["reply"]
    assert total in _numbers(reply)


def test_the_senior_role_wins_even_when_a_junior_one_is_bigger(db):
    """Adeel Raza leads 28 connects as Zonal Head and 34 as BCM. Rank
    decides, not size — and 34 is what picking by GROUP_LEVEL_ORDER
    returned before this phase."""
    reply = _ask(db, "connects of Adeel Raza's team")["reply"]
    assert ADEEL_ZONE in _numbers(reply)
    assert ADEEL_CENTRE not in _numbers(reply)


@pytest.mark.parametrize("name,members", [
    ("Tahir Malik", {"Tahir Malik", "Sana Riaz", "Bilal Khan", "Nida Aslam"}),
    ("Adeel Raza", {"Adeel Raza", "Kiran Shah"}),
    ("Hina Sethi", {"Hina Sethi", "Zaid Anwar"}),
])
def test_the_roster_is_the_highest_roles_subordinates(db, name, members):
    """Phase 27's breakdown, now reached without a clarification: the
    people listed are the ones under the RESOLVED role."""
    response = _ask(db, f"connects of {name}'s team")
    listed = {m["name"] for m in response["members"]}
    assert listed == members


@pytest.mark.parametrize("name,total", [
    ("Tahir Malik", TAHIR_UNIT), ("Adeel Raza", ADEEL_ZONE), ("Hina Sethi", HINA_CENTRE),
])
def test_the_members_still_sum_to_the_total(db, name, total):
    """Phase 27's invariant survives the new resolution — the roster and
    the headline describe the same population."""
    response = _ask(db, f"connects of {name}'s team")
    assert sum(m["value"] or 0 for m in response["members"]) == total


def test_omar_faruq_is_not_in_adeel_razas_team(db):
    """He reports to Adeel as BCM but not as Zonal Head. Picking the
    senior role must exclude him, which is the whole 34-vs-28 difference."""
    response = _ask(db, "connects of Adeel Raza's team")
    assert "Omar Faruq" not in response["reply"]


# ---------------------------------------------------------------------
# No clarification for a resolvable person
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Tahir Malik", "Adeel Raza", "Hina Sethi"])
def test_a_team_query_never_asks_which_role(db, name):
    assert not _is_clarification(_ask(db, f"connects of {name}'s team"))


@pytest.mark.parametrize("name", ["Tahir Malik", "Adeel Raza", "Hina Sethi"])
def test_a_ranking_under_a_person_never_asks_which_role(db, name):
    """A ranking asks for the group just as plainly as a possessive does."""
    assert not _is_clarification(_ask(db, f"top advisors under {name} by connects"))


@pytest.mark.parametrize("name", ["Tahir Malik", "Adeel Raza", "Hina Sethi"])
def test_a_person_query_never_asks_which_role(db, name):
    assert not _is_clarification(_ask(db, f"connects of {name}"))


def test_nothing_is_left_pending_when_no_question_was_asked(db):
    """A stored pending question with no question on screen would eat the
    next turn as if it were an answer."""
    session = "p28-pending"
    conversation_memory._store.pop(session, None)
    handle_chat_message(db, "connects of Tahir Malik's team", session_id=session)
    assert conversation_memory.get_pending_level(session) is None


# ---------------------------------------------------------------------
# Person queries are untouched (Phase 22)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,own", [
    ("Tahir Malik", TAHIR_OWN),
    ("Adeel Raza", ADEEL_OWN),
    ("Hina Sethi", HINA_OWN),
    ("Rabia Noor", RABIA_OWN),
])
def test_a_person_query_still_returns_their_own_connects(db, name, own):
    reply = _ask(db, f"connects of {name}")["reply"]
    assert own in _numbers(reply)


@pytest.mark.parametrize("name,group", [
    ("Tahir Malik", TAHIR_UNIT), ("Adeel Raza", ADEEL_ZONE), ("Hina Sethi", HINA_CENTRE),
])
def test_a_person_query_is_not_promoted_to_their_role(db, name, group):
    """The senior role is the default for the GROUP only. Reading it into
    a bare person query would undo Phase 22 — a manager's own record must
    stay reachable."""
    assert group not in _numbers(_ask(db, f"connects of {name}")["reply"])


def test_person_and_team_remain_different_scopes(db):
    person = _numbers(_ask(db, "connects of Tahir Malik")["reply"])
    team = _numbers(_ask(db, "connects of Tahir Malik's team")["reply"])
    assert TAHIR_OWN in person and TAHIR_UNIT not in person
    assert TAHIR_UNIT in team


def test_an_explicitly_stated_level_still_wins(db):
    """The user's own words outrank the default. Naming the junior role
    must reach the junior scope, or the senior default becomes a ceiling."""
    reply = _ask(db, "connects of BCM Tahir Malik")["reply"]
    assert TAHIR_CENTRE in _numbers(reply)
    assert TAHIR_UNIT not in _numbers(reply)


def test_an_advisor_with_no_role_leads_nobody(db):
    """Rabia Noor grounds at `advisor` alone. There is no role to promote
    her to, and Phase 20's guarantee holds: this must not become a global
    ranking of everyone."""
    response = _ask(db, "connects of Rabia Noor's team")
    assert TAHIR_UNIT not in _numbers(response["reply"])
    assert sum(c for _w, _n, _r, _p, _m, c in PEOPLE) not in _numbers(response["reply"])
