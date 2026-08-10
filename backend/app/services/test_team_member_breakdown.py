"""Phase 27 — a team total says who it is made of.

"connects of Haseeb Arslan's team" answered `9,635 Total MTD Connects`
and stopped. The number was right and unactionable: nothing said who is
in the team, who is carrying it, or who is at zero.

THE PEOPLE WERE NEVER MISSING FROM THE QUERY, only from the reply. A
manager scope is a filter on one hierarchy column, so the SAME QueryIR
with `subject_level` dropped to advisor enumerates exactly the same
population — no second definition of "who is under X", and no traversal,
because the hierarchy columns are denormalised and the manager filter
already reaches every advisor beneath them.

That is why the members sum to the total by construction rather than by
coincidence, and it is the property most of these tests assert.

THE TOTAL IS NOT RE-DERIVED. It stays the aggregation engine's number and
the headline of the reply; the list is appended. The displayed list is
capped (a Unit Head here has 76 advisors, and 140 in production), so
summing the DISPLAYED rows would understate the moment it truncates —
the full list lives in the response payload instead.
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

#   Zara Iqbal (zonal head)      own 5   | zone   = 5+11+22+33 = 71
#     Basit Khan (bcm)           own 11  | centre = 5+11+22    = 38
#       Umair Ahmed (unit head)  own 22  | unit   = 5+11+22    = 38
#       Nida Aslam               own 33
#   Kamil Yousaf                 own 99  — outside every scope
PEOPLE = [
    # wid, name,          team,    rm,            management_lead, portfolio_lead, connects
    (1, "Zara Iqbal",    "Alpha", "Umair Ahmed", "Basit Khan",  "Zara Iqbal",  5),
    (2, "Basit Khan",    "Alpha", "Umair Ahmed", "Basit Khan",  "Zara Iqbal",  11),
    (3, "Umair Ahmed",   "Alpha", "Umair Ahmed", "Basit Khan",  "Zara Iqbal",  22),
    (4, "Nida Aslam",    "Beta",  "Other Rm",    "Other Ml",    "Zara Iqbal",  33),
    (5, "Kamil Yousaf",  "Gamma", "Other Rm",    "Other Ml",    "Other Pl",    99),
]

ZONE = 5 + 11 + 22 + 33
CENTRE = 5 + 11 + 22
UNIT = 5 + 11 + 22
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


def _team(db, name, level):
    """A person's-team query, answering the level question if it is asked."""
    session = f"tm-{name}-{level}"
    conversation_memory._store.pop(session, None)
    response = handle_chat_message(db, f"connects of {name}'s team", session_id=session)
    if response["type"] == "clarification" and "could mean" in response["reply"]:
        response = handle_chat_message(db, level, session_id=session)
    return response


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
# Members appear, at every manager level
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,level,members", [
    ("Basit Khan", "BCM", {"Zara Iqbal", "Basit Khan", "Umair Ahmed"}),
    ("Zara Iqbal", "Zonal Head", {"Zara Iqbal", "Basit Khan", "Umair Ahmed", "Nida Aslam"}),
    ("Umair Ahmed", "Unit Head", {"Zara Iqbal", "Basit Khan", "Umair Ahmed"}),
])
def test_a_team_query_names_its_members(db, name, level, members):
    reply = _team(db, name, level)["reply"]
    for member in members:
        assert member in reply, f"{member} missing from the breakdown"


@pytest.mark.parametrize("name,level", [
    ("Basit Khan", "BCM"), ("Zara Iqbal", "Zonal Head"), ("Umair Ahmed", "Unit Head"),
])
def test_a_team_query_shows_each_members_own_value(db, name, level):
    reply = _team(db, name, level)["reply"]
    got = _numbers(reply)
    assert {5, 11, 22} <= got, "individual member values are not shown"


@pytest.mark.parametrize("name,level,total", [
    ("Basit Khan", "BCM", CENTRE),
    ("Zara Iqbal", "Zonal Head", ZONE),
    ("Umair Ahmed", "Unit Head", UNIT),
])
def test_the_total_is_still_the_headline(db, name, level, total):
    """The number the user asked for stays first and unchanged — the
    breakdown is appended, never substituted."""
    reply = _team(db, name, level)["reply"]
    assert total in _numbers(reply)
    assert reply.startswith(name)


def test_nobody_outside_the_scope_is_listed(db):
    reply = _team(db, "Zara Iqbal", "Zonal Head")["reply"]
    assert "Kamil Yousaf" not in reply
    assert OUTSIDE not in _numbers(reply)


# ---------------------------------------------------------------------
# The members sum to the total
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,level,total", [
    ("Basit Khan", "BCM", CENTRE),
    ("Zara Iqbal", "Zonal Head", ZONE),
    ("Umair Ahmed", "Unit Head", UNIT),
])
def test_the_members_sum_to_the_total(db, name, level, total):
    """The property that proves the breakdown and the total describe the
    same population. Read from the payload, which carries EVERY member —
    the reply caps its list, so summing the prose would understate a
    large team."""
    response = _team(db, name, level)
    members = response.get("members")
    assert members, "no member breakdown in the payload"
    assert sum(m["value"] or 0 for m in members) == total


def test_the_payload_carries_every_member_not_just_the_displayed_ones(db):
    response = _team(db, "Zara Iqbal", "Zonal Head")
    assert len(response["members"]) == 4


# ---------------------------------------------------------------------
# Person scope is untouched (Phase 22)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,own", [
    ("Zara Iqbal", 5), ("Basit Khan", 11), ("Umair Ahmed", 22),
])
def test_a_person_query_still_returns_only_their_own_value(db, name, own):
    reply = handle_chat_message(db, f"connects of {name}", session_id=None)["reply"]
    assert own in _numbers(reply)


def test_a_person_query_has_no_member_breakdown(db):
    """A person is not a group. The breakdown must not leak into the
    answer about one individual."""
    response = handle_chat_message(db, "connects of Zara Iqbal", session_id=None)
    assert not response.get("members")
    assert "👥" not in response["reply"]


@pytest.mark.parametrize("name,level,own,group", [
    ("Zara Iqbal", "Zonal Head", 5, ZONE),
    ("Basit Khan", "BCM", 11, CENTRE),
])
def test_person_and_team_remain_different_scopes(db, name, level, own, group):
    """Phase 22's distinction, re-pinned: adding the breakdown must not
    blur the two readings back together."""
    person = handle_chat_message(db, f"connects of {name}", session_id=None)["reply"]
    team = _team(db, name, level)["reply"]

    assert own in _numbers(person)
    assert group not in _numbers(person)
    assert group in _numbers(team)


# ---------------------------------------------------------------------
# Named teams and leaderboards are unaffected
# ---------------------------------------------------------------------


def test_a_named_team_gets_no_manager_breakdown(db):
    """"Alpha" is a team, not a person — there are no subordinates to
    enumerate, and its own rows are not a manager's team."""
    response = handle_chat_message(db, "connects of Alpha", session_id=None)
    assert not response.get("members")


def test_a_leaderboard_is_unaffected(db):
    """It already lists its own rows; a second list would duplicate them."""
    response = handle_chat_message(db, "top advisors by connects", session_id=None)
    assert not response.get("members")
