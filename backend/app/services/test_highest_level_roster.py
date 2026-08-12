"""Phase 33 — one person, one level.

"connects of all BCMs" listed 181 people. 87 of them are really Zonal
Heads or Unit Heads: a Unit Head is named in `rm` by his 75 advisors and
in `management_lead` by the handful directly beneath him, so he is a
value in both columns and appeared in both answers. The same person was
counted at two levels of the same hierarchy.

A ROLE LEVEL IS A COLUMN OF NAMES, not a set of people, which is why
this could not be fixed by raising a limit — every one of those 87 was a
genuine value in `management_lead`. The rule is the hierarchy's own:
someone belongs at the senior-most level they hold, decided per person
BEFORE the roster is built.

Expressed as "not named in any column above this one" — the same
statement read from the columns rather than from a per-person lookup, so
there is no traversal and no second ranking to keep in sync. The level
order is `hierarchy.CHAIN` minus `team`, exactly as
nlu_pipeline._ROLE_LEVELS derives it.

THE LEAF IS DELIBERATELY EXEMPT. Excluding managers from `advisor` would
drop 181 people out of "all advisors" and out of every advisor
leaderboard. An advisor row is a PERSON; the manager columns are ROLES,
and only roles can be held at several levels at once.

Production, before -> after: unit_head 11 -> 11, zonal_head 88 -> 78,
bcm 181 -> 94, advisor 573 -> 573.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, conversation_memory, entity_extractor, hierarchy,
    narrative, query_compiler, semantic_parser,
)
from app.services.chat_service import handle_chat_message, handle_show_more

# Uma Farooq holds all three roles; Zia Ahmed holds two; the fourteen
# "Bcm NN" hold one. Fourteen is above the old cap of 10 on purpose — a
# correct dedup that still truncated would look identical at ten.
UNIT_HEAD = "Uma Farooq"
BOTH_ZH_AND_BCM = "Zia Ahmed"
ZH_ONLY = "Qasim Butt"
BCM_COUNT = 14
ADVISORS = 30

EXPECTED_UNIT_HEADS = {UNIT_HEAD}
EXPECTED_ZONAL_HEADS = {BOTH_ZH_AND_BCM, ZH_ONLY}          # Uma excluded
EXPECTED_BCMS = {f"Bcm {i:02d}" for i in range(1, BCM_COUNT + 1)}  # Zia, Uma excluded


def _zonal_for(wid):
    if wid <= 10:
        return BOTH_ZH_AND_BCM
    if wid <= 20:
        return ZH_ONLY
    return UNIT_HEAD


def _bcm_for(wid):
    if wid == 29:
        return BOTH_ZH_AND_BCM
    if wid == 30:
        return UNIT_HEAD
    return f"Bcm {(wid - 1) % BCM_COUNT + 1:02d}"


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid in range(1, ADVISORS + 1):
        db_session.add(Advisor(
            wid=wid, name=f"Advisor {wid:02d}", team="Alpha", company="Graana",
            rm=UNIT_HEAD, portfolio_lead=_zonal_for(wid),
            management_lead=_bcm_for(wid), in_master_sheet=True))
        value = wid
        db_session.add(Calls(wid=wid, connects_mtd=value, connects_daily=0,
                             answered_calls_mtd=value, answered_calls_daily=0))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=value,
                                   mtd_followup_connect=0, mtd_cr=value,
                                   mtd_new_meeting=value, mtd_followup_meeting=0))
        db_session.add(Pipeline(wid=wid, pipeline=value, overdue=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=value))
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


def _everyone(db, text):
    """Every name the question reaches, across all its pages."""
    session = f"p33-{text}"
    conversation_memory._store.pop(session, None)
    response = handle_chat_message(db, text, session_id=session)
    names = [row["name"] for row in response["data"]]
    while response.get("has_more"):
        response = handle_show_more(db, session)
        names += [row["name"] for row in response["data"]]
    return names


# ---------------------------------------------------------------------
# Each level holds only the people whose highest role it is
# ---------------------------------------------------------------------


def test_all_bcms_returns_every_real_bcm(db):
    """Fourteen of them — above the old cap, so this fails both if the
    dedup is missing and if a truncation survives."""
    assert set(_everyone(db, "connects of all BCMs")) == EXPECTED_BCMS
    assert len(EXPECTED_BCMS) > 10


def test_all_zonal_heads_returns_every_real_zonal_head(db):
    assert set(_everyone(db, "connects of all zonal heads")) == EXPECTED_ZONAL_HEADS


def test_all_unit_heads_returns_every_unit_head(db):
    """The top of the chain has nothing above it, so nobody is excluded."""
    assert set(_everyone(db, "connects of all unit heads")) == EXPECTED_UNIT_HEADS


def test_a_bcm_who_is_also_a_zonal_head_appears_only_as_a_zonal_head(db):
    assert BOTH_ZH_AND_BCM not in _everyone(db, "connects of all BCMs")
    assert BOTH_ZH_AND_BCM in _everyone(db, "connects of all zonal heads")


def test_a_zonal_head_who_is_also_a_unit_head_appears_only_as_a_unit_head(db):
    assert UNIT_HEAD not in _everyone(db, "connects of all zonal heads")
    assert UNIT_HEAD in _everyone(db, "connects of all unit heads")


def test_someone_holding_all_three_roles_appears_once_in_the_whole_hierarchy(db):
    """The property the levels must satisfy together, not one at a time."""
    appearances = [
        level for level, query in (
            ("bcm", "connects of all BCMs"),
            ("zonal_head", "connects of all zonal heads"),
            ("unit_head", "connects of all unit heads"),
        ) if UNIT_HEAD in _everyone(db, query)
    ]
    assert appearances == ["unit_head"]


def test_no_person_is_listed_at_two_levels(db):
    """Stated over the whole roster rather than over one name: the three
    levels must partition the managers."""
    bcms = set(_everyone(db, "connects of all BCMs"))
    zonals = set(_everyone(db, "connects of all zonal heads"))
    units = set(_everyone(db, "connects of all unit heads"))
    assert bcms & zonals == set()
    assert zonals & units == set()
    assert bcms & units == set()


def test_the_level_order_is_the_hierarchys_own(db):
    """One ranking in the codebase. If CHAIN is rebound, this follows."""
    assert query_compiler._ROLE_LEVELS == [
        lvl for lvl in hierarchy.CHAIN if lvl != "team"]


# ---------------------------------------------------------------------
# The leaf is exempt, and metrics are unmoved
# ---------------------------------------------------------------------


def test_all_advisors_still_includes_everyone(db):
    """A manager is still an advisor with their own row and their own
    figures. Dropping them here would change every advisor answer."""
    assert len(_everyone(db, "connects of all advisors")) == ADVISORS


def test_an_advisor_leaderboard_is_unchanged(db):
    assert len(handle_chat_message(db, "top advisors by connects",
                                   session_id=None)["data"]) == 10


def test_the_metric_values_are_untouched(db):
    """Dedup decides WHO is listed, never what their number is. Bcm 01
    holds advisors 1, 15 and 29 -> 1 + 15 = 16, since 29 belongs to Zia."""
    rows = {r["name"]: r["value"] for r in
            handle_chat_message(db, "connects of all BCMs", session_id="v")["data"]}
    assert rows["Bcm 01"] == 1 + 15


def test_a_persons_own_metric_is_unaffected(db):
    reply = handle_chat_message(db, "connects of Advisor 07", session_id=None)["reply"]
    assert "7" in reply


def test_a_team_query_still_reaches_every_subordinate(db):
    """Scoping DOWN from a manager is a different operation from listing
    managers, and it must still see everyone beneath them."""
    response = handle_chat_message(db, f"connects of {UNIT_HEAD}'s team",
                                   session_id=None)
    assert sum(m["value"] or 0 for m in response["members"]) == sum(range(1, ADVISORS + 1))


def test_a_unit_head_roster_is_unaffected(db):
    """Phase 30's roster path reads the advisors under a manager, not the
    managers themselves — no dedup applies."""
    response = handle_chat_message(db, f"show all advisors under Unit Head {UNIT_HEAD}",
                                   session_id=None)
    assert response["type"] == "roster"
    assert response["data"]["count"] == ADVISORS
