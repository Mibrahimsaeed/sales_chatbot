"""End-to-end behaviour lock for M0 (Relationship Inference Engine).

M0 is foundations only — a relation registry, a derived MANAGER_COLUMNS,
and a provenance slot — and its single hardest requirement is that NONE
of it changes an answer. Unit tests can show each piece is correct
without showing that the product still says the same words, so the
replies below are pinned literally, captured from a run of this exact
fixture BEFORE M0 existed.

Two entries are load-bearing in opposite directions:

- "Who is Waqar Haider's BM" must STILL WORK. Reverse lookup is the one
  advisor->X capability that works today, and it is routed through
  MANAGER_COLUMNS, which M0 re-homed into relations.py.
- "Who is Waqar Haider's portfolio lead" must STILL BE BROKEN. It
  returns the person's own profile because REVERSE_RE cannot express a
  two-word role. Fixing it is M2. A green M0 that accidentally fixed it
  would mean M0 had changed routing, which is exactly what it must not
  do.

This file is the seed of the golden-set differential the design calls
for ahead of M1, narrowed to the paths M0 could plausibly disturb.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message

# Replies recorded from the SAME fixture before M0 (impact-matrix run).
BASELINE = {
    "Tell me about Waqar Haider's team": ("advisor", "Waqar Haider has 42 MTD connects and has cleared 500 against a target of 1,000 (50%)."),
    "Who is Waqar Haider's BM": ("manager", "Waqar Haider's BM is Kaleem Ullah."),
    "Who is Waqar Haider's zonal head": ("manager", "Waqar Haider's Zonal Head is Sana Malik."),
    "Who is Waqar Haider's RM": ("manager", "Waqar Haider's Unit Head is Tariq Mehmood."),
    "Who does Waqar Haider report to": ("manager", "Waqar Haider's Unit Head is Tariq Mehmood."),
    # UPDATED BY M2 (deliberately). This entry was recorded as BROKEN —
    # it returned the person's own profile, because REVERSE_RE's
    # hand-written role list could not express a two-word role. M2
    # derived that vocabulary from the relation registry, and the lock
    # fired on the next run, which is precisely what it exists to do:
    # announce that a recorded behaviour changed. The change was the
    # approved objective of M2, so the expectation moves rather than the
    # code. Full coverage now lives in test_m2_reverse_roles_e2e.py.
    "Who is Waqar Haider's portfolio lead": ("manager", "Waqar Haider's Zonal Head is Sana Malik."),
    "How is Blue Area doing": ("team", "Blue Area has 2 advisors and 84 MTD connects. No target on file for this team."),
}


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid, name, team in ((1, "Waqar Haider", "Blue Area"), (2, "Sana Tariq", "Blue Area"),
                            (3, "Imran Butt", "Downtown")):
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                               bm="Kaleem Ullah", zm="Adeel Dogar", office="Gulberg BC",
                               rm="Tariq Mehmood", portfolio_lead="Sana Malik",
                               management_lead="Imran Shah", region="North", unit="1 Unit"))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=40, mtd_followup_connect=2))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000, cleared=500))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "llm_first")
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    import app.llm.llm_client as llm_client
    monkeypatch.setattr(llm_client._client.chat.completions, "create",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("off")))
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


@pytest.mark.parametrize("query", list(BASELINE))
def test_reply_is_byte_identical_to_pre_m0(db, query):
    expected_type, expected_reply = BASELINE[query]
    r = handle_chat_message(db, query, session_id=None)
    assert r["type"] == expected_type
    assert r["reply"] == expected_reply


def test_leaderboard_scoping_unchanged(db):
    r = handle_chat_message(db, "Top 5 advisors in Blue Area", session_id=None)
    assert "filtered by team = Blue Area" in r["reply"]
    assert len(r["data"]) == 2


def test_roster_unchanged(db):
    r = handle_chat_message(db, "Who works under Tariq Mehmood", session_id=None)
    assert r["type"] == "roster"
    assert len(r["data"]["advisors"]) == 3
