"""End-to-end regression tests for the Phase 1 identity refactor.

Each test here corresponds to a SPECIFIC wrong answer reproduced in the
pipeline audit against production data. They run through the real
handle_chat_message() path (rules_first, no LLM) so a regression is caught
at the level the user actually experiences, not just in a unit.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def identity_db(db_session, monkeypatch):
    # "Adeel Dogar" is a UNIT HEAD (Advisor.bm) with reports, and is NOT an
    # advisor — but "Adeel Mubarik Dogar" IS a different, unrelated advisor.
    # This is the exact production shape that produced the wrong answer.
    db_session.add(Advisor(wid=1, name="Advisor One", team="Blue Area", company="Graana", bm="Adeel Dogar", rm="Adeel Dogar"))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=10, mtd_followup_connect=0))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, target=1000, cleared=500))

    db_session.add(Advisor(wid=2, name="Advisor Two", team="Downtown", company="IMARAT", bm="Adeel Dogar", rm="Adeel Dogar"))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=20, mtd_followup_connect=0))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, target=2000, cleared=1000))

    db_session.add(Advisor(wid=3, name="Adeel Mubarik Dogar", team="Gamma", company="Graana"))
    db_session.add(SalesFunnel(wid=3, mtd_new_connect=99, mtd_followup_connect=0))
    db_session.add(Performance(wid=3, period=PerformancePeriod.MTD, target=5000, cleared=2500))

    # two real people sharing one name — the duplicate-name case
    db_session.add(Advisor(wid=10, name="Yasir Ali", team="North/KPK", company="Agency21"))
    db_session.add(SalesFunnel(wid=10, mtd_new_connect=5, mtd_followup_connect=0))
    db_session.add(Advisor(wid=11, name="Yasir Ali", team="Blue Area", company="Graana"))
    db_session.add(SalesFunnel(wid=11, mtd_new_connect=7, mtd_followup_connect=0))

    # a uniquely-named advisor, for the happy path
    db_session.add(Advisor(wid=20, name="Waqar Haider", team="Blue Area", company="Graana"))
    db_session.add(SalesFunnel(wid=20, mtd_new_connect=42, mtd_followup_connect=0))
    db_session.add(Performance(wid=20, period=PerformancePeriod.MTD, target=800, cleared=400))
    db_session.commit()

    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


# ---- AUDIT C3+C4: "X's team" returned a lookalike's personal profile ----

def test_possessive_team_query_returns_the_unit_heads_team_not_a_lookalike(identity_db):
    """Reproduced in the audit: this returned "Adeel Mubarik Dogar has 99
    MTD connects…" — a different human's individual profile — because the
    whole sentence fuzzy-matched that name at 0.62 and the advisor-lookup
    branch fired before the hierarchy branch."""
    response = handle_chat_message(identity_db, "Show Adeel Dogar's team", session_id="id-1")

    assert response["type"] == "breakdown"
    assert response["data"]["value"] == "Adeel Dogar"
    assert response["data"]["advisors"] == 2
    assert "Adeel Mubarik Dogar" not in response["reply"]


def test_who_reports_to_is_consistent_with_the_possessive_phrasing(identity_db):
    """The audit's inconsistency: "Who reports to X" scored 0.57 (correct
    answer) while "Show X's team" scored 0.62 (wrong person) — a 0.05
    scoring difference silently flipped the result. Both must now agree."""
    a = handle_chat_message(identity_db, "Who reports to Adeel Dogar?", session_id="id-2")
    b = handle_chat_message(identity_db, "Show Adeel Dogar's team", session_id="id-3")

    assert a["type"] == b["type"] == "breakdown"
    assert a["data"]["value"] == b["data"]["value"] == "Adeel Dogar"


def test_who_is_in_x_team_routes_to_the_hierarchy_not_a_lookup(identity_db):
    response = handle_chat_message(identity_db, "Who is in Adeel Dogar's team?", session_id="id-4")
    assert response["type"] == "breakdown"


# ---- AUDIT C1+C2: duplicate names silently returned the lowest WID ----

def test_duplicate_name_asks_which_person_instead_of_guessing(identity_db):
    """Previously returned whichever of the two had the lower wid, with no
    indication that a choice had been made at all."""
    response = handle_chat_message(identity_db, "tell me about Yasir Ali", session_id="id-5")

    assert response["type"] == "clarification"
    assert "Yasir Ali" in response["reply"]
    # both real people are offered, each with distinguishing context
    assert "North/KPK" in response["reply"]
    assert "Blue Area" in response["reply"]
    assert len(response["options"]) == 2


def test_unique_name_still_answers_directly_without_a_prompt(identity_db):
    """The disambiguation must not regress the common case."""
    response = handle_chat_message(identity_db, "tell me about Waqar Haider", session_id="id-6")

    assert response["type"] == "advisor"
    assert response["data"]["wid"] == 20
    assert "Waqar Haider" in response["reply"]


def test_lookup_dispatches_by_wid_not_by_name(identity_db):
    """Identity is carried as a wid end to end — the returned record must
    be the exact person resolved, verifiable by primary key."""
    response = handle_chat_message(identity_db, "tell me about Adeel Mubarik Dogar", session_id="id-7")

    assert response["type"] == "advisor"
    assert response["data"]["wid"] == 3


# ---- no regression to aggregate queries ----

def test_leaderboards_are_unaffected_by_the_identity_refactor(identity_db):
    response = handle_chat_message(identity_db, "top 3 advisors by connects", session_id="id-8")

    assert response["type"] == "leaderboard"
    assert response["data"][0]["name"] == "Adeel Mubarik Dogar"   # 99 connects
    assert response["data"][0]["value"] == 99
