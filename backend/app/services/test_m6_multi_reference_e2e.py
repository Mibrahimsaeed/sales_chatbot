"""Multi-reference comparisons end to end (M6).

Every acceptance query, plus the regression classes: two advisors,
advisor vs explicit group, two explicit groups, mixed explicit/inferred,
ambiguous names, duplicate advisors, multiple possessives, nested
references and invalid references.

The comparisons are asserted on their TARGETS rather than on exact reply
text — the number and identity of the sides is what M6 changed; the
rendering belongs to a formatter this milestone did not touch.
"""

import pytest

from app.core.config import settings
from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def db(db_session, monkeypatch):
    people = [
        (1, "Waqar Haider", "Blue Area", "Graana"),
        (2, "Sana Tariq", "Downtown", "Agency21"),
        (3, "Imran Butt", "Downtown", "Agency21"),
        (4, "Yasir Ali", "Blue Area", "Graana"),
        (5, "Yasir Ali", "Downtown", "Agency21"),
    ]
    for wid, name, team, company in people:
        db_session.add(Advisor(wid=wid, name=name, team=team, company=company,
                               bm="Kaleem Ullah", rm="Kaleem Ullah", zm="Adeel Dogar", portfolio_lead="Adeel Dogar", office="Gulberg BC"))
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


@pytest.fixture(autouse=True)
def inference_on(monkeypatch):
    monkeypatch.setattr(settings, "relation_inference_enabled", True)
    monkeypatch.setattr(settings, "relation_inference_levels",
                        "team,company,office,unit_head,zonal_head,bcm")


def _sides(response):
    return [e["value"] for e in response["data"]["entities"]]


# ---------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------

def test_compare_two_advisors_teams(db):
    r = handle_chat_message(db, "Compare Waqar Haider's team with Sana Tariq's team", session_id=None)
    assert r["type"] == "comparison"
    assert _sides(r) == ["Blue Area", "Downtown"]


def test_compare_two_advisors_companies(db):
    r = handle_chat_message(db, "Compare Waqar Haider's company with Sana Tariq's company", session_id=None)
    assert r["type"] == "comparison"
    assert _sides(r) == ["Graana", "Agency21"]


def test_compare_an_advisors_team_with_an_explicit_team(db):
    r = handle_chat_message(db, "Compare Waqar Haider's team with Downtown", session_id=None)
    assert r["type"] == "comparison"
    assert set(_sides(r)) == {"Blue Area", "Downtown"}


def test_compare_an_advisors_team_with_an_explicit_company(db):
    r = handle_chat_message(db, "Compare Waqar Haider's team with Graana", session_id=None)
    assert r["type"] == "comparison"
    assert set(_sides(r)) == {"Blue Area", "Graana"}


def test_compare_an_advisor_with_their_own_team(db):
    r = handle_chat_message(db, "How does Waqar Haider compare to his team", session_id=None)
    assert r["type"] == "comparison"
    assert _sides(r) == ["Waqar Haider", "Blue Area"]


# ---------------------------------------------------------------------
# Regression classes
# ---------------------------------------------------------------------

def test_two_explicit_groups_unchanged(db):
    r = handle_chat_message(db, "Compare Blue Area with Downtown", session_id=None)
    assert _sides(r) == ["Blue Area", "Downtown"]


def test_cross_level_explicit_comparison_unchanged(db):
    r = handle_chat_message(db, "Compare Blue Area with Graana", session_id=None)
    assert set(_sides(r)) == {"Blue Area", "Graana"}


def test_duplicate_teams_do_not_produce_a_self_comparison(db):
    """Sana and Imran share a team. The important property is that
    Downtown is never compared with itself — deduplication leaves one
    target, so the query cannot render a column twice.

    KNOWN LIMITATION, characterised rather than left to be discovered:
    with one target the reply is the generic "I'm not tracking that one"
    rather than "they're both in Downtown". `_score_hierarchy` (0.92)
    outranks `comparison_incomplete` (0.76) and "compare" trips the
    compound gate, so the query reaches the semantic parser. This is
    unchanged from before M6 — the same query produced a single target
    then too — and improving it means planner scoring changes, which M6
    does not own."""
    r = handle_chat_message(db, "Compare Sana Tariq's team with Imran Butt's team", session_id=None)
    assert r["type"] != "comparison"
    assert r["reply"].count("Downtown") <= 1


def test_an_ambiguous_name_asks_instead_of_comparing(db):
    r = handle_chat_message(db, "Compare Yasir Ali's team with Downtown", session_id=None)
    assert r["type"] != "comparison"


def test_three_possessive_references(db):
    r = handle_chat_message(
        db, "Compare Waqar Haider's team with Sana Tariq's team and Yasir Ali's company",
        session_id=None)
    # Yasir Ali is ambiguous, so his side never grounds; the two that do
    # still compare rather than the whole query failing.
    assert r["type"] == "comparison"
    assert set(_sides(r)) >= {"Blue Area", "Downtown"}


def test_an_invalid_reference_does_not_invent_a_side(db):
    r = handle_chat_message(db, "Compare Nobody At All's team with Downtown", session_id=None)
    assert r["type"] != "comparison" or "Blue Area" not in _sides(r)


def test_mixed_explicit_and_inferred_keeps_both(db):
    r = handle_chat_message(db, "Compare Downtown with Waqar Haider's team", session_id=None)
    assert set(_sides(r)) == {"Downtown", "Blue Area"}


# ---------------------------------------------------------------------
# Everything M1-M5 must keep doing
# ---------------------------------------------------------------------

def test_single_reference_queries_unchanged(db):
    referred = handle_chat_message(db, "How is Waqar Haider's team doing", session_id="a")
    named = handle_chat_message(db, "How is Blue Area doing", session_id="b")
    assert referred["reply"] == named["reply"]


def test_person_lookup_unchanged(db):
    assert handle_chat_message(db, "tell me about Waqar Haider", session_id=None)["type"] == "advisor"


def test_reverse_lookup_unchanged(db):
    r = handle_chat_message(db, "Who is Waqar Haider's BM", session_id=None)
    assert r["type"] == "manager"
    assert "Kaleem Ullah" in r["reply"]


def test_cross_turn_pronouns_still_work(db):
    handle_chat_message(db, "Tell me about Waqar Haider", session_id="c")
    r = handle_chat_message(db, "How is his team doing?", session_id="c")
    assert r["type"] == "team"
    assert "Blue Area" in r["reply"]


def test_leaderboard_scoping_unchanged(db):
    r = handle_chat_message(db, "Top advisors in Waqar Haider's team", session_id=None)
    assert r["type"] == "leaderboard"
    assert {row["name"] for row in r["data"]} == {"Waqar Haider", "Yasir Ali"}


def test_flag_off_restores_pre_m6_behaviour(db, monkeypatch):
    monkeypatch.setattr(settings, "relation_inference_enabled", False)
    r = handle_chat_message(db, "Compare Waqar Haider's team with Sana Tariq's team", session_id=None)
    assert r["type"] != "comparison"
