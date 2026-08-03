"""Phase 5 (duplicate-name handling carried through the conversation) and
Phase 6 (reverse hierarchy), end to end through handle_chat_message.

The property these lock in: once the user has told us WHICH person they
meant, that answer must survive the rest of the session. Re-asking is the
same class of failure as guessing — both discard what the conversation
already established.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def person_db(db_session, monkeypatch):
    # two real people share a name; each has a different manager chain, so
    # a wrong pick produces a visibly wrong answer rather than a silent one
    db_session.add(Advisor(
        wid=10, name="Yasir Ali", team="North/KPK", company="Agency21",
        bm="Aimal Khan", zm="Zarak Khan", portfolio_lead="Zarak Khan", rm="Atif Irfan",
    ))
    db_session.add(SalesFunnel(wid=10, mtd_new_connect=5, mtd_followup_connect=0))
    db_session.add(Performance(wid=10, period=PerformancePeriod.MTD, target=1000, cleared=500))

    db_session.add(Advisor(
        wid=11, name="Yasir Ali", team="Downtown", company="IMARAT",
        bm="Fraz Khalid", zm="Salman Arshad", portfolio_lead="Salman Arshad", rm="Rashid Majeed",
    ))
    db_session.add(SalesFunnel(wid=11, mtd_new_connect=7, mtd_followup_connect=0))
    db_session.add(Performance(wid=11, period=PerformancePeriod.MTD, target=2000, cleared=1500))

    db_session.add(Advisor(
        wid=20, name="Kainat Khalid", team="Blue Area", company="Graana",
        bm="Aimal Khan", zm="Zarak Khan", portfolio_lead="Zarak Khan", rm="Atif Irfan",
    ))
    db_session.add(SalesFunnel(wid=20, mtd_new_connect=42, mtd_followup_connect=0))
    # deliberately no bm/zm/rm — "no manager on file" must be said plainly
    db_session.add(Advisor(wid=30, name="Orphan Advisor", team="Gamma", company="Graana"))
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
    conversation_memory._store.clear()


# ---- Phase 5: the choice is carried ----

def test_answering_with_a_wid_reruns_the_original_question(person_db):
    """The user shouldn't have to retype what they already asked."""
    first = handle_chat_message(person_db, "tell me about Yasir Ali", session_id="s1")
    assert first["type"] == "clarification"

    second = handle_chat_message(person_db, "11", session_id="s1")
    assert second["type"] == "advisor"
    assert second["data"]["wid"] == 11


def test_answering_by_distinguishing_team_also_works(person_db):
    handle_chat_message(person_db, "tell me about Yasir Ali", session_id="s2")
    answer = handle_chat_message(person_db, "the one in Downtown", session_id="s2")
    assert answer["type"] == "advisor"
    assert answer["data"]["wid"] == 11


def test_the_same_name_is_not_asked_about_twice(person_db):
    """Regression: re-running the original query re-resolved the name from
    text and hit the same ambiguity again, so the question could never be
    answered — the answer was exactly what got discarded."""
    handle_chat_message(person_db, "tell me about Yasir Ali", session_id="s3")
    handle_chat_message(person_db, "10", session_id="s3")

    again = handle_chat_message(person_db, "tell me about Yasir Ali", session_id="s3")
    assert again["type"] == "advisor"
    assert again["data"]["wid"] == 10


def test_pronoun_followup_inherits_the_chosen_person(person_db):
    handle_chat_message(person_db, "tell me about Yasir Ali", session_id="s4")
    handle_chat_message(person_db, "11", session_id="s4")

    followup = handle_chat_message(person_db, "who is his BM?", session_id="s4")
    assert followup["type"] == "manager"
    assert followup["data"]["manager"] == "Fraz Khalid"   # wid 11's BM, not wid 10's


def test_choice_is_recorded_in_session_state(person_db):
    handle_chat_message(person_db, "tell me about Yasir Ali", session_id="s5")
    handle_chat_message(person_db, "10", session_id="s5")
    assert conversation_memory.get_resolved_advisor("s5") == (10, "Yasir Ali")


def test_an_unhelpful_answer_reasks_rather_than_guessing(person_db):
    handle_chat_message(person_db, "tell me about Yasir Ali", session_id="s6")
    retry = handle_chat_message(person_db, "idk", session_id="s6")
    assert retry["type"] == "clarification"
    assert conversation_memory.get_resolved_advisor("s6") is None


def test_a_new_question_abandons_the_pending_choice(person_db):
    handle_chat_message(person_db, "tell me about Yasir Ali", session_id="s7")
    moved_on = handle_chat_message(person_db, "tell me about Kainat Khalid", session_id="s7")
    assert moved_on["type"] == "advisor"
    assert moved_on["data"]["wid"] == 20


def test_unambiguous_lookup_also_establishes_the_conversation_subject(person_db):
    """A follow-up needs a subject whether or not a disambiguation
    happened."""
    handle_chat_message(person_db, "tell me about Kainat Khalid", session_id="s8")
    followup = handle_chat_message(person_db, "who is her zonal head?", session_id="s8")
    assert followup["type"] == "manager"
    assert followup["data"]["manager"] == "Zarak Khan"


# ---- Phase 6: reverse hierarchy ----

@pytest.mark.parametrize("phrasing,expected_level,expected_manager", [
    ("who is Kainat Khalid's BM?", "bm", "Aimal Khan"),
    ("who is Kainat Khalid's unit head?", "unit_head", "Atif Irfan"),
    ("who is Kainat Khalid's ZM?", "zm", "Zarak Khan"),
    ("who is Kainat Khalid's zonal head?", "zonal_head", "Zarak Khan"),
    ("who is Kainat Khalid's RM?", "unit_head", "Atif Irfan"),
    ("who is Kainat Khalid's regional manager?", "unit_head", "Atif Irfan"),
    ("who does Kainat Khalid report to?", "unit_head", "Atif Irfan"),
    ("who is Kainat Khalid's manager?", "unit_head", "Atif Irfan"),
])
def test_reverse_hierarchy_phrasings(person_db, phrasing, expected_level, expected_manager):
    response = handle_chat_message(person_db, phrasing, session_id=None)
    assert response["type"] == "manager"
    assert response["data"]["level"] == expected_level
    assert response["data"]["manager"] == expected_manager


def test_missing_manager_is_stated_plainly_not_answered_with_something_else(person_db):
    """A capability/data gap must not silently become a different,
    confident answer — the audit's core complaint."""
    response = handle_chat_message(person_db, "who is Orphan Advisor's BM?", session_id=None)
    assert response["type"] == "not_found"
    assert "BM" in response["reply"]


def test_reverse_hierarchy_on_an_ambiguous_name_asks_first(person_db):
    """Identity must be settled before a manager can be looked up — the
    two Yasir Alis have different BMs."""
    response = handle_chat_message(person_db, "who is Yasir Ali's BM?", session_id="s9")
    assert response["type"] == "clarification"


def test_forward_and_reverse_are_not_confused(person_db):
    """"who reports to X" (X's reports) vs "who does X report to" (X's
    manager) — near-identical strings, opposite questions."""
    reverse = handle_chat_message(person_db, "who does Kainat Khalid report to?", session_id=None)
    assert reverse["type"] == "manager"

    # Scoped by the Unit Head column (Advisor.rm) after the rebind.
    forward = handle_chat_message(person_db, "who reports to Atif Irfan?", session_id=None)
    assert forward["type"] == "breakdown"
