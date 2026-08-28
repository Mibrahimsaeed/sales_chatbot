"""Cross-turn follow-ups end to end (M4).

Written as CONVERSATIONS rather than as single queries, because that is
where this feature can go wrong: each message is fine in isolation and
the failure is that turn N inherits the wrong subject from turn N-1.

Equivalence against the explicitly-named form is the assertion of choice
throughout — "how is his team doing" must produce exactly what "how is
Blue Area doing" produces, since the user means the same thing.
"""

import time

import pytest

from app.core.config import settings
from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def db(db_session, monkeypatch):
    people = [
        (1, "Waqar Haider", "Blue Area", "Graana", "Gulberg BC", "Kaleem Ullah"),
        (2, "Sana Tariq", "Blue Area", "Graana", "Gulberg BC", "Kaleem Ullah"),
        (3, "Imran Butt", "Downtown", "Agency21", "Saddar BC", "Nadia Rehman"),
    ]
    for wid, name, team, company, office, bm in people:
        db_session.add(Advisor(wid=wid, name=name, team=team, company=company,
                               office=office, bm=bm, rm=bm, zm="Adeel Dogar", portfolio_lead="Adeel Dogar"))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=40, mtd_followup_connect=2))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000, cleared=500))
    db_session.add(Advisor(wid=4, name="Yasir Ali", team="Blue Area", company="Graana"))
    db_session.add(Advisor(wid=5, name="Yasir Ali", team="Downtown", company="Agency21"))
    db_session.commit()

    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "llm_first")
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    import app.llm.llm_client as llm_client
    # Patch the PROVIDER BOUNDARY, not a vendor SDK's call shape. The
    # previous form named llm_client._client.chat.completions.create —
    # four OpenAI internals — and every one of them broke when the client
    # became Ollama, erroring 146 tests that were not about the provider.
    monkeypatch.setattr(llm_client, "_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("off")))
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()


@pytest.fixture(autouse=True)
def inference_on(monkeypatch):
    monkeypatch.setattr(settings, "relation_inference_enabled", True)
    monkeypatch.setattr(settings, "relation_inference_levels",
                        "team,company,office,unit_head,zonal_head,bcm")


# ---------------------------------------------------------------------
# The acceptance conversation
# ---------------------------------------------------------------------

def test_the_full_follow_up_conversation(db):
    first = handle_chat_message(db, "Tell me about Waqar Haider", session_id="c")
    assert first["type"] == "advisor"

    team = handle_chat_message(db, "How is his team doing?", session_id="c")
    assert team["type"] == "team"
    assert team["reply"] == handle_chat_message(db, "How is Blue Area doing", session_id="named")["reply"]

    company = handle_chat_message(db, "What company does he work for?", session_id="c")
    assert company["type"] == "company"
    assert company["reply"] == handle_chat_message(db, "How is Graana doing", session_id="named")["reply"]

    manager = handle_chat_message(db, "Who is his unit head?", session_id="c")
    assert manager["type"] == "manager"
    assert "Kaleem Ullah" in manager["reply"]


# ---------------------------------------------------------------------
# Required regressions
# ---------------------------------------------------------------------

def test_topic_switch_to_another_advisor(db):
    """Mentioning a new person re-subjects the conversation."""
    handle_chat_message(db, "Tell me about Waqar Haider", session_id="c")
    assert "Blue Area" in handle_chat_message(db, "How is his team doing?", session_id="c")["reply"]

    handle_chat_message(db, "Tell me about Imran Butt", session_id="c")
    assert "Downtown" in handle_chat_message(db, "How is his team doing?", session_id="c")["reply"]


def test_topic_switch_away_from_people_entirely(db):
    """A non-person question must not inherit a subject, and must not
    destroy one either — the pronoun after it still works."""
    handle_chat_message(db, "Tell me about Waqar Haider", session_id="c")
    board = handle_chat_message(db, "Top 5 advisors by revenue", session_id="c")
    assert board["type"] == "leaderboard"

    assert "Blue Area" in handle_chat_message(db, "How is his team doing?", session_id="c")["reply"]


def test_explicit_name_overrides_memory_mid_conversation(db):
    handle_chat_message(db, "Tell me about Waqar Haider", session_id="c")
    r = handle_chat_message(db, "How is Imran Butt's team doing?", session_id="c")
    assert "Downtown" in r["reply"]


def test_multi_person_message_does_not_inherit(db):
    handle_chat_message(db, "Tell me about Waqar Haider", session_id="c")
    r = handle_chat_message(db, "Compare Imran Butt and Sana Tariq", session_id="c")
    assert r["type"] != "team"


def test_pronoun_after_an_unresolved_ambiguity_does_not_infer(db):
    """"Yasir Ali" matches two people, so no identity is settled and no
    subject may be inherited."""
    ambiguous = handle_chat_message(db, "Tell me about Yasir Ali", session_id="c")
    assert ambiguous["type"] == "clarification"

    follow_up = handle_chat_message(db, "How is his team doing?", session_id="c")
    assert follow_up["type"] != "team"


def test_pronoun_after_a_RESOLVED_ambiguity_does_infer(db):
    """Once the user picks, the follow-up works — the identity is now
    settled, which is the whole point of remembering it."""
    handle_chat_message(db, "Tell me about Yasir Ali", session_id="c")
    handle_chat_message(db, "the one in Downtown", session_id="c")

    r = handle_chat_message(db, "How is his team doing?", session_id="c")
    assert r["type"] == "team"
    assert "Downtown" in r["reply"]


def test_session_reset_forgets_the_subject(db):
    handle_chat_message(db, "Tell me about Waqar Haider", session_id="c")
    r = handle_chat_message(db, "How is his team doing?", session_id="fresh-session")
    assert r["type"] != "team"


def test_expired_memory_behaves_like_no_memory(db):
    handle_chat_message(db, "Tell me about Waqar Haider", session_id="c")
    conversation_memory._store["c"].saved_at = time.time() - (conversation_memory._TTL_SECONDS + 1)

    r = handle_chat_message(db, "How is his team doing?", session_id="c")
    assert r["type"] != "team"


def test_no_session_id_at_all(db):
    handle_chat_message(db, "Tell me about Waqar Haider", session_id=None)
    r = handle_chat_message(db, "How is his team doing?", session_id=None)
    assert r["type"] != "team"


# ---------------------------------------------------------------------
# Preservation of M0-M3
# ---------------------------------------------------------------------

def test_person_follow_ups_still_answer_about_the_person(db):
    """"what about his overdue" has a pronoun but no group word, so the
    subject stays the PERSON rather than becoming a group.

    UPDATED BY M7: the answer is now that one metric instead of the whole
    profile, because the follow-up names a metric. The property this test
    guards — the subject is Waqar, not his team — is unchanged, and M7
    composing with M4's cross-turn subject is the intended behaviour."""
    handle_chat_message(db, "Tell me about Waqar Haider", session_id="c")
    r = handle_chat_message(db, "What about his overdue?", session_id="c")
    assert r["type"] == "advisor_metric"
    assert "Waqar Haider" in r["reply"]
    assert "overdue" in r["reply"]


def test_a_pronoun_follow_up_with_no_metric_still_returns_the_profile(db):
    handle_chat_message(db, "Tell me about Waqar Haider", session_id="c")
    r = handle_chat_message(db, "What about him?", session_id="c")
    assert r["type"] == "advisor"


def test_reverse_lookup_follow_ups_unchanged(db):
    handle_chat_message(db, "Tell me about Waqar Haider", session_id="c")
    for query, expected in (("Who is his BM?", "Waqar Haider's BM is Kaleem Ullah."),
                            ("Who is his zonal head?", "Waqar Haider's Zonal Head is Adeel Dogar.")):
        r = handle_chat_message(db, query, session_id="c")
        assert r["type"] == "manager"
        assert r["reply"] == expected


def test_in_message_inference_unchanged(db):
    referred = handle_chat_message(db, "How is Waqar Haider's team doing", session_id="a")
    named = handle_chat_message(db, "How is Blue Area doing", session_id="b")
    assert referred["reply"] == named["reply"]


def test_flag_off_restores_pre_m4_behaviour(db, monkeypatch):
    monkeypatch.setattr(settings, "relation_inference_enabled", False)
    handle_chat_message(db, "Tell me about Waqar Haider", session_id="c")
    r = handle_chat_message(db, "How is his team doing?", session_id="c")
    assert r["type"] == "advisor"
