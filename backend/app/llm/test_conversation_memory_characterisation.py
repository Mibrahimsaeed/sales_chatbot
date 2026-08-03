"""Characterisation of cross-turn advisor memory (written BEFORE M4).

Records what already exists, because a surprising amount does:
`conversation_memory` has carried a resolved advisor since Phase 5, and
nlu_pipeline already re-applies it to a pronoun follow-up. M4 is not
building cross-turn memory — it is letting relationship inference reach
the subject that memory already holds.

Two current behaviours are load-bearing for M4's design and are pinned
here:

- The carry POPS `advisor_resolution` from the entity dict. That key is
  exactly what relation_resolver needs, so the M1 engine cannot see a
  remembered person no matter where it runs. M4 must source the identity
  from memory rather than from the entity dict.
- Inference runs inside extract_entities, which never receives a
  session_id, so it structurally cannot consult memory.
"""

import time

import pytest

from app.database.models import Advisor
from app.llm import advisor_resolver, conversation_memory, entity_extractor, nlu_pipeline


@pytest.fixture(autouse=True)
def clean_memory():
    conversation_memory._store.clear()
    yield
    conversation_memory._store.clear()


# ---------------------------------------------------------------------
# conversation_memory's existing contract
# ---------------------------------------------------------------------

def test_resolved_advisor_round_trips():
    conversation_memory.set_resolved_advisor("s1", 1, "Waqar Haider")
    assert conversation_memory.get_resolved_advisor("s1") == (1, "Waqar Haider")


def test_no_memory_returns_none():
    assert conversation_memory.get_resolved_advisor("unknown") is None
    assert conversation_memory.get_resolved_advisor(None) is None


def test_sessions_are_isolated():
    conversation_memory.set_resolved_advisor("s1", 1, "Waqar Haider")
    assert conversation_memory.get_resolved_advisor("s2") is None


def test_a_later_advisor_replaces_the_earlier_one():
    conversation_memory.set_resolved_advisor("s1", 1, "Waqar Haider")
    conversation_memory.set_resolved_advisor("s1", 3, "Imran Butt")
    assert conversation_memory.get_resolved_advisor("s1") == (3, "Imran Butt")


def test_memory_expires_with_the_session_ttl():
    conversation_memory.set_resolved_advisor("s1", 1, "Waqar Haider")
    state = conversation_memory._store["s1"]
    state.saved_at = time.time() - (conversation_memory._TTL_SECONDS + 1)
    assert conversation_memory.get_resolved_advisor("s1") is None


def test_resolved_advisor_survives_a_new_ir(db_session):
    """Deliberate: a new analytical query does not un-answer "who are we
    talking about"."""
    from app.llm.query_ir import QueryIR

    conversation_memory.set_resolved_advisor("s1", 1, "Waqar Haider")
    conversation_memory.set("s1", QueryIR(intent="leaderboard"))
    assert conversation_memory.get_resolved_advisor("s1") == (1, "Waqar Haider")


# ---------------------------------------------------------------------
# The pronoun heuristic that already exists
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "how is his team doing", "what about her overdue", "who is their unit head",
    "what company does he work for", "show me them", "this person's revenue",
])
def test_pronoun_follow_ups_are_recognised(text):
    assert nlu_pipeline._looks_like_person_followup(text)


@pytest.mark.parametrize("text", [
    "tell me about Waqar Haider", "top 5 advisors by revenue", "how is Blue Area doing",
])
def test_non_follow_ups_are_not(text):
    assert not nlu_pipeline._looks_like_person_followup(text)


# ---------------------------------------------------------------------
# Why M1/M3 inference cannot currently reach a remembered person
# ---------------------------------------------------------------------

def test_extract_entities_does_not_receive_a_session(db_session):
    """Inference runs inside extract_entities, which has no way to look
    up conversation memory. Recorded because M4's integration point
    follows from it."""
    import inspect

    params = inspect.signature(entity_extractor.extract_entities).parameters
    assert "session_id" not in params


def test_a_pronoun_message_alone_resolves_no_advisor(db_session):
    """Without memory there is no subject — extract_entities sees only
    the words in front of it."""
    db_session.add(Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana"))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()

    entities = entity_extractor.extract_entities("how is his team doing", db_session)
    assert entities.get("advisor_wid") is None
    assert entities.get("team") is None
