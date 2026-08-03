"""Cross-turn reference resolution (app/llm/cross_turn_resolver.py) — M4.

The gate is the whole design, so it is what these tests exercise: five
requirements collapsed into one rule, each clause independently checked.
A cross-turn inference that fires when it should not is a wrong-subject
answer — the same failure class as resolving the wrong person, delivered
one turn later.
"""

import time

import pytest

from app.core.config import settings
from app.database.models import Advisor
from app.llm import (
    advisor_resolver, conversation_memory, cross_turn_resolver,
    entity_extractor, reference_parser,
)
from app.llm.entity_extractor import PROVENANCE_KEY


@pytest.fixture()
def db(db_session):
    db_session.add(Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana",
                           office="Gulberg BC", bm="Kaleem Ullah", rm="Kaleem Ullah", zm="Adeel Dogar", portfolio_lead="Adeel Dogar"))
    db_session.add(Advisor(wid=3, name="Imran Butt", team="Downtown", company="Agency21",
                           office="Saddar BC", bm="Nadia Rehman", rm="Nadia Rehman", zm="Faisal Iqbal", portfolio_lead="Faisal Iqbal"))
    db_session.add(Advisor(wid=4, name="Yasir Ali", team="Blue Area", company="Graana"))
    db_session.add(Advisor(wid=5, name="Yasir Ali", team="Downtown", company="Agency21"))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()


@pytest.fixture(autouse=True)
def inference_on(monkeypatch):
    monkeypatch.setattr(settings, "relation_inference_enabled", True)
    monkeypatch.setattr(settings, "relation_inference_levels",
                        "team,company,office,unit_head,zonal_head,bcm")


def _resolve(text, db, session_id, entities=None):
    entities = entities if entities is not None else entity_extractor.extract_entities(text, db)
    bound = cross_turn_resolver.resolve(text, entities, db, session_id, PROVENANCE_KEY)
    return bound, entities


# ---------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------

def test_pronoun_reference_resolves_from_memory(db):
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    bound, entities = _resolve("how is his team doing", db, "s")
    assert bound == ["team"]
    assert entities["team"] == "Blue Area"


def test_inferred_entity_carries_source_provenance(db):
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    _, entities = _resolve("how is his team doing", db, "s")
    assert entity_extractor.provenance_of(entities, "team") == "inferred:advisor:1"


def test_non_adjacent_phrasing_resolves(db):
    """"what company does he work for" separates the pronoun from the
    level word; requiring adjacency would have missed it."""
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    bound, entities = _resolve("what company does he work for", db, "s")
    assert bound == ["company"]
    assert entities["company"] == "Graana"


def test_the_newly_cached_relations_work_too(db):
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    _, entities = _resolve("how is his centre doing", db, "s")
    assert entities["office"] == "Gulberg BC"


# ---------------------------------------------------------------------
# The gate — requirements 1 to 5
# ---------------------------------------------------------------------

def test_explicit_name_in_the_message_always_wins(db):
    """Requirement 2. The message names Imran Butt; memory holds Waqar."""
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    bound, entities = _resolve("how is Imran Butt's team doing", db, "s")
    assert bound == []
    # M1's in-message inference resolved it instead — to the named person.
    assert entities["team"] == "Downtown"


def test_two_advisors_in_the_message_blocks_memory_inference(db):
    """Requirement 3: choosing either would be arbitrary."""
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    bound, _ = _resolve("compare Imran Butt and Yasir Ali on his team", db, "s")
    assert bound == []


def test_no_memory_means_no_inference(db):
    """Requirement 4: behave exactly as before."""
    bound, entities = _resolve("how is his team doing", db, "no-such-session")
    assert bound == []
    assert entities.get("team") is None


def test_no_pronoun_means_no_inference(db):
    """Requirement 5: a topic change cannot inherit a subject."""
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    bound, entities = _resolve("top 5 advisors by revenue", db, "s")
    assert bound == []
    assert entities.get("team") is None


def test_a_bare_group_question_does_not_inherit(db):
    """"how is the team doing" names no pronoun — not a follow-up."""
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    bound, _ = _resolve("how is the team doing", db, "s")
    assert bound == []


def test_expired_memory_does_not_infer(db):
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    conversation_memory._store["s"].saved_at = time.time() - (conversation_memory._TTL_SECONDS + 1)
    bound, _ = _resolve("how is his team doing", db, "s")
    assert bound == []


def test_session_isolation(db):
    conversation_memory.set_resolved_advisor("s1", 1, "Waqar Haider")
    bound, _ = _resolve("how is his team doing", db, "s2")
    assert bound == []


def test_a_remembered_wid_that_no_longer_exists_infers_nothing(db):
    conversation_memory.set_resolved_advisor("s", 9999, "Ghost")
    bound, _ = _resolve("how is his team doing", db, "s")
    assert bound == []


def test_explicitly_named_group_is_not_overwritten(db):
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    bound, entities = _resolve("how is his Downtown team doing", db, "s")
    assert entities["team"] == "Downtown"
    assert "team" not in bound


# ---------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------

def test_disabled_flag_blocks_cross_turn_inference(db, monkeypatch):
    monkeypatch.setattr(settings, "relation_inference_enabled", False)
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    bound, _ = _resolve("how is his team doing", db, "s")
    assert bound == []


def test_level_enablement_is_respected(db, monkeypatch):
    monkeypatch.setattr(settings, "relation_inference_levels", "company")
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    bound, _ = _resolve("how is his team doing", db, "s")
    assert bound == []


# ---------------------------------------------------------------------
# Role versus group
# ---------------------------------------------------------------------

def test_role_phrasings_are_left_to_reverse_lookup(db):
    """"who is his unit head" asks WHO — reverse lookup answers it, and
    binding the group would turn it into a breakdown."""
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    for text in ("who is his unit head", "who is his bm", "who is his zonal head"):
        bound, _ = _resolve(text, db, "s")
        assert bound == [], text


def test_bare_group_words_do_bind(db):
    """UPDATED BY M5's alias split: "his unit" now means his UNIT (the
    organisational grouping), not his unit head. The property under test
    — a bare group word binds where a role phrasing does not — is
    unchanged; only which level "unit" denotes moved. "his division"
    still reaches unit_head, since that word did not move."""
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")

    bound, entities = _resolve("how is his division doing", db, "s")
    assert bound == ["unit_head"]
    assert entities["unit_head"] == "Kaleem Ullah"


# ---------------------------------------------------------------------
# The shared writer
# ---------------------------------------------------------------------

def test_bind_relation_produces_the_same_shape_as_in_message_inference(db):
    """One writer for both paths — a follow-up must not produce a subtly
    different entity dict from the query that established the subject."""
    conversation_memory.set_resolved_advisor("s", 1, "Waqar Haider")
    _, cross_turn = _resolve("how is his team doing", db, "s")
    in_message = entity_extractor.extract_entities("how is Waqar Haider's team doing", db)

    for key in ("team", "teams", "team_matches"):
        assert cross_turn[key] == in_message[key]
    assert (cross_turn[PROVENANCE_KEY]["team"] == in_message[PROVENANCE_KEY]["team"])


# ---------------------------------------------------------------------
# Parser-level unit checks
# ---------------------------------------------------------------------

def test_parse_pronoun_requires_a_pronoun():
    assert reference_parser.parse_pronoun("how is the team doing") == []
    assert reference_parser.parse_pronoun("") == []


def test_parse_pronoun_finds_the_level():
    [reference] = reference_parser.parse_pronoun("how is his team doing")
    assert reference.target_level == "team"
    assert reference.kind == reference_parser.PRONOUN
    assert reference.source_span == ""


def test_named_parse_is_unchanged_by_pronoun_support():
    """M1's parse() must return exactly what it always did."""
    assert reference_parser.parse("how is his team doing") == []
    [named] = reference_parser.parse("Waqar Haider's team")
    assert named.kind == reference_parser.NAMED
