"""Entity provenance (M0 — audit debt D7).

Provenance records HOW each entity got into the dict. Today every answer
is "explicit" (the user's own words), which is precisely why it is worth
adding now: M1 introduces entities produced by a JOIN rather than by the
query text, and once both kinds coexist, telling them apart after the
fact is impossible without this.

The tests that matter most here are the ones asserting M0 changed
NOTHING: the entity values, and the prompt built from them, must be
byte-identical to before the provenance slot existed.
"""

import pytest

from app.database.models import Advisor
from app.llm import advisor_resolver, entity_extractor
from app.llm.entity_extractor import (
    PROVENANCE_EXPLICIT, _PROVENANCE_KEY, extract_entities, provenance_of,
)
from app.llm.prompt_builder import build_ir_prompt


@pytest.fixture()
def db(db_session):
    db_session.add(Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana",
                           bm="Kaleem Ullah", rm="Kaleem Ullah", zm="Adeel Dogar", portfolio_lead="Adeel Dogar", office="Gulberg BC"))
    db_session.add(Advisor(wid=2, name="Sana Tariq", team="Downtown", company="Agency21"))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


def test_every_extracted_entity_carries_provenance(db):
    """Total by construction — a new extraction cannot be forgotten."""
    entities = extract_entities("top 5 advisors in Blue Area who were late", db)

    marks = entities[_PROVENANCE_KEY]
    tracked = {k for k in entities if k != _PROVENANCE_KEY}
    assert tracked == set(marks), f"unmarked keys: {tracked - set(marks)}"


def test_entities_from_the_query_text_are_marked_explicit(db):
    entities = extract_entities("how is Blue Area doing", db)
    assert provenance_of(entities, "team") == PROVENANCE_EXPLICIT
    assert provenance_of(entities, "teams") == PROVENANCE_EXPLICIT


def test_advisor_identity_is_marked_explicit(db):
    entities = extract_entities("tell me about Waqar Haider", db)
    assert provenance_of(entities, "advisor_wid") == PROVENANCE_EXPLICIT
    assert provenance_of(entities, "advisor_name") == PROVENANCE_EXPLICIT


def test_provenance_of_returns_none_for_an_unknown_key(db):
    entities = extract_entities("how is Blue Area doing", db)
    assert provenance_of(entities, "unit_head") is None
    assert provenance_of({}, "team") is None


def test_a_pre_existing_mark_is_not_overwritten(db):
    """M1 tags inferred entities where they are produced and relies on
    the finalising pass leaving those marks alone."""
    entities = {"team": "Blue Area", _PROVENANCE_KEY: {"team": "inferred:advisor:1"}}
    entity_extractor._finalize_provenance(entities)
    assert entities[_PROVENANCE_KEY]["team"] == "inferred:advisor:1"


def test_finalize_is_idempotent():
    entities = {"team": "Blue Area"}
    entity_extractor._finalize_provenance(entities)
    first = dict(entities[_PROVENANCE_KEY])
    entity_extractor._finalize_provenance(entities)
    assert entities[_PROVENANCE_KEY] == first


# ---------------------------------------------------------------------
# No-behaviour-change guarantees
# ---------------------------------------------------------------------

def test_provenance_key_is_the_only_addition_to_the_entity_dict(db):
    """The values themselves must be untouched: M0 adds a slot, not data."""
    entities = extract_entities("top 5 advisors in Blue Area", db)
    del entities[_PROVENANCE_KEY]

    assert entities["team"] == "Blue Area"
    assert entities["teams"] == ["Blue Area"]
    assert entities["limit"] == 5
    assert "_provenance" not in entities


def test_the_llm_prompt_is_unchanged_by_provenance(db):
    """The entity dict is interpolated verbatim into the parser prompt,
    so a meta key that reached it would change every prompt the product
    sends. Underscore keys are filtered — this pins that."""
    entities = extract_entities("top 5 advisors in Blue Area", db)
    without = {k: v for k, v in entities.items() if k != _PROVENANCE_KEY}

    with_prompt = build_ir_prompt("top 5 advisors in Blue Area", ["Blue Area"], ["Graana"], entities)
    without_prompt = build_ir_prompt("top 5 advisors in Blue Area", ["Blue Area"], ["Graana"], without)

    assert with_prompt == without_prompt
    assert "_provenance" not in with_prompt


def test_prompt_still_contains_the_grounded_entities(db):
    """Guarding the filter itself: it must strip META, not content."""
    entities = extract_entities("top 5 advisors in Blue Area", db)
    prompt = build_ir_prompt("top 5 advisors in Blue Area", ["Blue Area"], ["Graana"], entities)
    assert "Entities already found by rule-based grounding" in prompt
    assert "Blue Area" in prompt


def test_an_entity_dict_of_only_meta_adds_no_grounding_line():
    prompt = build_ir_prompt("top 5", ["Blue Area"], ["Graana"], {_PROVENANCE_KEY: {}})
    assert "Entities already found by rule-based grounding" not in prompt
