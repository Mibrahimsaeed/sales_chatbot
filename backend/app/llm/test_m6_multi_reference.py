"""Multiple independent references in one query (M6).

The four defects `test_multi_reference_characterisation.py` recorded from
the failing side, asserted from the working side, plus the properties
that keep them fixed.

The central idea M6 adds: a person named in order to REACH something
("compare X's team with ...") is a reference SOURCE, not one of the
things being compared. Distinguishing those two roles is what lets a
two-sided question have exactly two sides.
"""

import pytest

from app.core.config import settings
from app.database.models import Advisor
from app.llm import advisor_resolver, entity_extractor, reference_parser
from app.llm.entity_extractor import PROVENANCE_KEY, REFERENCE_SOURCES_KEY


@pytest.fixture()
def db(db_session):
    db_session.add(Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana"))
    db_session.add(Advisor(wid=2, name="Sana Tariq", team="Downtown", company="Agency21"))
    db_session.add(Advisor(wid=3, name="Imran Butt", team="Downtown", company="Agency21"))
    db_session.add(Advisor(wid=4, name="Yasir Ali", team="Blue Area", company="Graana"))
    db_session.add(Advisor(wid=5, name="Yasir Ali", team="Downtown", company="Agency21"))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


@pytest.fixture(autouse=True)
def inference_on(monkeypatch):
    monkeypatch.setattr(settings, "relation_inference_enabled", True)
    monkeypatch.setattr(settings, "relation_inference_levels", "team,company")


# ---------------------------------------------------------------------
# Defect 1 — span isolation
# ---------------------------------------------------------------------

def test_each_source_span_is_independent():
    references = reference_parser.parse("Compare Waqar Haider's team with Sana Tariq's team")
    assert "waqar haider" in references[0].source_span.lower()
    assert "Haider" not in references[1].source_span
    assert "sana tariq" in references[1].source_span.lower()


def test_three_references_stay_separate():
    references = reference_parser.parse(
        "Compare Waqar Haider's team with Sana Tariq's team and Imran Butt's team")
    assert len(references) == 3
    assert "Sana" not in references[2].source_span
    assert "imran butt" in references[2].source_span.lower()


def test_a_single_reference_span_is_unchanged():
    """Backward compatibility: one reference behaves exactly as in M1."""
    [reference] = reference_parser.parse("How is Waqar Haider's team doing")
    assert "waqar haider" in reference.source_span.lower()


# ---------------------------------------------------------------------
# Defect 2 — each reference resolves its own person
# ---------------------------------------------------------------------

def test_two_references_resolve_to_two_different_people(db):
    entities = entity_extractor.extract_entities(
        "Compare Waqar Haider's team with Sana Tariq's team", db)
    assert entities["teams"] == ["Blue Area", "Downtown"]


def test_two_company_references_resolve_independently(db):
    entities = entity_extractor.extract_entities(
        "Compare Waqar Haider's company with Sana Tariq's company", db)
    assert entities["companies"] == ["Graana", "Agency21"]


def test_references_at_different_levels_both_bind(db):
    entities = entity_extractor.extract_entities(
        "Compare Waqar Haider's team with Sana Tariq's company", db)
    assert entities["teams"] == ["Blue Area"]
    assert entities["companies"] == ["Agency21"]


def test_two_people_on_the_same_team_do_not_produce_a_self_comparison(db):
    """Sana and Imran share a team. Binding it twice would compare
    Downtown with itself."""
    entities = entity_extractor.extract_entities(
        "Compare Sana Tariq's team with Imran Butt's team", db)
    assert entities["teams"] == ["Downtown"]


# ---------------------------------------------------------------------
# Defect 3 — explicit and inferred coexist
# ---------------------------------------------------------------------

def test_an_explicit_group_no_longer_suppresses_the_inferred_one(db):
    entities = entity_extractor.extract_entities(
        "Compare Waqar Haider's team with Downtown", db)
    assert set(entities["teams"]) == {"Downtown", "Blue Area"}


def test_the_explicit_value_keeps_the_primary_slot(db):
    """Requirement 6: precedence is expressed through the singular key."""
    entities = entity_extractor.extract_entities(
        "Compare Waqar Haider's team with Downtown", db)
    assert entities["team"] == "Downtown"
    assert entity_extractor.provenance_of(entities, "team") == entity_extractor.PROVENANCE_EXPLICIT


def test_a_purely_inferred_key_is_marked_inferred(db):
    entities = entity_extractor.extract_entities("How is Waqar Haider's team doing", db)
    assert entity_extractor.provenance_of(entities, "team") == "inferred:advisor:1"


# ---------------------------------------------------------------------
# Defect 4 — in-message pronoun
# ---------------------------------------------------------------------

def test_an_in_message_pronoun_binds_to_the_named_person(db):
    entities = entity_extractor.extract_entities(
        "How does Waqar Haider compare to his team", db)
    assert entities["team"] == "Blue Area"
    assert entity_extractor.provenance_of(entities, "team") == "inferred:advisor:1"


def test_an_in_message_pronoun_needs_exactly_one_named_person(db):
    """Two people named — the pronoun is genuinely ambiguous."""
    entities = entity_extractor.extract_entities(
        "Compare Waqar Haider and Sana Tariq to his team", db)
    assert entities.get("team") in (None, "Downtown")  # never Blue Area by a guess


def test_an_ambiguous_name_never_binds(db):
    """Two people are called Yasir Ali."""
    entities = entity_extractor.extract_entities("How does Yasir Ali compare to his team", db)
    assert entities.get("team") is None


# ---------------------------------------------------------------------
# Reference SOURCE vs comparison TARGET
# ---------------------------------------------------------------------

def test_a_possessive_source_is_recorded_as_a_source(db):
    entities = entity_extractor.extract_entities(
        "Compare Waqar Haider's team with Downtown", db)
    assert entities[REFERENCE_SOURCES_KEY] == [1]


def test_a_pronoun_source_is_not_recorded_as_a_source(db):
    """"How does X compare to his team" names X as one SIDE of the
    comparison, not merely as a route to the team."""
    entities = entity_extractor.extract_entities(
        "How does Waqar Haider compare to his team", db)
    assert entities.get(REFERENCE_SOURCES_KEY) in (None, [])


def test_reference_sources_are_meta_and_carry_no_provenance(db):
    entities = entity_extractor.extract_entities(
        "Compare Waqar Haider's team with Downtown", db)
    assert REFERENCE_SOURCES_KEY not in entities[PROVENANCE_KEY]


def test_meta_keys_never_reach_the_llm_prompt(db):
    from app.llm.prompt_builder import build_ir_prompt

    entities = entity_extractor.extract_entities(
        "Compare Waqar Haider's team with Downtown", db)
    prompt = build_ir_prompt("q", ["Blue Area"], ["Graana"], entities)
    assert REFERENCE_SOURCES_KEY not in prompt
    assert PROVENANCE_KEY not in prompt


# ---------------------------------------------------------------------
# Planner target selection
# ---------------------------------------------------------------------

def test_a_reference_source_is_not_a_comparison_target(db):
    from app.llm.query_planner import score_intents

    entities = entity_extractor.extract_entities(
        "Compare Waqar Haider's team with Sana Tariq's team", db)
    ctx, _ = score_intents("compare waqar haider's team with sana tariq's team", entities)
    assert ctx.comparison_targets() == [("team", "Blue Area"), ("team", "Downtown")]


def test_a_named_person_is_a_comparison_target_when_not_a_source(db):
    from app.llm.query_planner import score_intents

    entities = entity_extractor.extract_entities(
        "How does Waqar Haider compare to his team", db)
    ctx, _ = score_intents("how does waqar haider compare to his team", entities)
    assert ctx.comparison_targets() == [("advisor", "Waqar Haider"), ("team", "Blue Area")]


def test_group_only_comparisons_are_unchanged(db):
    from app.llm.query_planner import score_intents

    entities = entity_extractor.extract_entities("Compare Blue Area with Downtown", db)
    ctx, _ = score_intents("compare blue area with downtown", entities)
    assert ctx.comparison_targets() == ctx.all_group_entities()
