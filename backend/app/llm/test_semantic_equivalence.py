"""Equivalent phrasings must produce equivalent SemanticModels.

    "unit head in Blue Area"  ==  "unit heads of Blue Area"
    ==  "Blue Area unit head"  ==  "unit head for Blue Area"

WHY THIS IS A PROMPT PROBLEM AND NOT A PARSING ONE. Measured against the
real model, the deterministic layer was already surface-form insensitive:
all five phrasings ground "Blue Area" to the same team and detect the
same `level_word`. Five phrasings still produced FOUR different
SemanticModels, so the divergence was entirely in what the model
returned. That is why the fix is in the prompt and nothing here repairs a
parse after the fact — a normaliser in deterministic code would have hidden
the problem rather than fixed it, and would have violated the rule that
the interpretation is the model's to make.

TWO LAYERS ARE TESTED, DELIBERATELY:

  - hermetic (always): the deterministic half really is blind to surface
    form, and the same model output maps to the same SemanticModel
    whichever phrasing produced it. These would pass even if the prompt
    change did nothing, and they are not evidence that it worked.
  - live (opt-in, `pytest -m live`): the real model returns equivalent
    interpretations. THIS is the test whose subject is the fix. It costs
    money, so it is deselected by default — but a claim that equivalent
    queries now agree rests on this one, not on the hermetic tests.
"""

import pytest

from app.database.models import Advisor
from app.llm import entity_extractor, semantic_parser
from app.llm.query_ir import MetricRef, QueryIR, Sort, Subject
from app.llm.semantic_model import from_query_ir

# The groups under test. Each list is one meaning, said several ways.
UNIT_HEAD_GROUP = [
    "unit head in blue area",
    "unit heads in blue area",
    "blue area unit head",
    "unit heads of blue area",
    "unit head for blue area",
]
METRIC_GROUP = [
    "connects of blue area",
    "blue area connects",
    "connects for blue area",
]
ADVISOR_SCOPE_GROUP = [
    "connects of advisors in blue area",
    "connects for advisors in blue area",
]

ALL_GROUPS = [
    pytest.param(UNIT_HEAD_GROUP, "unit_head", id="unit_head_in_team"),
    pytest.param(METRIC_GROUP, None, id="metric_of_team"),
    pytest.param(ADVISOR_SCOPE_GROUP, "advisor", id="advisors_in_team"),
]


@pytest.fixture()
def org(db_session):
    db_session.add_all([
        Advisor(wid=1, name="Ahmed Raza", team="Blue Area", company="Graana",
                rm="Zeeshan Tariq", portfolio_lead="Bilal Khan",
                management_lead="Usman Ali"),
        Advisor(wid=2, name="Sara Iqbal", team="Blue Area", company="Graana",
                rm="Zeeshan Tariq", portfolio_lead="Bilal Khan",
                management_lead="Usman Ali"),
        Advisor(wid=3, name="Omar Farooq", team="Downtown", company="IMARAT",
                rm="Hasan Danish", portfolio_lead="Bilal Khan",
                management_lead="Usman Ali"),
    ])
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _shape(model):
    """The eight properties that must agree across a group."""
    return {
        "operation": model.operation,
        "metrics": [m.name for m in model.metrics],
        "subject": (model.subject.name.lower(), model.subject.level) if model.subject else None,
        "subject_level": model.subject_level,
        "scope": [(e.name.lower(), e.level) for e in model.scope],
        "requested_level": model.requested_level,
        "relationship": ((model.relationship.kind, model.relationship.depth)
                         if model.relationship else None),
        "is_hierarchy": model.is_hierarchy_query(),
    }


# ---------------------------------------------------------------------
# HERMETIC — the deterministic half is blind to surface form
# ---------------------------------------------------------------------

@pytest.mark.parametrize("group,expected_level", ALL_GROUPS)
def test_grounding_and_level_word_agree_across_a_group(group, expected_level, org):
    """Every phrasing must reach the model with the same evidence. If this
    ever diverges, the prompt cannot be blamed for what comes back."""
    seen = {
        phrasing: (
            tuple(entity_extractor.extract_entities(phrasing, org).get("teams") or ()),
            entity_extractor.extract_entities(phrasing, org).get("level_word"),
        )
        for phrasing in group
    }
    distinct = set(seen.values())

    assert len(distinct) == 1, f"deterministic layer diverged: {seen}"
    teams, level_word = distinct.pop()
    assert teams == ("Blue Area",)
    assert level_word == expected_level


@pytest.mark.parametrize("group,_level", ALL_GROUPS)
def test_the_same_model_output_maps_to_the_same_semantic_model(group, _level, org):
    """Surface form must not change the CONVERSION either.

    Holds the model's answer fixed and varies only the wording, so a
    failure here would mean deterministic code was reading the sentence
    and reinterpreting the parse — which this phase forbids.
    """
    ir_kwargs = dict(
        intent="filtered_list", operation="population", subject_level="team",
        subjects=[Subject(type="team", value="Blue Area")],
        target_level="unit_head", relation="subtree",
        metric=MetricRef(key="total_connects"), sort=Sort(metric="total_connects"),
    )
    shapes = {}
    for phrasing in group:
        entities = entity_extractor.extract_entities(phrasing, org)
        model = from_query_ir(QueryIR(**ir_kwargs), level_word=entities.get("level_word"))
        shapes[phrasing] = _shape(model)

    distinct = {repr(s) for s in shapes.values()}
    assert len(distinct) == 1, f"conversion diverged by wording: {shapes}"


def test_a_metric_query_is_still_not_a_hierarchy_read(org):
    """Requirement 7, pinned against the change: adding the level-word
    rule must not turn "connects of Blue Area" into an enumeration."""
    entities = entity_extractor.extract_entities("connects of blue area", org)
    model = from_query_ir(
        QueryIR(intent="filtered_list", operation="group_metric", subject_level="team",
                subjects=[Subject(type="team", value="Blue Area")],
                metric=MetricRef(key="total_connects"), sort=Sort(metric="total_connects")),
        level_word=entities.get("level_word"))

    assert model.subject is not None and model.subject.name == "Blue Area"
    assert model.requested_level is None
    assert not model.is_hierarchy_query()


# ---------------------------------------------------------------------
# The prompt states the rule
# ---------------------------------------------------------------------

def test_the_prompt_states_that_surface_form_is_not_meaning():
    from app.llm.prompt_builder import build_ir_prompt

    prompt = build_ir_prompt("unit head in blue area", ["Blue Area"], ["Graana"],
                             grounded_entities={})

    assert "SURFACE FORM IS NOT MEANING" in prompt
    assert "in/of/for/under" in prompt
    assert "`target_level`" in prompt


def test_the_prompt_illustrates_with_values_that_are_not_real():
    """THE MEASURED DEFECT. The worked example used a real team ("AMD")
    and a real Unit Head, and the model returned that team as the SCOPE
    of a question about a different one — a grounded, plausible, wrong
    answer. Illustrative values must not be resolvable against the data.
    """
    from app.llm.prompt_builder import build_ir_prompt

    prompt = build_ir_prompt("unit head in blue area", ["Blue Area"], ["Graana"],
                             grounded_entities={})

    assert "AMD" not in prompt
    assert "Faisal" not in prompt


# ---------------------------------------------------------------------
# LIVE — the test whose subject is the model itself
# ---------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.parametrize("group,_level", ALL_GROUPS)
def test_the_real_model_returns_equivalent_interpretations(group, _level):
    """`pytest -m live`. Costs money; this is the only test that can show
    the prompt change worked."""
    from app.database.session import SessionLocal
    from app.llm import conversation_memory

    db = SessionLocal()
    try:
        shapes = {}
        for phrasing in group:
            conversation_memory._store.clear()
            entities = entity_extractor.extract_entities(phrasing, db)
            interpretation = semantic_parser.interpret(phrasing, entities, db,
                                                       session_id=None)
            assert interpretation.understood, phrasing
            shapes[phrasing] = _shape(interpretation.model)
    finally:
        db.close()

    distinct = {repr(s) for s in shapes.values()}
    assert len(distinct) == 1, (
        "the model interprets equivalent phrasings differently:\n"
        + "\n".join(f"  {q}: {s}" for q, s in shapes.items()))
