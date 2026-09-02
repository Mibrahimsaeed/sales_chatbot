"""The relationship grammar, not three example queries.

    "how many advisors that directly report to the Unit Head in AMD"
    "advisors that directly report to the Unit Head in AMD"
    "who reports to unit head in AMD"

all mean one thing: the advisors who report to a Unit Head, that Unit
Head being the one in the team AMD.

THREE THINGS WERE WRONG, and none of them was a missing special case:

1. THE PROMPT CONTRADICTED THE SCHEMA. query_ir documents `subject_of` as
   "the level the target sits BENEATH … 'the Unit Head in AMD' -> scope
   =team AMD, subject_of=unit_head". The prompt defined it as "level of
   the scope entity" — which for that sentence is `team`. Every existing
   example named the manager directly ("advisors under Haseeb"), where
   both readings coincide, so the contradiction was invisible until a
   query named a ROLE inside a GROUP.

2. THE GRAMMAR WAS INCOMPLETE. "reports to", "who reports to" and
   "managed by" appeared in NEITHER the direct list nor the subtree list.
   The model had no rule for the commonest phrasing of the relationship,
   so it guessed, and guessed differently per phrasing.

3. THE SEMANTIC MODEL COULD NOT HOLD THE ANSWER. `Relationship` carried
   only kind and depth; `from_query_ir` read `subject_of` to decide
   whether a read had happened and then dropped it. So even a perfect
   parse lost the MANAGER — leaving a structure indistinguishable from
   "advisors in AMD". `Relationship.of_level` now carries it.

The hermetic tests below pin the contract and the conversion. The `live`
ones are the only evidence about the model itself, and they LOG the
SemanticModel they got, so a failure says what was understood rather than
just that something differed.
"""

import json

import pytest

from app.llm import entity_extractor, semantic_parser
from app.llm.query_ir import QueryIR, Sort, Subject
from app.llm.semantic_model import Relationship, from_query_ir

# One meaning, said many ways. Deliberately more than the three reported:
# a grammar that only works on the reported sentences is a special case
# with extra steps.
DIRECT_PHRASINGS = [
    "how many advisors that directly report to the unit head in AMD",
    "advisors that directly report to the unit head in AMD",
    "who reports to unit head in AMD",
    "who reports to the unit head in AMD",
    "advisors reporting to the unit head in AMD",
    "advisors managed by the unit head in AMD",
    "how many people report to the unit head in AMD",
    "the unit head in AMD's direct reports",
]

# The SAME shape, but containment rather than reporting — these must NOT
# collapse into the group above, or the grammar has just become "any
# hierarchy word means direct".
SUBTREE_PHRASINGS = [
    "advisors under the unit head in AMD",
    "advisors beneath the unit head in AMD",
    "advisors within the unit head in AMD's organisation",
]


def _shape(model, ir=None):
    """The structure under test. `ir` adds the fields that DIAGNOSE a
    failure: a lost scope is either a subject the model never emitted or
    one grounding dropped, and only `subjects`/`missing` separate them."""
    shape = {}
    if ir is not None:
        shape["ir_subjects"] = [(s.type, s.value) for s in ir.subjects]
        shape["ir_missing"] = list(ir.missing)
    shape.update({
        "requested_level": model.requested_level,
        "scope": [(e.name.lower(), e.level) for e in model.scope],
        "relationship": (None if model.relationship is None else
                         (model.relationship.kind, model.relationship.depth,
                          model.relationship.of_level)),
        "is_hierarchy": model.is_hierarchy_query(),
    })
    return shape


# ---------------------------------------------------------------------
# The contract: the model can express a role scoped by a group
# ---------------------------------------------------------------------

def test_the_relationship_carries_the_manager_level():
    """Without `of_level` the manager is dropped and the parse reads as
    "advisors in AMD" — the same structure for a different question."""
    ir = QueryIR(intent="filtered_list", operation="population",
                 subject_level="advisor",
                 subjects=[Subject(type="team", value="AMD")],
                 target_level="advisor", subject_of="unit_head",
                 relation="direct", metric=None, sort=Sort(metric=None))

    model = from_query_ir(ir, level_word="advisor")

    assert model.requested_level == "advisor"
    assert [(e.name, e.level) for e in model.scope] == [("AMD", "team")]
    assert model.relationship == Relationship(kind="membership", depth="direct",
                                              of_level="unit_head")


def test_a_directly_named_manager_leaves_of_level_empty():
    """"advisors under Haseeb" names the manager itself, so repeating its
    level would say nothing — and a spurious of_level would read as a
    role-inside-a-group that was never asked for."""
    ir = QueryIR(intent="filtered_list", operation="population",
                 subject_level="advisor",
                 subjects=[Subject(type="unit_head", value="Haseeb")],
                 target_level="advisor", subject_of="unit_head",
                 relation="subtree", metric=None, sort=Sort(metric=None))

    model = from_query_ir(ir, level_word="advisor")

    assert model.relationship.of_level is None
    assert model.relationship.depth == "subtree"


def test_depth_survives_the_conversion():
    for depth in ("direct", "subtree"):
        ir = QueryIR(intent="filtered_list", operation="population",
                     subject_level="advisor",
                     subjects=[Subject(type="team", value="AMD")],
                     target_level="advisor", subject_of="unit_head",
                     relation=depth, metric=None, sort=Sort(metric=None))
        assert from_query_ir(ir, level_word="advisor").relationship.depth == depth


# ---------------------------------------------------------------------
# The prompt states the grammar
# ---------------------------------------------------------------------

def _prompt():
    from app.llm.prompt_builder import build_ir_prompt
    return build_ir_prompt("who reports to the unit head in AMD",
                           ["AMD"], ["Graana"], grounded_entities={})


@pytest.mark.parametrize("phrase", [
    "reports to", "reporting to", "managed by", "manages",
    "directly reports to", "immediately under",
])
def test_every_reporting_phrase_has_a_rule(phrase):
    """The gap that caused this: three of these appeared in NEITHER list,
    so the commonest phrasing of the relationship had no rule at all."""
    prompt = _prompt()
    direct = prompt.split("Use subtree for CONTAINMENT")[0]
    assert phrase in direct.split("Use direct for REPORTING language")[-1], phrase


@pytest.mark.parametrize("phrase", ["under", "beneath", "within"])
def test_containment_words_stay_subtree(phrase):
    prompt = _prompt()
    subtree = prompt.split("Use subtree for CONTAINMENT")[-1][:400]
    assert phrase in subtree, phrase


def test_the_prompt_defines_subject_of_as_the_manager_level():
    """It used to say "level of the scope entity", which cannot express a
    role inside a group."""
    prompt = _prompt()
    assert "the level the target sits BENEATH" in prompt
    assert "ROLE INSIDE A GROUP" in prompt
    assert "level of the scope entity" not in prompt


def test_the_prompt_says_directly_does_not_create_the_relationship():
    assert "does not\ncreate it" in _prompt() or "does not create it" in _prompt()


# ---------------------------------------------------------------------
# LIVE — what the model actually understands
# ---------------------------------------------------------------------

def _interpret(text, db):
    from app.llm import conversation_memory

    conversation_memory._store.clear()
    entities = entity_extractor.extract_entities(text, db)
    return semantic_parser.interpret(text, entities, db, session_id=None)


@pytest.mark.live
def test_the_model_reads_every_reporting_phrasing_the_same_way(capsys):
    """`pytest -m live`. Logs the SemanticModel for each phrasing so a
    failure states what was understood, not merely that they differed."""
    from app.database.session import SessionLocal

    db = SessionLocal()
    shapes = {}
    try:
        for text in DIRECT_PHRASINGS:
            interpretation = _interpret(text, db)
            assert interpretation.understood, text
            shapes[text] = _shape(interpretation.model, interpretation.ir)
    finally:
        db.close()

    print("\nRAW SEMANTIC MODELS — reporting phrasings")
    for text, shape in shapes.items():
        print(f"  {text}\n      {json.dumps(shape)}")

    for text, shape in shapes.items():
        assert shape["requested_level"] == "advisor", f"{text}: {shape}"
        assert shape["relationship"] is not None, f"{text}: no relationship"
        assert shape["relationship"][1] == "direct", f"{text}: {shape}"
        assert shape["relationship"][2] == "unit_head", f"{text}: manager lost"
        assert ("amd", "team") in shape["scope"], f"{text}: scope lost"

    assert len({json.dumps(s, sort_keys=True) for s in shapes.values()}) == 1, \
        "equivalent phrasings produced different structures"


@pytest.mark.live
def test_containment_phrasings_stay_subtree():
    """The grammar must still SEPARATE the two readings — otherwise it has
    only learned that any hierarchy word means direct."""
    from app.database.session import SessionLocal

    db = SessionLocal()
    shapes = {}
    try:
        for text in SUBTREE_PHRASINGS:
            interpretation = _interpret(text, db)
            assert interpretation.understood, text
            shapes[text] = _shape(interpretation.model, interpretation.ir)
    finally:
        db.close()

    print("\nRAW SEMANTIC MODELS — containment phrasings")
    for text, shape in shapes.items():
        print(f"  {text}\n      {json.dumps(shape)}")

    for text, shape in shapes.items():
        assert shape["relationship"] is not None, f"{text}: no relationship"
        assert shape["relationship"][1] == "subtree", f"{text}: {shape}"
