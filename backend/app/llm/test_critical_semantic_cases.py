"""The four semantic distinctions the parser must not confuse.

These are stated as a contract rather than discovered from failures, and
each is a MINIMAL PAIR: the cases differ by one or two words and mean
different things, so a change that collapses them fails here rather than
in production.

    1. "connects of Blue Area"              subject — the team's own figure
    2. "connects of advisors in Blue Area"  scope   — its members' figures
    3. "teams under Faisal"                 read    — enumerate a level
    4. "who reports directly to Faisal"     read    — an explicit relation

Case 1 is the guard: it is the query that must NOT become a hierarchy
read. Cases 2-4 are the ones that must survive being parsed as one.

WHY THESE RUN THROUGH validate_ir RATHER THAN THROUGH THE MODEL: the
prompt already taught all four shapes — ir_examples has "How many
advisors directly report to Unit Head Ahmed?" with relation="direct" —
and case 4 was still answered with the whole subtree. The parse was
correct and the VALIDATOR discarded it, so a test of the model's output
would have passed throughout. What is pinned here is that a correct
parse survives validation.
"""

import pytest

from app.database.models import Advisor
from app.llm import entity_extractor
from app.llm.ir_validator import validate_ir
from app.llm.query_ir import MetricRef, QueryIR, Sort, Subject
from app.llm.semantic_model import from_query_ir

UNIT_HEAD = "Faisal Hussain Naqvi"


@pytest.fixture()
def org(db_session):
    """One unit head, two teams, three advisors — enough for a real
    grounding pass, which is what supplies level_word/relation_word."""
    db_session.add_all([
        Advisor(wid=1, name="Ahmed Raza", team="Blue Area", company="Graana",
                rm=UNIT_HEAD, portfolio_lead="Bilal Khan", management_lead="Usman Ali"),
        Advisor(wid=2, name="Sara Iqbal", team="Blue Area", company="Graana",
                rm=UNIT_HEAD, portfolio_lead="Bilal Khan", management_lead="Usman Ali"),
        Advisor(wid=3, name="Omar Farooq", team="Downtown", company="Graana",
                rm=UNIT_HEAD, portfolio_lead="Bilal Khan", management_lead="Usman Ali"),
    ])
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _parse(text, db, *, subject_type, subject_value,
           target_level=None, relation="subtree", subject_of=None):
    """The IR a CORRECT parse produces, put through validation.

    The point of the helper is that the model's half is held fixed: every
    case below hands validate_ir the right answer, so a failure can only
    mean validation destroyed it.
    """
    ir = QueryIR(
        intent="breakdown",
        subject_level=subject_type,
        subjects=[Subject(type=subject_type, value=subject_value)],
        target_level=target_level,
        subject_of=subject_of,
        relation=relation,
        metric=MetricRef(key="total_connects"),
        sort=Sort(metric="total_connects"),
    )
    entities = entity_extractor.extract_entities(text, db)
    validated = validate_ir(ir, db, entities=entities).ir
    model = from_query_ir(validated, level_word=entities.get("level_word"))
    return entities, validated, model


# ---------------------------------------------------------------------
# CASE 1 — a named group with no level beneath it is the SUBJECT
# ---------------------------------------------------------------------

def test_case1_connects_of_a_team_is_the_teams_own_figure(org):
    """The regression this whole discriminator exists for: the model
    likes to answer "connects of Blue Area" with the advisors inside it."""
    entities, ir, model = _parse("connects of Blue Area", org,
                                 subject_type="team", subject_value="Blue Area")

    assert entities.get("level_word") is None
    assert entities.get("relation_word") is False

    assert ir.target_level is None
    assert ir.subject_of is None
    assert ir.relation == "subtree"

    assert model.subject is not None and model.subject.name == "Blue Area"
    assert model.subject.level == "team"
    assert model.requested_level is None
    assert not model.is_hierarchy_query()


def test_case1_a_wrongly_claimed_read_is_repaired_away(org):
    """The same sentence with the model's WRONG parse handed in: it must
    be corrected, and the correction recorded rather than done silently."""
    _entities, ir, model = _parse("connects of Blue Area", org,
                                  subject_type="team", subject_value="Blue Area",
                                  target_level="advisor", subject_of="team")

    assert ir.target_level is None
    assert not model.is_hierarchy_query()
    assert any(r.get("field") == "target_level" for r in ir.repairs), \
        "the rewrite must leave a repair record"


# ---------------------------------------------------------------------
# CASE 2 — naming a level makes the group the SCOPE
# ---------------------------------------------------------------------

def test_case2_advisors_in_a_team_makes_the_team_the_scope(org):
    entities, ir, model = _parse("connects of advisors in Blue Area", org,
                                 subject_type="team", subject_value="Blue Area",
                                 target_level="advisor")

    assert entities.get("level_word") == "advisor"

    assert ir.target_level == "advisor"
    assert [s.value for s in ir.subjects] == ["Blue Area"]

    assert model.requested_level == "advisor"
    assert [e.name for e in model.scope] == ["Blue Area"]
    assert model.is_hierarchy_query()


def test_case1_and_case2_differ_by_one_word(org):
    """Stated as a pair, because the failure mode is collapsing them."""
    _e1, _ir1, one = _parse("connects of Blue Area", org,
                            subject_type="team", subject_value="Blue Area")
    _e2, _ir2, two = _parse("connects of advisors in Blue Area", org,
                            subject_type="team", subject_value="Blue Area",
                            target_level="advisor")

    assert one.is_hierarchy_query() is False
    assert two.is_hierarchy_query() is True
    assert one.subject is not None and not one.scope
    assert two.subject is None and two.scope


# ---------------------------------------------------------------------
# CASE 3 — an explicit traversal to a named level
# ---------------------------------------------------------------------

def test_case3_teams_under_a_person_is_a_traversal(org):
    entities, ir, model = _parse(f"teams under {UNIT_HEAD}", org,
                                 subject_type="unit_head", subject_value=UNIT_HEAD,
                                 target_level="team")

    assert entities.get("level_word") == "team"

    assert ir.target_level == "team"
    assert [s.value for s in ir.subjects] == [UNIT_HEAD]

    assert model.requested_level == "team"
    assert [e.name for e in model.scope] == [UNIT_HEAD]
    assert model.is_hierarchy_query()


# ---------------------------------------------------------------------
# CASE 4 — a RELATIONSHIP names the read when no level word does
# ---------------------------------------------------------------------

def test_case4_reports_directly_to_survives_as_a_direct_relation(org):
    """THE DEFECT THIS PHASE FIXED. "reports" is not a level word, so the
    read was demoted to "not a hierarchy read at all": target_level was
    nulled and relation reset direct -> subtree, answering with the whole
    subtree instead of the immediate reports."""
    entities, ir, model = _parse(f"who reports directly to {UNIT_HEAD}", org,
                                 subject_type="unit_head", subject_value=UNIT_HEAD,
                                 target_level="advisor", relation="direct")

    assert entities.get("level_word") is None, "no level noun in this sentence"
    assert entities.get("relation_word") is True, "the RELATION is what names it"

    assert ir.target_level == "advisor"
    assert ir.relation == "direct"
    assert not ir.repairs, "a correct parse must not be rewritten"

    assert model.is_hierarchy_query()
    assert model.relationship is not None
    assert model.relationship.depth == "direct"


@pytest.mark.parametrize("text,depth", [
    ("who reports directly to {name}", "direct"),
    ("direct reports of {name}", "direct"),
    ("who reports to {name}", "subtree"),
    ("who works under {name}", "subtree"),
])
def test_case4_relationship_phrasings_all_survive(text, depth, org):
    """`relation` must carry "directly" as MEANING. Routing used to hinge
    on the literal token, so dropping it re-routed the question entirely;
    these pin that each phrasing keeps its own depth."""
    sentence = text.format(name=UNIT_HEAD)
    entities, ir, model = _parse(sentence, org,
                                 subject_type="unit_head", subject_value=UNIT_HEAD,
                                 target_level="advisor", relation=depth)

    assert entities.get("relation_word") is True, sentence
    assert ir.target_level == "advisor", sentence
    assert ir.relation == depth, sentence
    assert model.is_hierarchy_query(), sentence


def test_a_plain_metric_question_is_still_not_a_read(org):
    """The widened discriminator must not turn ordinary questions into
    hierarchy reads — the failure mode in the other direction."""
    entities, ir, model = _parse("revenue of Blue Area", org,
                                 subject_type="team", subject_value="Blue Area",
                                 target_level="advisor")

    assert entities.get("relation_word") is False
    assert ir.target_level is None
    assert not model.is_hierarchy_query()
