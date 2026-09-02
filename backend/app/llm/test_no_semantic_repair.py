"""The validator no longer rewrites the model's interpretation.

PHASE 11. `ir_validator` used to change what a query MEANT on its way to
the compiler: `_retyped_subject` swapped a subject's type from pre-LLM
extraction, and a second rule nulled the target_level / subject_of /
relation triple. Both produced better answers much of the time, and
neither left any trace in the reply — so a parser that had regressed
looked exactly like a parser that was working.

Those two rewrites are now off on the LLM path. Conflicts surface through
Phase 4 grounding and Phase 5 validation instead, where they are visible
and can be rejected or asked about.

WHAT REMOVING THEM EXPOSED, and why this file exists. `_retyped_subject`
was masking a hazard rather than only overruling the model: with the
level corrected before grounding ran, the loose 0.55 fuzzy floor in
`_ground_subject` never had to cope with a WRONG level. Uncorrected, it
did — and "Blue Area" scores 0.60 against the company "IMARAT", so a
question about a team was answered about a company, with a real number,
and marked valid. That is a worse failure than the one the rewrite fixed,
and it is fixed here in grounding, not by restoring the rewrite.
"""

import pytest

from app.database.models import Advisor, SalesFunnel
from app.llm import entity_extractor, ir_validator, semantic_parser
from app.llm.ir_validator import validate_ir
from app.llm.query_ir import MetricRef, QueryIR, Sort, Subject


@pytest.fixture()
def org(db_session):
    db_session.add_all([
        Advisor(wid=1, name="Ahmed Raza", team="Blue Area", company="Graana"),
        Advisor(wid=2, name="Sara Iqbal", team="Blue Area", company="IMARAT"),
        Advisor(wid=3, name="Omar Farooq", team="Downtown", company="IMARAT"),
    ])
    db_session.add_all([
        SalesFunnel(wid=1, mtd_new_connect=6, mtd_followup_connect=4),
        SalesFunnel(wid=2, mtd_new_connect=1, mtd_followup_connect=1),
        SalesFunnel(wid=3, mtd_new_connect=5, mtd_followup_connect=0),
    ])
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _ir(**kw):
    base = dict(
        intent="filtered_list", operation="group_metric", subject_level="team",
        subjects=[Subject(type="team", value="Blue Area")],
        metric=MetricRef(key="total_connects"), sort=Sort(metric="total_connects"),
    )
    base.update(kw)
    return QueryIR(**base)


def _llm_path(ir, text, db):
    """validate_ir exactly as the LLM path calls it."""
    entities = entity_extractor.extract_entities(text, db)
    return validate_ir(ir, db, entities=entities, allow_semantic_repair=False).ir


# ---------------------------------------------------------------------
# The behaviour that must be preserved
# ---------------------------------------------------------------------

def test_connects_of_blue_area_still_resolves_as_a_team(org):
    """The requirement, with the repair gone: Blue Area is a team and the
    answer is the team's figure, not its members."""
    out = _llm_path(_ir(), "connects of blue area", org)

    assert [(s.type, s.value) for s in out.subjects] == [("team", "Blue Area")]
    assert out.subject_level == "team"
    assert out.target_level is None
    assert not out.is_hierarchy_read()
    assert not out.missing


def test_connects_of_blue_area_executes_to_the_teams_own_total(org):
    """Proven by the number, not the shape: 10 + 2, as one row."""
    from app.llm.query_compiler import compile_and_run

    rows = compile_and_run(org, _llm_path(_ir(), "connects of blue area", org))

    assert len(rows) == 1
    assert rows[0]["name"] == "Blue Area"
    assert rows[0]["value"] == 12


def test_a_typo_in_the_subject_still_grounds(org):
    """Raising the floor must not cost ordinary fuzzy correction — the
    match is still inside the RIGHT gazetteer."""
    out = _llm_path(_ir(subjects=[Subject(type="team", value="blue aera")]),
                    "connects of blue aera", org)

    assert [s.value for s in out.subjects] == ["Blue Area"]
    assert not out.missing


# ---------------------------------------------------------------------
# The rewrites are gone
# ---------------------------------------------------------------------

def test_the_subject_type_is_not_rewritten(org):
    """The model said company. Whatever else happens, it must not be
    quietly turned into a team and answered as though it had said so."""
    out = _llm_path(_ir(subjects=[Subject(type="company", value="Blue Area")],
                        subject_level="company"),
                    "connects of blue area", org)

    assert not any(r.get("field") == "subjects[].type" for r in out.repairs)


def test_a_wrong_level_refuses_instead_of_matching_a_different_entity(org):
    """THE HAZARD THE REWRITE WAS HIDING. "Blue Area" is 0.60 similar to
    the company "IMARAT" — above the legacy 0.55 floor. Answering IMARAT's
    connects to a question about Blue Area is a real number, correctly
    computed, for a question nobody asked."""
    out = _llm_path(_ir(subjects=[Subject(type="company", value="Blue Area")],
                        subject_level="company"),
                    "connects of blue area", org)

    assert [s.value for s in out.subjects] != ["IMARAT"]
    assert out.subjects == [], "the subject is dropped, not substituted"
    assert "subject:company:Blue Area" in out.missing, "and the query is refused"


def test_the_hierarchy_triple_is_not_nulled(org):
    """The model claimed a read. It may be wrong, but the validator no
    longer edits it behind the model's back."""
    out = _llm_path(_ir(target_level="advisor", subject_of="team"),
                    "connects of blue area", org)

    assert out.target_level == "advisor"
    assert out.relation == "subtree"
    assert not any(r.get("field") == "target_level" for r in out.repairs)


def test_the_deterministic_path_keeps_its_repairs(org):
    """Switched off for the LLM, unchanged for the rule-based callers:
    there, a repair is a module keeping its own output self-consistent,
    not one component overruling another."""
    entities = entity_extractor.extract_entities("connects of blue area", org)
    out = validate_ir(_ir(target_level="advisor", subject_of="team"), org,
                      entities=entities).ir

    assert out.target_level is None
    assert any(r.get("field") == "target_level" for r in out.repairs)


# ---------------------------------------------------------------------
# Conflicts are reported instead
# ---------------------------------------------------------------------

def test_an_unsupported_read_is_reported_by_validation(org):
    """Not rewritten, not ignored: the conflict between "enumerate
    advisors" and a sentence naming no level becomes a finding a caller
    can act on."""
    from app.llm import grounding, hierarchy_grounding, semantic_validation
    from app.llm.semantic_model import from_query_ir

    entities = entity_extractor.extract_entities("connects of blue area", org)
    ir = _llm_path(_ir(target_level="advisor", subject_of="team"),
                   "connects of blue area", org)
    model = from_query_ir(ir, level_word=entities.get("level_word"))

    grounded = grounding.ground(model, org)
    hier = hierarchy_grounding.verify(model, grounded, org)
    verdict = semantic_validation.validate(model, grounded, hier, org, entities=entities)

    assert verdict.status == semantic_validation.NEEDS_CLARIFICATION
    finding = next(f for f in verdict.findings
                   if f.field == "requested_level")
    assert finding.severity == semantic_validation.CLARIFY


def test_a_genuine_read_is_not_flagged(org):
    """The check must fire on the absence of a level word, not on reads —
    "advisors in Blue Area" names one and is perfectly supported."""
    from app.llm import grounding, hierarchy_grounding, semantic_validation
    from app.llm.semantic_model import from_query_ir

    text = "connects of advisors in blue area"
    entities = entity_extractor.extract_entities(text, org)
    ir = _llm_path(_ir(target_level="advisor", subject_of="team"), text, org)
    model = from_query_ir(ir, level_word=entities.get("level_word"))

    grounded = grounding.ground(model, org)
    hier = hierarchy_grounding.verify(model, grounded, org)
    verdict = semantic_validation.validate(model, grounded, hier, org, entities=entities)

    assert not [f for f in verdict.findings if f.field == "requested_level"]


def test_a_plain_metric_query_is_not_flagged(org):
    """And it must not fire on the case it exists to protect."""
    from app.llm import grounding, hierarchy_grounding, semantic_validation
    from app.llm.semantic_model import from_query_ir

    entities = entity_extractor.extract_entities("connects of blue area", org)
    model = from_query_ir(_llm_path(_ir(), "connects of blue area", org))

    grounded = grounding.ground(model, org)
    hier = hierarchy_grounding.verify(model, grounded, org)
    verdict = semantic_validation.validate(model, grounded, hier, org, entities=entities)

    assert verdict.status == semantic_validation.VALID
