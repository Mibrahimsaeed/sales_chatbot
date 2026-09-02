"""Grounding verifies the interpretation. It does not reinterpret it.

The contract under test, stated once:

    the LLM decides WHAT was named and WHAT TYPE it is;
    grounding decides WHETHER THAT EXISTS and WHICH RECORD it is.

Most of this file is about the second half of one sentence — "it must not
silently change it to advisor because another match exists". That failure
is invisible in production: a real record comes back, the reply is
well-formed, and it answers a question nobody asked. So the mismatch
cases are pinned harder than the happy path.
"""

import pytest

from app.database.models import Advisor
from app.llm import entity_extractor, grounding
from app.llm.grounding import (
    AMBIGUOUS, COMPARISON, NOT_FOUND, RESOLVED, SCOPE, SUBJECT, TYPE_MISMATCH,
)
from app.llm.semantic_model import EntityRef, SemanticModel


@pytest.fixture()
def org(db_session):
    """A deliberately awkward population:

      - "Blue Area" is a TEAM and also a person's NAME (the retype trap)
      - two people share the name "Yasir Ali" (the duplicate trap)
      - "Ahmed Raza" is unique
    """
    db_session.add_all([
        Advisor(wid=1, name="Ahmed Raza", team="Blue Area", company="Graana",
                rm="Faisal Naqvi", portfolio_lead="Bilal Khan",
                management_lead="Usman Ali", office="F-8", region="Center"),
        Advisor(wid=2, name="Blue Area", team="Downtown", company="Graana",
                rm="Faisal Naqvi", portfolio_lead="Bilal Khan",
                management_lead="Usman Ali", office="F-8", region="Center"),
        Advisor(wid=3, name="Yasir Ali", team="Blue Area", company="IMARAT",
                rm="Faisal Naqvi", portfolio_lead="Bilal Khan",
                management_lead="Usman Ali", office="G-11", region="North"),
        Advisor(wid=4, name="Yasir Ali", team="Downtown", company="IMARAT",
                rm="Faisal Naqvi", portfolio_lead="Bilal Khan",
                management_lead="Usman Ali", office="G-11", region="North"),
    ])
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _ref(name, level=None, stated=None):
    return EntityRef(name=name, level=level,
                     level_was_stated=bool(level) if stated is None else stated)


# ---------------------------------------------------------------------
# THE CONTRACT: a stated type is verified, never traded
# ---------------------------------------------------------------------

def test_a_stated_team_that_exists_resolves_as_a_team(org):
    entity = grounding.ground_entity(_ref("Blue Area", "team"), org)
    assert entity.status == RESOLVED
    assert entity.value == "Blue Area"
    assert entity.resolved.level == "team"


def test_a_stated_team_is_not_retyped_because_a_person_shares_the_name(org):
    """THE CENTRAL GUARANTEE. "Blue Area" is both a team and a person in
    this fixture. The model said team; an advisor match must not win."""
    entity = grounding.ground_entity(_ref("Blue Area", "team"), org)

    assert entity.status == RESOLVED
    assert entity.resolved.level == "team"
    assert entity.wid is None, "a team has no advisor wid — this would be the retype"


def test_a_stated_type_that_does_not_exist_is_a_mismatch_not_a_correction(org):
    """The other half: the model said advisor, and "Blue Area" IS a real
    advisor here — but suppose it says company, where it is not. The
    levels it was found at are REPORTED, and nothing is applied."""
    entity = grounding.ground_entity(_ref("Blue Area", "company"), org)

    assert entity.status == TYPE_MISMATCH
    assert "team" in entity.found_at
    assert entity.resolved is None, "a mismatch must expose no record to read"
    assert entity.value is None and entity.wid is None
    # the interpretation is preserved verbatim for the reader
    assert entity.ref.level == "company"
    assert entity.stated_level == "company"


def test_a_mismatch_keeps_the_alternatives_for_the_pipeline_to_use(org):
    """Reported, not applied — but reported USEFULLY, so a later phase can
    ask "did you mean the team?" without redoing the search."""
    entity = grounding.ground_entity(_ref("Blue Area", "company"), org)

    assert entity.candidates, "the alternatives must survive for the caller"
    assert {c.level for c in entity.candidates} == {"team", "advisor"}


# ---------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------

def test_an_advisor_resolves_to_a_wid(org):
    entity = grounding.ground_entity(_ref("Ahmed Raza", "advisor"), org)
    assert entity.status == RESOLVED
    assert entity.wid == 1


def test_a_management_level_resolves_to_a_name_not_a_wid(org):
    """Unit Heads, Zonal Heads and BCMs have no identifier of their own —
    the hierarchy table stores them by name in their column. The canonical
    VALUE is the identifier for those levels, and a None wid here is the
    data model, not a gap."""
    for level, value in (("unit_head", "Faisal Naqvi"),
                         ("zonal_head", "Bilal Khan"),
                         ("bcm", "Usman Ali")):
        entity = grounding.ground_entity(_ref(value, level), org)
        assert entity.status == RESOLVED, level
        assert entity.value == value, level
        assert entity.wid is None, level


def test_a_company_resolves(org):
    assert grounding.ground_entity(_ref("Graana", "company"), org).status == RESOLVED


# ---------------------------------------------------------------------
# Ambiguity — never resolved by picking
# ---------------------------------------------------------------------

def test_duplicate_people_are_ambiguous_not_a_coin_flip(org):
    """Two Yasir Alis. The old lookup returned the lowest wid."""
    entity = grounding.ground_entity(_ref("Yasir Ali", "advisor"), org)

    assert entity.status == AMBIGUOUS
    assert {c.wid for c in entity.candidates} == {3, 4}
    assert entity.resolved is None, "no single record may be readable"
    assert entity.wid is None


def test_no_stated_level_and_several_levels_match_is_ambiguous(org):
    """The model declined to say what type this is, and the name is real
    at two levels. Choosing one here would invent the interpretation the
    model withheld."""
    entity = grounding.ground_entity(_ref("Blue Area", None), org)

    assert entity.status == AMBIGUOUS
    assert set(entity.found_at) == {"team", "advisor"}
    assert entity.resolved is None


def test_no_stated_level_with_one_match_resolves(org):
    entity = grounding.ground_entity(_ref("Ahmed Raza", None), org)
    assert entity.status == RESOLVED
    assert entity.resolved.level == "advisor"


# ---------------------------------------------------------------------
# Absence, typos, aliases
# ---------------------------------------------------------------------

def test_a_nonexistent_entity_is_not_found(org):
    entity = grounding.ground_entity(_ref("Qwerty Zzz", "team"), org)
    assert entity.status == NOT_FOUND
    assert not entity.candidates and entity.resolved is None


def test_not_found_is_an_answer_not_a_question(org):
    """Nothing to choose between, so it must not be routed to
    clarification the way an ambiguity is."""
    result = grounding.Grounding(entities=[
        grounding.ground_entity(_ref("Qwerty Zzz", "team"), org)])
    assert result.not_found
    assert not result.needs_clarification


def test_a_typo_still_grounds_through_the_existing_fuzzy_tier(org):
    entity = grounding.ground_entity(_ref("Blue Aera", "team"), org)
    assert entity.status == RESOLVED
    assert entity.value == "Blue Area"
    assert entity.resolved.score < 1.0, "a typo is not an exact match"


@pytest.mark.parametrize("alias,canonical", [
    ("management_lead", "bcm"),
    ("management lead", "bcm"),
    ("portfolio_lead", "zonal_head"),
    ("portfolio lead", "zonal_head"),
    ("rm", "unit_head"),
    ("regional manager", "unit_head"),
    ("business center manager", "bcm"),
    ("team", "team"),
    ("company", "company"),
])
def test_supported_level_aliases_canonicalise(alias, canonical):
    """The alias vocabulary is read from relations.role_alias_pairs(),
    which already owns it for the rule planner — registering one there
    must be enough."""
    assert grounding.canonical_level(alias) == canonical


# ---------------------------------------------------------------------
# Whole-interpretation grounding
# ---------------------------------------------------------------------

def _model(**kw):
    kw.setdefault("operation", "group_metric")
    return SemanticModel(**kw)


def test_ground_walks_subject_scope_and_comparisons(org):
    result = grounding.ground(_model(
        subject=_ref("Blue Area", "team"),
        scope=[_ref("Graana", "company")],
        comparison_subjects=[_ref("Ahmed Raza", "advisor")],
    ), org)

    assert [e.role for e in result.entities] == [SUBJECT, SCOPE, COMPARISON]
    assert result.is_fully_grounded
    assert not result.needs_clarification
    assert result.subject.value == "Blue Area"


def test_ground_reports_a_mixed_result_without_dropping_anything(org):
    result = grounding.ground(_model(
        subject=_ref("Yasir Ali", "advisor"),      # ambiguous
        scope=[_ref("Qwerty Zzz", "team"),         # not found
               _ref("Blue Area", "company")],      # type mismatch
    ), org)

    assert len(result.entities) == 3, "every named entity is reported"
    assert not result.is_fully_grounded
    assert result.needs_clarification
    assert [e.name for e in result.ambiguous] == ["Yasir Ali"]
    assert [e.name for e in result.not_found] == ["Qwerty Zzz"]
    assert [e.name for e in result.mismatched] == ["Blue Area"]


def test_grounding_does_not_mutate_the_interpretation(org):
    """The model and its verification stay separable — that is what makes
    a mismatch reviewable instead of invisible."""
    model = _model(subject=_ref("Blue Area", "company"))
    before = model.model_dump()

    grounding.ground(model, org)

    assert model.model_dump() == before


def test_an_interpretation_naming_nothing_grounds_to_nothing(org):
    result = grounding.ground(_model(), org)
    assert not result.entities
    assert result.is_fully_grounded, "vacuously — there was nothing to ground"
    assert not result.needs_clarification


def test_the_report_is_serialisable_for_tracing(org):
    result = grounding.ground(_model(subject=_ref("Yasir Ali", "advisor")), org)
    payload = result.to_dict()

    assert payload["needs_clarification"] is True
    entity = payload["entities"][0]
    assert entity["status"] == AMBIGUOUS
    assert entity["stated_level"] == "advisor"
    assert len(entity["candidates"]) == 2
