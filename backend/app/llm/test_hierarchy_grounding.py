"""Hierarchy grounding and validation, over the six shapes that matter.

    normal metric query · hierarchy query · direct hierarchy query
    invalid hierarchy   · wrong entity type · ambiguous entity

Two properties are pinned throughout, because they are the ones a later
change is most likely to erode:

  1. A METRIC QUERY IS NOT A TRAVERSAL. "connects of Blue Area" names a
     team and asks for its figure. A team having members is not a reason
     to enumerate them, and no amount of grounding may invent that.

  2. NOTHING IS REWRITTEN TO MAKE IT RUN. An unsupported relationship, an
     unknown metric, a name that exists as another type — each is
     rejected or sent for clarification with the interpretation intact.
     A repaired query answers a question nobody asked, and the reply
     carries no sign that it happened.
"""

import pytest

from app.database.models import Advisor
from app.llm import entity_extractor, grounding, hierarchy_grounding, semantic_validation
from app.llm.hierarchy_grounding import EMPTY, NOT_APPLICABLE, UNSUPPORTED, VERIFIED
from app.llm.query_ir import MetricRef, QueryIR, Sort, Subject
from app.llm.semantic_model import EntityRef, SemanticModel, from_query_ir
from app.llm.semantic_validation import (
    CLARIFY, ENTITY, ENTITY_TYPE, HIERARCHY, INVALID, METRIC,
    NEEDS_CLARIFICATION, REJECT, VALID,
)

UNIT_HEAD = "Faisal Naqvi"
ZONAL = "Bilal Khan"
BCM = "Usman Ali"


@pytest.fixture()
def org(db_session):
    """A small org with the awkward cases built in.

    Directness is explicit: an advisor is an IMMEDIATE report of whoever
    holds their `management_lead`, so Sara reports directly to the Unit
    Head while Ahmed reaches him only through Usman Ali.
    """
    db_session.add_all([
        # in Faisal's subtree, but NOT a direct report of his
        Advisor(wid=1, name="Ahmed Raza", team="Blue Area", company="Graana",
                rm=UNIT_HEAD, portfolio_lead=ZONAL, management_lead=BCM),
        # a DIRECT advisor report: the Unit Head is her own management_lead
        Advisor(wid=2, name="Sara Iqbal", team="Blue Area", company="Graana",
                rm=UNIT_HEAD, portfolio_lead=UNIT_HEAD, management_lead=UNIT_HEAD),
        # duplicate name -> ambiguity
        Advisor(wid=3, name="Yasir Ali", team="Downtown", company="IMARAT",
                rm=UNIT_HEAD, portfolio_lead=ZONAL, management_lead=BCM),
        Advisor(wid=4, name="Yasir Ali", team="Blue Area", company="IMARAT",
                rm=UNIT_HEAD, portfolio_lead=ZONAL, management_lead=BCM),
        # a PERSON named after a team -> wrong-entity-type material
        Advisor(wid=5, name="Blue Area", team="Downtown", company="Graana",
                rm=UNIT_HEAD, portfolio_lead=ZONAL, management_lead=BCM),
        # THE BCM IS ALSO AN ADVISOR ROW, in Blue Area. Every person in
        # this organisation is an advisor; holding a management role does
        # not remove them from the roster of the team they sit in.
        # He is his own management_lead, which is ordinary in this data
        # and is why direct_scope_filter excludes self.
        Advisor(wid=6, name=BCM, team="Blue Area", company="Graana",
                rm=UNIT_HEAD, portfolio_lead=ZONAL, management_lead=BCM),
    ])
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _model_from_ir(**kw):
    """Build the SemanticModel the way the pipeline does — through
    from_query_ir — so these test the shapes the parser really produces
    rather than a hand-assembled model that cannot occur."""
    kw.setdefault("intent", "breakdown")
    kw.setdefault("metric", MetricRef(key="total_connects"))
    kw.setdefault("sort", Sort(metric="total_connects"))
    level_word = kw.pop("level_word", None)
    return from_query_ir(QueryIR(**kw), level_word=level_word)


def _run(model, db, principal=None):
    grounded = grounding.ground(model, db)
    hier = hierarchy_grounding.verify(model, grounded, db)
    verdict = semantic_validation.validate(model, grounded, hier, db, principal=principal)
    return grounded, hier, verdict


# ---------------------------------------------------------------------
# 1. NORMAL METRIC QUERY — and the traversal it must not become
# ---------------------------------------------------------------------

def test_normal_metric_query_is_not_a_hierarchy_read(org):
    """"connects of Blue Area" — the team's own figure."""
    model = _model_from_ir(subject_level="team",
                           subjects=[Subject(type="team", value="Blue Area")])
    _grounded, hier, verdict = _run(model, org)

    assert hier.status == NOT_APPLICABLE
    assert hier.member_count == 0, "nothing may be enumerated"
    assert verdict.status == VALID


def test_a_team_having_members_is_not_a_reason_to_list_them(org):
    """The team genuinely has members — that is precisely why this is the
    tempting mistake."""
    model = _model_from_ir(subject_level="team",
                           subjects=[Subject(type="team", value="Blue Area")])
    _grounded, hier, _verdict = _run(model, org)

    members = hierarchy_grounding._members_at(org, "team", "Blue Area", "advisor", "subtree")
    assert len(members) == 4, "the members exist"
    assert hier.status == NOT_APPLICABLE, "and are still not enumerated"


# ---------------------------------------------------------------------
# 2. HIERARCHY QUERY
# ---------------------------------------------------------------------

def test_hierarchy_query_resolves_the_actual_members(org):
    """"advisors in Blue Area" — the team is the scope, advisors the
    output, and the members come from the data rather than from a
    declared chain."""
    model = _model_from_ir(subject_level="team",
                           subjects=[Subject(type="team", value="Blue Area")],
                           target_level="advisor", level_word="advisor")
    _grounded, hier, verdict = _run(model, org)

    assert hier.status == VERIFIED
    assert hier.subject_value == "Blue Area"
    assert hier.target_level == "advisor"
    assert set(hier.members) == {"Ahmed Raza", "Sara Iqbal", "Yasir Ali", BCM}
    assert verdict.status == VALID


def test_every_person_is_an_advisor_including_the_managers(org):
    """"advisors in Blue Area" must not exclude people who also hold a
    management role. The table is advisor-centric: Usman Ali is the BCM
    of three people AND an advisor sitting in Blue Area, so a roster of
    that team includes him. Filtering managers out would under-report the
    team by exactly the people running it.

    Confirmed against production, where "advisors in Blue Area" returns
    48 people including 2 Unit Heads, 4 Zonal Heads and 10 BCMs.
    """
    members = hierarchy_grounding._members_at(org, "team", "Blue Area",
                                              "advisor", "subtree")

    assert BCM in members, "the BCM is an advisor in this team too"
    assert hierarchy_grounding.highest_level_of(BCM, org) == "bcm", \
        "and is still a BCM when asked about as a person"


def test_a_persons_highest_level_governs(org):
    """"Asked about a person" answers at their senior-most role. The Unit
    Head is also somebody's management_lead here, and answering from that
    junior role would describe a scope of one."""
    assert hierarchy_grounding.highest_level_of(UNIT_HEAD, org) == "unit_head"
    assert hierarchy_grounding.highest_level_of(BCM, org) == "bcm"
    assert hierarchy_grounding.highest_level_of("Ahmed Raza", org) == "advisor"


# ---------------------------------------------------------------------
# 3. DIRECT HIERARCHY QUERY
# ---------------------------------------------------------------------

def test_direct_relationship_is_narrower_than_the_subtree(org):
    """"who reports directly to Faisal" — Sara only. Ahmed is beneath him
    but reaches him through a BCM."""
    model = _model_from_ir(subject_level="unit_head",
                           subjects=[Subject(type="unit_head", value=UNIT_HEAD)],
                           target_level="advisor", relation="direct",
                           level_word="advisor")
    _grounded, hier, verdict = _run(model, org)

    assert hier.status == VERIFIED
    assert hier.relation == "direct"
    assert hier.members == ["Sara Iqbal"]
    assert verdict.status == VALID


def test_subtree_and_direct_disagree_and_both_are_right(org):
    """The pair, stated together: this is the distinction `relation`
    exists to carry, and it was answered identically before it existed."""
    subtree = hierarchy_grounding._members_at(org, "unit_head", UNIT_HEAD, "advisor", "subtree")
    direct = hierarchy_grounding._members_at(org, "unit_head", UNIT_HEAD, "advisor", "direct")

    assert len(subtree) == 6
    assert direct == ["Sara Iqbal"]


def test_an_empty_relationship_is_a_true_answer_not_an_error(org):
    """No advisor names the Zonal Head as their immediate manager — they
    reach him through a BCM. That is a fact about the organisation: it
    must not be an error, and it must not silently descend to the level
    that DOES have members."""
    model = _model_from_ir(subject_level="zonal_head",
                           subjects=[Subject(type="zonal_head", value=ZONAL)],
                           target_level="advisor", relation="direct",
                           level_word="advisor")
    _grounded, hier, verdict = _run(model, org)

    assert hier.status == EMPTY
    assert hier.member_count == 0
    assert hier.reason and "no Advisor found beneath" in hier.reason
    assert verdict.status == VALID, "empty is executable; it answers zero"


# ---------------------------------------------------------------------
# 4. INVALID HIERARCHY
# ---------------------------------------------------------------------

def test_an_attribute_has_nothing_beneath_it(org):
    """company/office/region describe entities, they do not nest. A
    traversal built on one returns the wrong population rather than
    failing, so it is refused here."""
    model = _model_from_ir(subject_level="unit_head",
                           subjects=[Subject(type="unit_head", value=UNIT_HEAD)],
                           target_level="company", level_word="company")
    _grounded, hier, verdict = _run(model, org)

    assert hier.status == UNSUPPORTED
    assert "attribute" in hier.reason
    assert verdict.status == INVALID
    assert [f.check for f in verdict.rejections] == [HIERARCHY]


def test_nothing_reports_to_an_advisor(org):
    """The leaf has no reports."""
    model = _model_from_ir(subject_level="advisor",
                           subjects=[Subject(type="advisor", value="Ahmed Raza")],
                           target_level="team", level_word="team")
    _grounded, hier, verdict = _run(model, org)

    assert hier.status == UNSUPPORTED
    assert "leaf" in hier.reason
    assert verdict.status == INVALID


def test_an_invalid_hierarchy_is_rejected_not_repaired(org):
    """The contract: no substitution of a level that would have worked."""
    model = _model_from_ir(subject_level="unit_head",
                           subjects=[Subject(type="unit_head", value=UNIT_HEAD)],
                           target_level="company", level_word="company")
    before = model.model_dump()

    _grounded, hier, verdict = _run(model, org)

    assert model.model_dump() == before, "the interpretation must survive intact"
    assert hier.target_level == "company", "still what was asked for"
    assert not verdict.is_executable


# ---------------------------------------------------------------------
# 5. WRONG ENTITY TYPE
# ---------------------------------------------------------------------

def test_wrong_entity_type_asks_rather_than_retyping(org):
    """"Blue Area" is a team and, in this fixture, also a person. Read as
    a company it is neither — and the answer is a question, not a quiet
    switch to the reading that exists."""
    model = SemanticModel(operation="group_metric",
                          subject=EntityRef(name="Blue Area", level="company",
                                            level_was_stated=True))
    grounded, _hier, verdict = _run(model, org)

    assert grounded.mismatched
    assert verdict.status == NEEDS_CLARIFICATION
    finding = next(f for f in verdict.findings if f.check == ENTITY_TYPE)
    assert finding.severity == CLARIFY
    assert "team" in finding.message and "company" in finding.message


def test_a_nonexistent_entity_is_rejected_because_there_is_nothing_to_ask(org):
    model = SemanticModel(operation="group_metric",
                          subject=EntityRef(name="Qwerty Zzz", level="team",
                                            level_was_stated=True))
    _grounded, _hier, verdict = _run(model, org)

    assert verdict.status == INVALID
    assert [f.check for f in verdict.rejections] == [ENTITY]


# ---------------------------------------------------------------------
# 6. AMBIGUOUS ENTITY
# ---------------------------------------------------------------------

def test_an_ambiguous_person_needs_clarification(org):
    """Two Yasir Alis. Neither is chosen."""
    model = SemanticModel(operation="group_metric",
                          subject=EntityRef(name="Yasir Ali", level="advisor",
                                            level_was_stated=True))
    grounded, _hier, verdict = _run(model, org)

    assert grounded.ambiguous
    assert verdict.status == NEEDS_CLARIFICATION
    assert not verdict.is_executable, "a query with two meanings must not run"
    finding = next(f for f in verdict.findings if f.check == ENTITY)
    assert finding.severity == CLARIFY and "2 records" in finding.message


def test_an_ambiguous_anchor_cannot_silently_pick_a_subtree(org):
    """A traversal from an unresolved name would scope to whichever
    record was picked first."""
    model = _model_from_ir(subject_level="advisor",
                           subjects=[Subject(type="advisor", value="Yasir Ali")],
                           target_level="team", level_word="team")
    _grounded, hier, verdict = _run(model, org)

    assert hier.status == UNSUPPORTED
    assert not verdict.is_executable


# ---------------------------------------------------------------------
# The remaining validation checks
# ---------------------------------------------------------------------

def test_an_unknown_metric_is_rejected_not_substituted(org):
    model = SemanticModel(operation="leaderboard",
                          metrics=[{"name": "not_a_real_metric"}])
    _grounded, _hier, verdict = _run(model, org)

    assert verdict.status == INVALID
    assert [f.check for f in verdict.rejections] == [METRIC]


def test_an_unsupported_operation_is_rejected(org):
    model = SemanticModel(operation="leaderboard")
    model.operation = "teleport"          # bypass the constructor validator
    _grounded, _hier, verdict = _run(model, org)

    assert verdict.status == INVALID


def test_a_comparison_needs_two_subjects(org):
    model = SemanticModel(operation="comparison",
                          metrics=[{"name": "total_connects"}],
                          comparison_subjects=[EntityRef(name="Blue Area", level="team")])
    _grounded, _hier, verdict = _run(model, org)

    assert verdict.status == INVALID


def test_every_failure_is_reported_not_just_the_first(org):
    """A user with two problems is told both."""
    model = SemanticModel(operation="leaderboard",
                          metrics=[{"name": "not_a_real_metric"}],
                          subject=EntityRef(name="Qwerty Zzz", level="team"))
    _grounded, _hier, verdict = _run(model, org)

    assert {f.check for f in verdict.findings} == {METRIC, ENTITY}


def test_authorization_runs_and_passes_with_no_policy(org):
    """DECLARED, NOT DECIDED. There is no authorization policy to enforce
    — the posture is an open decision — so this check must not invent
    one. It exists so that when a policy is settled every caller already
    routes through it."""
    model = _model_from_ir(subject_level="team",
                           subjects=[Subject(type="team", value="Blue Area")])
    _grounded, _hier, verdict = _run(model, org, principal={"role": "advisor"})

    assert verdict.status == VALID
    assert not [f for f in verdict.findings
                if f.check == semantic_validation.AUTHORIZATION]


def test_a_valid_query_is_executable(org):
    model = _model_from_ir(subject_level="team",
                           subjects=[Subject(type="team", value="Blue Area")])
    _grounded, _hier, verdict = _run(model, org)
    assert verdict.is_executable and verdict.status == VALID
