"""Representative Semantic Models become correct executable QueryIR.

"Executable" is tested literally: the last cases compile and run through
the real compiler against a real (small) database, because an IR that
type-checks and returns the wrong rows is the failure this layer exists
to prevent.

The property pinned hardest is the identifier contract:

    NO NATURAL-LANGUAGE NAME REACHES A FIELD THE SQL LAYER TREATS AS AN
    IDENTIFIER.

`Subject.value` carries the canonical database value, never the user's
phrasing; advisors additionally carry `resolved_wid`, because a name
cannot address one person.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod
from app.llm import (
    entity_extractor, grounding, hierarchy_grounding, ir_conversion,
    semantic_validation,
)
from app.llm.ir_conversion import NotValidated, to_query_ir
from app.llm.query_compiler import compile_and_run
from app.llm.semantic_model import (
    Condition, EntityRef, MetricRequest, Ordering, Relationship, SemanticModel,
    TimeRange, from_query_ir,
)

UNIT_HEAD = "Faisal Naqvi"


@pytest.fixture()
def org(db_session):
    db_session.add_all([
        Advisor(wid=1, name="Ahmed Raza", team="Blue Area", company="Graana",
                rm=UNIT_HEAD, portfolio_lead="Bilal Khan", management_lead="Usman Ali"),
        Advisor(wid=2, name="Sara Iqbal", team="Blue Area", company="Graana",
                rm=UNIT_HEAD, portfolio_lead=UNIT_HEAD, management_lead=UNIT_HEAD),
        Advisor(wid=3, name="Omar Farooq", team="Downtown", company="IMARAT",
                rm=UNIT_HEAD, portfolio_lead="Bilal Khan", management_lead="Usman Ali"),
        # duplicate name -> unresolvable without a wid
        Advisor(wid=4, name="Yasir Ali", team="Blue Area", company="Graana",
                rm=UNIT_HEAD, portfolio_lead="Bilal Khan", management_lead="Usman Ali"),
        Advisor(wid=5, name="Yasir Ali", team="Downtown", company="IMARAT",
                rm=UNIT_HEAD, portfolio_lead="Bilal Khan", management_lead="Usman Ali"),
    ])
    db_session.add_all([
        Performance(wid=1, period=PerformancePeriod.MTD, cleared=900),
        Performance(wid=2, period=PerformancePeriod.MTD, cleared=100),
        Performance(wid=3, period=PerformancePeriod.MTD, cleared=500),
        Performance(wid=4, period=PerformancePeriod.MTD, cleared=50),
        Performance(wid=5, period=PerformancePeriod.MTD, cleared=25),
    ])
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _convert(model, db, principal=None):
    """The whole Phase 4-6 chain, exactly as interpret() runs it."""
    grounded = grounding.ground(model, db)
    hier = hierarchy_grounding.verify(model, grounded, db)
    verdict = semantic_validation.validate(model, grounded, hier, db, principal=principal)
    ir = to_query_ir(model, grounded, hier, verdict, principal=principal)
    return ir, verdict


def _metric_model(**kw):
    kw.setdefault("operation", "group_metric")
    kw.setdefault("metrics", [MetricRequest(name="mtd_cleared")])
    return SemanticModel(**kw)


# ---------------------------------------------------------------------
# The identifier contract
# ---------------------------------------------------------------------

def test_the_canonical_value_reaches_the_ir_not_the_users_words(org):
    """A typo is resolved during grounding and must not survive into the
    IR: the SQL layer matches `Subject.value` against the column."""
    ir, _ = _convert(_metric_model(subject=EntityRef(name="blue aera", level="team")), org)

    assert ir.subjects[0].value == "Blue Area"
    assert ir.subjects[0].resolved_id == "Blue Area"


def test_an_advisor_subject_carries_a_wid(org):
    ir, _ = _convert(_metric_model(
        subject=EntityRef(name="Ahmed Raza", level="advisor")), org)

    subject = ir.subjects[0]
    assert subject.resolved_wid == 1
    assert subject.type == "advisor"


def test_a_group_subject_carries_its_value_as_the_identifier(org):
    """Managers and groups have no id of their own in this schema, so the
    canonical value IS the identifier — recorded in resolved_id to say
    so, rather than left to be inferred from `value`."""
    ir, _ = _convert(_metric_model(
        subject=EntityRef(name=UNIT_HEAD, level="unit_head")), org)

    subject = ir.subjects[0]
    assert subject.resolved_id == UNIT_HEAD
    assert subject.resolved_wid is None


def test_an_unresolved_entity_never_reaches_the_ir(org):
    """The gate is validation, and this proves the gate holds: an
    ambiguous name produces no IR at all rather than an IR carrying the
    raw string into a scope filter."""
    model = _metric_model(subject=EntityRef(name="Yasir Ali", level="advisor"))
    _grounded = grounding.ground(model, org)

    with pytest.raises(NotValidated):
        _convert(model, org)


# ---------------------------------------------------------------------
# Normal metric query
# ---------------------------------------------------------------------

def test_a_normal_metric_query_reports_at_the_subjects_own_level(org):
    """THE "connects of Blue Area" DEFECT, from the conversion end.

    `subject_level` is where the answer is reported. Defaulting it to
    "advisor" groups by advisor INSIDE the team and answers with a list
    of members instead of the team's own figure — the same wrong answer
    the group_metric re-label exists to prevent.
    """
    ir, verdict = _convert(_metric_model(
        subject=EntityRef(name="Blue Area", level="team")), org)

    assert verdict.status == semantic_validation.VALID
    assert ir.subject_level == "team"
    assert ir.grouping_level() == "team"
    assert ir.target_level is None
    assert not ir.is_hierarchy_read()


def test_a_normal_metric_query_executes_to_one_row(org):
    """The team's own total: 900 + 100 + 50."""
    ir, _ = _convert(_metric_model(
        subject=EntityRef(name="Blue Area", level="team")), org)

    rows = compile_and_run(org, ir)
    assert len(rows) == 1
    assert rows[0]["name"] == "Blue Area"
    assert rows[0]["value"] == 1050


# ---------------------------------------------------------------------
# Hierarchy query
# ---------------------------------------------------------------------

def test_a_hierarchy_query_carries_the_verified_scope(org):
    ir, _ = _convert(_metric_model(
        scope=[EntityRef(name="Blue Area", level="team")],
        requested_level="advisor",
        relationship=Relationship(kind="membership", depth="subtree")), org)

    assert ir.target_level == "advisor"
    assert ir.subject_of == "team"
    assert ir.relation == "subtree"
    assert ir.is_hierarchy_read()
    assert ir.grouping_level() == "advisor"


def test_the_hierarchy_scope_comes_from_verification_not_the_request(org):
    """`target_level` is taken from the traversal that was actually run,
    so an IR can only ever carry a relationship the data supports."""
    model = _metric_model(
        scope=[EntityRef(name="Blue Area", level="team")],
        requested_level="advisor",
        relationship=Relationship(kind="membership", depth="subtree"))
    grounded = grounding.ground(model, org)
    hier = hierarchy_grounding.verify(model, grounded, org)
    verdict = semantic_validation.validate(model, grounded, hier, org)

    ir = to_query_ir(model, grounded, hier, verdict)

    assert hier.status == hierarchy_grounding.VERIFIED
    assert ir.target_level == hier.target_level
    assert ir.relation == hier.relation


def test_a_hierarchy_query_executes_to_the_members(org):
    ir, _ = _convert(_metric_model(
        scope=[EntityRef(name="Blue Area", level="team")],
        requested_level="advisor",
        relationship=Relationship(kind="membership", depth="subtree")), org)

    rows = compile_and_run(org, ir)
    assert {r["name"] for r in rows} == {"Ahmed Raza", "Sara Iqbal", "Yasir Ali"}


def test_a_direct_relationship_survives_conversion(org):
    ir, _ = _convert(_metric_model(
        scope=[EntityRef(name=UNIT_HEAD, level="unit_head")],
        requested_level="advisor",
        relationship=Relationship(kind="membership", depth="direct")), org)

    assert ir.relation == "direct"
    assert ir.target_level == "advisor"


# ---------------------------------------------------------------------
# Everything else the executable IR must carry
# ---------------------------------------------------------------------

def test_filters_time_sorting_grouping_and_limit_are_carried(org):
    ir, _ = _convert(_metric_model(
        operation="leaderboard",
        subject_level="advisor",
        conditions=[Condition(field="company", operator="=", value="Graana")],
        time_range=TimeRange(period="YTD", stated=True),
        ordering=Ordering(metric="mtd_cleared", direction="asc", stated=True),
        group_by="team",
        limit=5), org)

    assert [(f.field, f.operator, f.value) for f in ir.filters] == [("company", "=", "Graana")]
    assert ir.time_range.period == "YTD"
    assert ir.sort.metric == "mtd_cleared" and ir.sort.direction == "asc"
    assert ir.group_by == "team"
    assert ir.limit == 5


def test_an_unstated_direction_keeps_the_ir_default(org):
    """Only a stated direction may override the measure's own polarity."""
    ir, _ = _convert(_metric_model(
        operation="leaderboard", subject_level="advisor",
        ordering=Ordering(metric="mtd_cleared", direction=None, stated=False)), org)

    assert ir.sort.direction == "desc"


def test_a_period_comparison_sets_the_compare_mode(org):
    ir, _ = _convert(_metric_model(
        time_range=TimeRange(period="MTD", compare_to="YTD", stated=True),
        subject=EntityRef(name="Blue Area", level="team")), org)

    assert ir.time_range.mode == "compare"
    assert ir.compare_period() == "YTD"


def test_several_metrics_keep_their_order_with_the_first_as_primary(org):
    ir, _ = _convert(_metric_model(
        operation="leaderboard", subject_level="advisor",
        metrics=[MetricRequest(name="mtd_cleared"),
                 MetricRequest(name="total_connects")]), org)

    assert ir.metric.key == "mtd_cleared"
    assert ir.metric_keys() == ["mtd_cleared", "total_connects"]


def test_a_comparison_carries_its_targets_as_subjects(org):
    ir, _ = _convert(_metric_model(
        operation="comparison",
        comparison_subjects=[EntityRef(name="Blue Area", level="team"),
                             EntityRef(name="Downtown", level="team")]), org)

    assert ir.intent == "comparison"
    assert [s.value for s in ir.subjects] == ["Blue Area", "Downtown"]


def test_the_authorization_scope_is_carried_but_not_enforced(org):
    """Recorded so a policy has one place to be applied. Nothing reads it,
    so results are unchanged — which is the point until a posture is
    actually decided."""
    principal = {"sub": "u1", "role": "unit_head"}
    ir, _ = _convert(_metric_model(
        subject=EntityRef(name="Blue Area", level="team")), org, principal=principal)

    assert ir.authorization_scope == principal
    assert len(compile_and_run(org, ir)) == 1, "carrying it changes no result"


# ---------------------------------------------------------------------
# What conversion refuses to do
# ---------------------------------------------------------------------

def test_an_operation_no_ir_expresses_returns_none(org):
    """A roster is answered from the plan. That is an expected state, so
    it is None rather than an exception — the caller falls through."""
    model = SemanticModel(operation="roster",
                          subject=EntityRef(name="Blue Area", level="team"))
    ir, verdict = _convert(model, org)

    assert verdict.status == semantic_validation.VALID
    assert ir is None


def test_conversion_refuses_an_invalid_interpretation(org):
    model = _metric_model(subject=EntityRef(name="Qwerty Zzz", level="team"))

    with pytest.raises(NotValidated) as excinfo:
        _convert(model, org)
    assert "does not exist" in str(excinfo.value)


def test_conversion_does_not_mutate_the_semantic_model(org):
    model = _metric_model(subject=EntityRef(name="blue aera", level="team"))
    before = model.model_dump()

    _convert(model, org)

    assert model.model_dump() == before, "the model still says what the user said"


# ---------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------

@pytest.mark.parametrize("build", [
    lambda: _metric_model(subject=EntityRef(name="Blue Area", level="team")),
    lambda: _metric_model(scope=[EntityRef(name="Blue Area", level="team")],
                          requested_level="advisor",
                          relationship=Relationship(kind="membership", depth="subtree")),
    lambda: _metric_model(operation="leaderboard", subject_level="advisor",
                          ordering=Ordering(metric="mtd_cleared", direction="desc",
                                            stated=True), limit=5),
])
def test_a_converted_ir_reads_back_as_the_same_meaning(build, org):
    """to_query_ir is the inverse of from_query_ir for what both express.
    A round trip that changed the operation, the measures or the shape
    would mean one of the two is lying about the same parse."""
    model = build()
    ir, _ = _convert(model, org)

    back = from_query_ir(ir)

    assert back.operation == model.operation
    assert [m.name for m in back.metrics] == [m.name for m in model.metrics]
    assert back.is_hierarchy_query() == model.is_hierarchy_query()
    assert back.limit == model.limit
