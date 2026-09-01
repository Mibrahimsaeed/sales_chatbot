"""A subject scopes the query to itself, at whatever level it names.

THE DEFECT. `_apply_subject_filter` built its predicate from the level
the answer is REPORTED at (`subject_level`) and kept only the subjects
whose type already equalled it:

    column   = hierarchy.column_for(ir.subject_level)
    subjects = [s for s in ir.subjects if s.type == ir.subject_level]

So a subject naming a CONTAINER matched nothing and was dropped without
a trace. "top teams in <a company> by revenue" carries subject_level=
"team" and a `company` subject, so the scope disappeared and the query
ranked every team in the business — a confident answer to a question
nobody asked. Measured against SQL truth, the leak was total: 9 teams
where 5 were in scope, 544 advisors where 210 were, 544 where 52 were.

It was mostly invisible because the parser emits the scope TWICE, once
in `subjects` and once in `filters`, and the filter path was already
correct. In 12 parses carrying a cross-level subject the duplicate was
present 10 times; the other 2 answered globally.

The fix reuses `hierarchy.scope_filter` — "THE one definition of in
scope", the same helper the filter path calls — rather than adding a
second notion of containment. No join is needed: every level is a column
on the advisor row, which is why scoping across levels is a grouping
rather than a traversal.

These tests pin containment against independently computed SQL truth, so
they check the POPULATION and not merely that some rows came back.
"""

import pytest
from sqlalchemy import func

from app.database.models import Advisor, Calls
from app.llm import entity_extractor, hierarchy
from app.llm.ir_validator import validate_ir
from app.llm.query_compiler import compile_and_run
from app.llm.query_ir import Filter, MetricRef, QueryIR, Sort, Subject

# Two companies x two regions x three teams, arranged so that every
# containment pair below has a DIFFERENT correct answer from every other.
# A wrong scope therefore shows up as a wrong set, not just a wrong count.
_ORG = [
    # wid, name,       team,         company,          region
    (1, "Adviser One",   "Team Alpha", "Acme Holdings",  "Northland"),
    (2, "Adviser Two",   "Team Alpha", "Acme Holdings",  "Northland"),
    (3, "Adviser Three", "Team Beta",  "Acme Holdings",  "Southland"),
    (4, "Adviser Four",  "Team Gamma", "Borealis Group", "Northland"),
    (5, "Adviser Five",  "Team Gamma", "Borealis Group", "Southland"),
]


@pytest.fixture()
def org(db_session):
    for wid, name, team, company, region in _ORG:
        db_session.add(Advisor(wid=wid, name=name, team=team, company=company,
                               region=region, rm="UH One", portfolio_lead="ZH One",
                               management_lead="BCM One", in_master_sheet=True))
        db_session.add(Calls(wid=wid, connects_mtd=10 * wid))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _truth(db, scope_level, scope_value, report_level):
    """Independently computed containment, straight from the columns."""
    scope_col = hierarchy.column_for(scope_level)
    report_col = hierarchy.column_for(report_level)
    rows = (db.query(report_col)
              .filter(Advisor.in_master_sheet.is_(True),
                      scope_col.ilike(scope_value),
                      report_col.isnot(None))
              .distinct().all())
    return {r[0] for r in rows}


def _run(db, operation, scope_level, scope_value, report_level, **over):
    kw = dict(intent="leaderboard" if operation == "leaderboard" else "filtered_list",
              operation=operation, subject_level=report_level,
              subjects=[Subject(type=scope_level, value=scope_value,
                                match_confidence=1.0)],
              metric=MetricRef(key="total_connects", confidence=0.95),
              sort=Sort(metric="total_connects", direction="desc"), limit=None)
    kw.update(over)
    ir = QueryIR(**kw)
    rows = compile_and_run(db, validate_ir(ir, db).ir) or []
    return {r["name"] for r in rows}


# =====================================================================
# Containment, every pair, against SQL truth
# =====================================================================
CONTAINMENT = [
    ("company", "Acme Holdings", "team"),
    ("company", "Acme Holdings", "advisor"),
    ("team", "Team Alpha", "advisor"),
    ("region", "Northland", "team"),
    ("region", "Northland", "advisor"),
]


@pytest.mark.parametrize("scope_level,scope_value,report_level", CONTAINMENT)
@pytest.mark.parametrize("operation", ["leaderboard", "population"])
def test_a_subject_scopes_the_answer_to_its_own_level(
        org, operation, scope_level, scope_value, report_level):
    """The whole defect, at every containment pair and on both operations
    that reach the compiler with the subject intact."""
    expected = _truth(org, scope_level, scope_value, report_level)
    got = _run(org, operation, scope_level, scope_value, report_level)
    assert got == expected, (
        f"{scope_level}={scope_value!r} reported at {report_level}: "
        f"out of scope {sorted(got - expected)}, missing {sorted(expected - got)}"
    )


@pytest.mark.parametrize("scope_level,scope_value,report_level", CONTAINMENT)
def test_the_scope_is_the_same_whether_it_arrives_as_a_subject_or_a_filter(
        org, scope_level, scope_value, report_level):
    """ONE definition of in-scope. The parser emits the scope in both
    places; the two mechanisms must not disagree — that divergence is
    what let the defect hide."""
    as_subject = _run(org, "leaderboard", scope_level, scope_value, report_level)
    as_filter = _run(org, "leaderboard", scope_level, scope_value, report_level,
                     subjects=[],
                     filters=[Filter(field=scope_level, operator="=",
                                     value=scope_value, confidence=1.0)])
    assert as_subject == as_filter


def test_a_container_is_not_reported_instead_of_its_members(org):
    """The failure mode in words: the answer must be the teams, never the
    single company that contains them."""
    got = _run(org, "leaderboard", "company", "Acme Holdings", "team")
    assert got == {"Team Alpha", "Team Beta"}
    assert "Acme Holdings" not in got


# =====================================================================
# What must not change
# =====================================================================
def test_a_same_level_subject_still_selects_itself(org):
    assert _run(org, "leaderboard", "team", "Team Alpha", "team") == {"Team Alpha"}


def test_two_subjects_at_one_level_still_mean_either(org):
    """A comparison's subjects are OR-ed — the meaning the previous
    `column.in_(names)` carried, preserved."""
    ir = QueryIR(intent="comparison", operation="comparison", subject_level="team",
                 subjects=[Subject(type="team", value="Team Alpha", match_confidence=1.0),
                           Subject(type="team", value="Team Beta", match_confidence=1.0)],
                 metric=MetricRef(key="total_connects", confidence=0.95),
                 sort=Sort(metric="total_connects", direction="desc"))
    rows = compile_and_run(org, validate_ir(ir, org).ir) or []
    assert {r["name"] for r in rows} == {"Team Alpha", "Team Beta"}


def test_subjects_at_different_levels_intersect(org):
    """AND across levels — the same combination `_apply_entity_filters`
    already produces for two filters on different fields. Team Gamma sits
    in Borealis, so scoping it to Acme leaves nothing, and the empty
    result is the honest one."""
    both = _run(org, "leaderboard", "company", "Acme Holdings", "advisor",
                subjects=[Subject(type="company", value="Acme Holdings", match_confidence=1.0),
                          Subject(type="team", value="Team Alpha", match_confidence=1.0)])
    assert both == {"Adviser One", "Adviser Two"}

    disjoint = _run(org, "leaderboard", "company", "Acme Holdings", "advisor",
                    subjects=[Subject(type="company", value="Acme Holdings", match_confidence=1.0),
                              Subject(type="team", value="Team Gamma", match_confidence=1.0)])
    assert disjoint == set()


def test_a_comparisons_sides_are_not_scoped_into(org):
    """DELIBERATELY UNCHANGED. A comparison's subjects are the things
    being set beside each other, so a subject of another type is noise
    rather than a container — scoping into it would intersect the sides
    away and answer with nothing. Pinned upstream too, in
    test_comparison_still_requires_exact_subject_type_match."""
    ir = QueryIR(intent="comparison", operation="comparison", subject_level="team",
                 subjects=[Subject(type="team", value="Team Alpha", match_confidence=1.0),
                           Subject(type="company", value="Borealis Group",
                                   match_confidence=1.0)],
                 metric=MetricRef(key="total_connects", confidence=0.95),
                 sort=Sort(metric="total_connects", direction="desc"))
    rows = compile_and_run(org, validate_ir(ir, org).ir) or []
    assert {r["name"] for r in rows} == {"Team Alpha"}


def test_an_advisor_subject_still_binds_by_wid(org):
    """Identity, not name: several real people share a name, so a name
    match sums them into one row. Unchanged by this fix."""
    ir = QueryIR(intent="filtered_list", operation="group_metric", subject_level="advisor",
                 subjects=[Subject(type="advisor", value="Adviser One",
                                   resolved_wid=1, match_confidence=1.0)],
                 metric=MetricRef(key="total_connects", confidence=0.95),
                 sort=Sort(metric=None))
    rows = compile_and_run(org, validate_ir(ir, org).ir) or []
    assert [r["wid"] for r in rows] == [1]


def test_a_subject_whose_type_names_no_column_is_skipped(org):
    """`attendance_status` and metric keys are not hierarchy levels; a
    subject typed at one must not become a filter on nothing."""
    from app.llm.query_compiler import _subjects_by_level
    ir = QueryIR(intent="filtered_list", operation="group_metric", subject_level="team",
                 subjects=[Subject(type="team", value="Team Alpha", match_confidence=1.0)])
    assert set(_subjects_by_level(ir)) == {"team"}


def test_no_subject_leaves_the_query_unscoped(org):
    """A ranking that names nobody still ranks everybody."""
    got = _run(org, "leaderboard", "team", "Team Alpha", "advisor", subjects=[])
    assert got == {n for _, n, _, _, _ in _ORG}
