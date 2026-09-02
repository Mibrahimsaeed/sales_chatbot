"""End to end: Semantic Model -> QueryIR -> database -> verified rows.

Every test here runs real SQL against a real (small) database through
`query_compiler`, the same functions chat_service and the leaderboard API
call. Nothing is mocked, because the failures this phase must catch are
precisely the ones that only appear once the query runs: a scope that
compiled to the wrong population, a wid that stopped addressing one
person, a period recorded on the IR and ignored by the binding.

The numbers below are chosen so a wrong population produces a DIFFERENT
number rather than an empty result — 900/100/50 in Blue Area against
500/25 in Downtown — so a broken scope fails loudly instead of looking
like "no data".
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod
from app.llm import advisor_resolver, entity_extractor, ir_execution
from app.llm.ir_execution import FAILED, PASSED, SKIPPED, execute, execute_semantic_model
from app.llm.semantic_model import (
    Condition, EntityRef, MetricRequest, Ordering, Relationship, SemanticModel, TimeRange,
)

UNIT_HEAD = "Faisal Naqvi"


@pytest.fixture()
def org(db_session):
    db_session.add_all([
        Advisor(wid=1, name="Ahmed Raza", team="Blue Area", company="Graana",
                rm=UNIT_HEAD, portfolio_lead="Bilal Khan", management_lead="Usman Ali"),
        Advisor(wid=2, name="Sara Iqbal", team="Blue Area", company="Graana",
                rm=UNIT_HEAD, portfolio_lead=UNIT_HEAD, management_lead=UNIT_HEAD),
        Advisor(wid=3, name="Yasir Ali", team="Blue Area", company="IMARAT",
                rm=UNIT_HEAD, portfolio_lead="Bilal Khan", management_lead="Usman Ali"),
        Advisor(wid=4, name="Omar Farooq", team="Downtown", company="IMARAT",
                rm=UNIT_HEAD, portfolio_lead="Bilal Khan", management_lead="Usman Ali"),
        Advisor(wid=5, name="Zara Khan", team="Downtown", company="Graana",
                rm=UNIT_HEAD, portfolio_lead="Bilal Khan", management_lead="Usman Ali"),
    ])
    db_session.add_all([
        Performance(wid=1, period=PerformancePeriod.MTD, cleared=900),
        Performance(wid=2, period=PerformancePeriod.MTD, cleared=100),
        Performance(wid=3, period=PerformancePeriod.MTD, cleared=50),
        Performance(wid=4, period=PerformancePeriod.MTD, cleared=500),
        Performance(wid=5, period=PerformancePeriod.MTD, cleared=25),
        # a different window, so "respecting the time range" is falsifiable
        Performance(wid=1, period=PerformancePeriod.YTD, cleared=9000),
        Performance(wid=2, period=PerformancePeriod.YTD, cleared=1000),
        Performance(wid=3, period=PerformancePeriod.YTD, cleared=500),
    ])
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _model(**kw):
    kw.setdefault("operation", "group_metric")
    kw.setdefault("metrics", [MetricRequest(name="mtd_cleared")])
    return SemanticModel(**kw)


def _run(db, model, **kw):
    result, verdict = execute_semantic_model(db, model, **kw)
    assert result is not None, f"expected an executable query, got {verdict.status}"
    return result


def _check(result, name):
    return next(c for c in result.checks if c.name == name)


# ---------------------------------------------------------------------
# The two shapes that must not collapse into each other
# ---------------------------------------------------------------------

def test_a_group_metric_returns_the_groups_own_figure(org):
    """900 + 100 + 50, as ONE row."""
    result = _run(org, _model(subject=EntityRef(name="Blue Area", level="team")))

    assert result.answered and result.verified
    assert result.row_count == 1
    assert result.rows[0]["name"] == "Blue Area"
    assert result.rows[0]["value"] == 1050


def test_a_hierarchy_read_returns_the_members(org):
    result = _run(org, _model(
        scope=[EntityRef(name="Blue Area", level="team")],
        requested_level="advisor",
        relationship=Relationship(kind="membership", depth="subtree")))

    assert result.verified
    assert {r["name"] for r in result.rows} == {"Ahmed Raza", "Sara Iqbal", "Yasir Ali"}
    assert sum(r["value"] for r in result.rows) == 1050, "same population, same total"


def test_the_two_shapes_agree_on_the_total(org):
    """The strongest available cross-check: one row of 1050 and three
    rows summing to 1050 are the same population counted two ways. If a
    scope silently widened, these diverge."""
    grouped = _run(org, _model(subject=EntityRef(name="Blue Area", level="team")))
    members = _run(org, _model(
        scope=[EntityRef(name="Blue Area", level="team")],
        requested_level="advisor",
        relationship=Relationship(kind="membership", depth="subtree")))

    assert grouped.rows[0]["value"] == sum(r["value"] for r in members.rows)


# ---------------------------------------------------------------------
# Resolved identifiers
# ---------------------------------------------------------------------

def test_an_advisor_subject_is_addressed_by_wid(org):
    result = _run(org, _model(subject=EntityRef(name="Ahmed Raza", level="advisor")))

    assert result.row_count == 1
    assert result.rows[0]["wid"] == 1
    assert result.rows[0]["value"] == 900
    assert _check(result, ir_execution.IDENTIFIERS).status == PASSED


def test_the_identifier_check_would_catch_a_widened_result(org):
    """The check has teeth: hand it a result containing someone else and
    it fails. Without this the PASSED above could be vacuous."""
    result = _run(org, _model(subject=EntityRef(name="Ahmed Raza", level="advisor")))
    tampered = ir_execution._check_identifiers(
        result.ir, result.rows + [{"wid": 99, "name": "Someone Else"}])

    assert tampered.status == FAILED
    assert "99" in tampered.detail


# ---------------------------------------------------------------------
# Filters, time, grouping, hierarchy scope
# ---------------------------------------------------------------------

def test_filters_are_applied(org):
    """Blue Area has two Graana advisors (900 + 100) and one IMARAT."""
    result = _run(org, _model(
        subject=EntityRef(name="Blue Area", level="team"),
        conditions=[Condition(field="company", operator="=", value="Graana")]))

    assert result.rows[0]["value"] == 1000


def test_the_time_range_is_respected(org):
    """The same question over two windows must return the two windows'
    numbers — 1050 for MTD, 10500 for YTD."""
    mtd = _run(org, _model(subject=EntityRef(name="Blue Area", level="team"),
                           time_range=TimeRange(period="MTD", stated=True)))
    ytd = _run(org, _model(subject=EntityRef(name="Blue Area", level="team"),
                           time_range=TimeRange(period="YTD", stated=True)))

    assert mtd.rows[0]["value"] == 1050
    assert ytd.rows[0]["value"] == 10500
    assert _check(ytd, ir_execution.TIME).status == PASSED
    assert "YTD" in _check(ytd, ir_execution.TIME).detail


def test_grouping_produces_one_row_per_group(org):
    result = _run(org, _model(operation="leaderboard", subject_level="team",
                              ordering=Ordering(metric="mtd_cleared", direction="desc",
                                                stated=True)))

    assert _check(result, ir_execution.GROUPING).status == PASSED
    names = [r["name"] for r in result.rows]
    assert len(names) == len(set(names))
    assert names[0] == "Blue Area", "1050 outranks 525"


def test_the_hierarchy_scope_matches_what_verification_found(org):
    """Cross-checks the compiled SQL against the traversal Phase 5 ran
    against the data — two independent routes to the same population."""
    result = _run(org, _model(
        scope=[EntityRef(name=UNIT_HEAD, level="unit_head")],
        requested_level="advisor",
        relationship=Relationship(kind="membership", depth="subtree")))

    check = _check(result, ir_execution.HIERARCHY)
    assert check.status == PASSED
    assert result.row_count == 5


def test_a_direct_relationship_narrows_the_executed_population(org):
    """Only Sara names the Unit Head as her own management_lead."""
    result = _run(org, _model(
        scope=[EntityRef(name=UNIT_HEAD, level="unit_head")],
        requested_level="advisor",
        relationship=Relationship(kind="membership", depth="direct")))

    assert [r["name"] for r in result.rows] == ["Sara Iqbal"]
    assert _check(result, ir_execution.HIERARCHY).status == PASSED


def test_the_scope_check_would_catch_a_leaked_row(org):
    result = _run(org, _model(subject=EntityRef(name="Blue Area", level="team")))
    leaked = ir_execution._check_scope(
        result.ir, result.rows + [{"team": "Downtown", "name": "Downtown"}])

    assert any(c.status == FAILED for c in leaked)


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------

def test_numerical_results_are_deterministic(org):
    """Pinned here rather than checked on every request: re-running each
    query in production would double its cost to detect a property of the
    SQL layer rather than of one execution."""
    model = _model(operation="leaderboard", subject_level="advisor",
                   ordering=Ordering(metric="mtd_cleared", direction="desc", stated=True))

    runs = [[(r["wid"], r["value"]) for r in _run(org, model).rows] for _ in range(3)]

    assert runs[0] == runs[1] == runs[2]
    assert runs[0][0][1] == 900


def test_pagination_reuses_the_existing_offset_path(org):
    """`offset` is the compiler's own, so paging cannot drift from the
    unpaged ordering."""
    model = _model(operation="leaderboard", subject_level="advisor",
                   ordering=Ordering(metric="mtd_cleared", direction="desc", stated=True))
    everything = _run(org, model, with_total=True)
    second = _run(org, model, offset=1)

    assert everything.total == 5
    assert [r["wid"] for r in second.rows] == [r["wid"] for r in everything.rows][1:]


# ---------------------------------------------------------------------
# What does not execute
# ---------------------------------------------------------------------

def test_an_invalid_interpretation_never_reaches_the_database(org):
    result, verdict = execute_semantic_model(
        org, _model(subject=EntityRef(name="Qwerty Zzz", level="team")))

    assert result is None
    assert verdict.status == "invalid"


def test_an_ambiguous_entity_never_reaches_the_database(org):
    """Two Yasir Alis would sum into one row."""
    org.add(Advisor(wid=6, name="Yasir Ali", team="Downtown", company="Graana"))
    org.commit()
    # BOTH caches. advisor_resolver keeps its own name index under its own
    # TTL, and resolve_by_name refreshes only that one — so clearing the
    # extractor's alone leaves a stale index in which this second Yasir
    # Ali does not exist yet, and the ambiguity is invisible.
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()

    result, verdict = execute_semantic_model(
        org, _model(subject=EntityRef(name="Yasir Ali", level="advisor")))

    assert result is None
    assert verdict.status == "needs_clarification"


def test_a_plan_served_operation_does_not_execute_as_ir(org):
    result, verdict = execute_semantic_model(
        org, SemanticModel(operation="roster",
                           subject=EntityRef(name="Blue Area", level="team")))

    assert result is None
    assert verdict.status == "valid", "valid, but no IR expresses a roster"


def test_an_unanswerable_ir_reports_unanswered_rather_than_empty(org):
    """None and [] mean different things: the compiler could not answer,
    versus it ran and matched nothing."""
    model = _model(subject_level="company", operation="leaderboard",
                   metrics=[MetricRequest(name="mtd_cleared")],
                   ordering=Ordering(metric="mtd_cleared", direction="desc", stated=True))
    result, _ = execute_semantic_model(org, model)

    assert result is not None
    if not result.answered:
        assert result.rows is None and not result.verified


# ---------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------

def test_every_data_route_still_requires_a_user(org):
    """Authorization is enforced at the API boundary, and this phase must
    not have opened a path around it."""
    from fastapi.routing import APIRoute

    from app.core.dependencies import get_current_user
    from app.main import app

    # KNOWN GAP, recorded rather than hidden. This route triggers an
    # embedding index rebuild and requires no token. It predates this
    # phase, it is not on the query path, and closing it is part of the
    # authorization posture that is still an open decision — so it is
    # listed here explicitly. If a SECOND route ever joins it, this test
    # fails, which is the point.
    known_unauthenticated = {"/health/embeddings/rebuild"}

    unprotected = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or route.path.startswith("/api/token"):
            continue
        if route.path in ("/", "/health", "/docs", "/openapi.json", "/redoc"):
            continue
        names = {d.call for d in route.dependant.dependencies}
        if get_current_user not in names:
            unprotected.append(route.path)

    assert set(unprotected) == known_unauthenticated, \
        f"authentication changed on: {set(unprotected) ^ known_unauthenticated}"


def test_the_query_path_itself_is_authenticated(org):
    """The routes this phase actually feeds: chat, leaderboard, advisor,
    hierarchy. These must never be reachable without a token."""
    from fastapi.routing import APIRoute

    from app.core.dependencies import get_current_user
    from app.main import app

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if not route.path.startswith(("/api/chat", "/api/leaderboard",
                                      "/api/advisor", "/api/hierarchy")):
            continue
        assert get_current_user in {d.call for d in route.dependant.dependencies}, \
            route.path


def test_the_principal_is_carried_onto_the_executed_ir(org):
    principal = {"sub": "u1", "role": "unit_head"}
    result = _run(org, _model(subject=EntityRef(name="Blue Area", level="team")),
                  principal=principal)

    assert result.ir.authorization_scope == principal
    assert result.rows[0]["value"] == 1050, "carrying it changes no number"
