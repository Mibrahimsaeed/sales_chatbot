"""Multiple measures survive EXECUTION and are all computed.

Companion to test_multi_metric.py, which covers the other end of the
same problem: that file pins multi-measure ALIAS RESOLUTION (a question
naming two measures must not resolve to one), this one pins that all of
them are actually computed once the IR reaches SQL.

    "Show answered calls, connects and meeting rate for Blue Area"

The defect this guards is a SILENT reduction to the first measure: the
query names three, the reply shows one, and nothing anywhere says the
other two were dropped.

Two properties carry the file:

  1. EVERY NAMED MEASURE COMES BACK. Not "the primary plus whatever a
     bundle happens to include" — the ones the user asked for, and a
     measure that cannot be computed at this level is REPORTED rather
     than omitted.

  2. A COMPANION EQUALS A PRIMARY. The same measure must produce the same
     number whether it is the one being ranked by or one of the others.
     They go through the same compiler, the same bindings and the same
     period resolution, and these tests hold that to account — including
     for a RATIO measure, which averages across a group where a count
     measure sums.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, SalesFunnel,
)
from app.llm import entity_extractor, ir_execution
from app.llm.ir_execution import FAILED, METRIC_CELLS_KEY, PASSED, SKIPPED
from app.llm.semantic_model import EntityRef, MetricRequest, Ordering, SemanticModel

# Three measures over three different fact tables, one of them a ratio.
CLEARED = "mtd_cleared"          # Performance.cleared      SUM
ACHIEVEMENT = "achievement_pct"  # Performance.pct          RATIO
CONNECTS = "total_connects"      # SalesFunnel new+followup SUM
ANSWERED = "answered_calls"      # Calls.answered_calls_mtd SUM


@pytest.fixture()
def org(db_session):
    db_session.add_all([
        Advisor(wid=1, name="Ahmed Raza", team="Blue Area", company="Graana"),
        Advisor(wid=2, name="Sara Iqbal", team="Blue Area", company="Graana"),
        Advisor(wid=3, name="Omar Farooq", team="Downtown", company="IMARAT"),
    ])
    db_session.add_all([
        Performance(wid=1, period=PerformancePeriod.MTD, cleared=900, target=1000, pct=90),
        Performance(wid=2, period=PerformancePeriod.MTD, cleared=100, target=200, pct=50),
        Performance(wid=3, period=PerformancePeriod.MTD, cleared=500, target=500, pct=100),
    ])
    db_session.add_all([
        SalesFunnel(wid=1, mtd_new_connect=5, mtd_followup_connect=5),
        SalesFunnel(wid=2, mtd_new_connect=1, mtd_followup_connect=1),
        SalesFunnel(wid=3, mtd_new_connect=3, mtd_followup_connect=0),
    ])
    db_session.add_all([
        Calls(wid=1, answered_calls_mtd=10),
        Calls(wid=2, answered_calls_mtd=20),
        Calls(wid=3, answered_calls_mtd=7),
    ])
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _model(keys, **kw):
    kw.setdefault("operation", "group_metric")
    kw.setdefault("subject", EntityRef(name="Blue Area", level="team"))
    return SemanticModel(metrics=[MetricRequest(name=k) for k in keys], **kw)


def _run(db, model):
    result, verdict = ir_execution.execute_semantic_model(db, model)
    assert result is not None, f"expected execution, got {verdict.status}"
    assert result.answered, "the compiler could not answer this IR"
    return result


def _check(result, name):
    return next(c for c in result.checks if c.name == name)


def _alone(db, key, **kw):
    """The measure as the ONLY one asked for — the reference value."""
    return _run(db, _model([key], **kw)).rows[0]["value"]


# ---------------------------------------------------------------------
# Two, three, and mixed types
# ---------------------------------------------------------------------

def test_two_metrics_are_both_computed(org):
    result = _run(org, _model([CLEARED, CONNECTS]))

    assert result.metrics == [CLEARED, CONNECTS]
    assert not result.unavailable_metrics
    cells = result.rows[0][METRIC_CELLS_KEY]
    assert cells[CLEARED] == 1000        # 900 + 100
    assert cells[CONNECTS] == 12         # (5+5) + (1+1)


def test_three_metrics_across_three_fact_tables(org):
    """The phase's own example shape: three measures, three sources."""
    result = _run(org, _model([ANSWERED, CONNECTS, CLEARED]))

    assert result.metrics == [ANSWERED, CONNECTS, CLEARED]
    cells = result.rows[0][METRIC_CELLS_KEY]
    assert cells[ANSWERED] == 30         # Calls
    assert cells[CONNECTS] == 12         # SalesFunnel
    assert cells[CLEARED] == 1000        # Performance
    assert _check(result, ir_execution.METRICS).status == PASSED


def test_mixed_metric_types_keep_their_own_rollup(org):
    """A count SUMS across the group; a ratio is RE-DERIVED from its own
    parts.

    achievement is not the mean of the members' percentages (that would
    be 70 here) and certainly not their sum (140). It is the group's
    cleared over the group's target — 1000/1200 — so a large advisor
    counts for more than a small one. Computing it the way a count is
    computed is the failure this pins, and it can only surface once a
    ratio travels the companion path.
    """
    result = _run(org, _model([CLEARED, ACHIEVEMENT]))

    cells = result.rows[0][METRIC_CELLS_KEY]
    assert cells[CLEARED] == 1000, "counts sum"
    assert round(cells[ACHIEVEMENT], 2) == 83.33, "1000 cleared / 1200 target"


# ---------------------------------------------------------------------
# A companion must equal a primary
# ---------------------------------------------------------------------

@pytest.mark.parametrize("key", [CLEARED, CONNECTS, ANSWERED, ACHIEVEMENT])
def test_a_companion_measure_equals_the_same_measure_alone(key, org):
    """The property that makes multi-metric trustworthy. Includes the
    RATIO case, where a wrong rollup would show up only here."""
    reference = _alone(org, key)
    together = _run(org, _model([CLEARED, CONNECTS, ANSWERED, ACHIEVEMENT]))

    assert together.rows[0][METRIC_CELLS_KEY][key] == reference


def test_the_primary_value_is_reused_not_recomputed(org):
    """`value` is the number the rows were ranked and rendered by. Its own
    column must be that same number, not a second read of it."""
    result = _run(org, _model([CLEARED, CONNECTS]))
    row = result.rows[0]

    assert row[METRIC_CELLS_KEY][CLEARED] == row["value"]


# ---------------------------------------------------------------------
# Per-row correctness, not just per-group
# ---------------------------------------------------------------------

def test_each_row_gets_its_own_values(org):
    """Merging by identity, not by position: a wrong join key shows one
    person's answered calls beside another's connects."""
    result = _run(org, _model(
        [CLEARED, ANSWERED], operation="leaderboard", subject=None,
        subject_level="advisor",
        ordering=Ordering(metric=CLEARED, direction="desc", stated=True)))

    by_name = {r["name"]: r[METRIC_CELLS_KEY] for r in result.rows}
    assert by_name["Ahmed Raza"][CLEARED] == 900
    assert by_name["Ahmed Raza"][ANSWERED] == 10
    assert by_name["Sara Iqbal"][CLEARED] == 100
    assert by_name["Sara Iqbal"][ANSWERED] == 20


def test_a_limit_does_not_truncate_the_companion_columns(org):
    """The companion query is ordered by its OWN measure, so the rows the
    primary selected are not necessarily inside its top N.

    Ranked by cleared, the top two are Ahmed (900) and Omar (500). Ranked
    by answered calls the top two would be Sara (20) and Ahmed (10) — so
    a companion that inherited limit=2 would never see Omar, and his
    answered-calls cell would come back empty while looking like a
    legitimate "no data".
    """
    result = _run(org, _model(
        [CLEARED, ANSWERED], operation="leaderboard", subject=None,
        subject_level="advisor", limit=2,
        ordering=Ordering(metric=CLEARED, direction="desc", stated=True)))

    cells = {r["name"]: r[METRIC_CELLS_KEY] for r in result.rows}
    assert set(cells) == {"Ahmed Raza", "Omar Farooq"}, "the primary's limit still applies"
    assert cells["Omar Farooq"][ANSWERED] == 7, "not None, and not truncated away"


def test_scope_applies_to_every_measure(org):
    """Downtown's numbers must not leak into Blue Area's companions."""
    result = _run(org, _model([CLEARED, CONNECTS, ANSWERED]))

    cells = result.rows[0][METRIC_CELLS_KEY]
    assert cells[CLEARED] == 1000, "500 from Downtown excluded"
    assert cells[CONNECTS] == 12, "3 from Downtown excluded"
    assert cells[ANSWERED] == 30, "7 from Downtown excluded"


# ---------------------------------------------------------------------
# Nothing is dropped silently
# ---------------------------------------------------------------------

def test_an_uncomputable_measure_is_reported_not_omitted(org):
    """The check fails loudly. A measure with no binding at this grouping
    level must never just be absent from the answer."""
    result = _run(org, _model([CLEARED, CONNECTS]))
    tampered = ir_execution.ExecutedResult(
        ir=result.ir, rows=result.rows, unavailable_metrics=["something_unbindable"])

    assert ir_execution._check_metrics(result.ir, tampered).status == FAILED


def test_the_metrics_check_is_reported_in_the_result(org):
    result = _run(org, _model([CLEARED, CONNECTS, ANSWERED]))
    payload = result.to_dict()

    assert payload["metrics"] == [CLEARED, CONNECTS, ANSWERED]
    assert payload["unavailable_metrics"] == []


# ---------------------------------------------------------------------
# Single-metric behaviour is preserved
# ---------------------------------------------------------------------

def test_a_single_metric_query_is_unchanged(org):
    result = _run(org, _model([CLEARED]))

    assert result.row_count == 1
    assert result.rows[0]["value"] == 1000
    assert result.metrics == [CLEARED]
    assert _check(result, ir_execution.METRICS).status == SKIPPED


def test_a_single_metric_query_runs_no_companion_queries(org, monkeypatch):
    """Preserving existing behaviour means not paying for a feature the
    query does not use: one measure must still be exactly one query."""
    from app.llm import ir_execution as module

    calls = []
    real = module.compile_and_run
    monkeypatch.setattr(module, "compile_and_run",
                        lambda db, ir, offset=0: calls.append(ir) or real(db, ir, offset=offset))

    _run(org, _model([CLEARED]))
    assert len(calls) == 1

    calls.clear()
    _run(org, _model([CLEARED, CONNECTS, ANSWERED]))
    assert len(calls) == 3, "one per measure — not one per row"


def test_a_single_metric_leaderboard_still_ranks_the_same(org):
    result = _run(org, _model(
        [CLEARED], operation="leaderboard", subject=None, subject_level="advisor",
        ordering=Ordering(metric=CLEARED, direction="desc", stated=True)))

    assert [r["name"] for r in result.rows] == ["Ahmed Raza", "Omar Farooq", "Sara Iqbal"]
    assert [r["value"] for r in result.rows] == [900, 500, 100]
