""""Connects of advisors under X" asks for connects.

THE DEFECT. `_is_population_query` returned True for ANY hierarchy read:

    if ir.is_hierarchy_read():
        return True

A population is metric-free by definition, so the query ran down the
path that joins no fact table, and every row came back with value=None.
The failure was quiet in the worst way — the column was still THERE, with
its right heading, and only the number was missing:

    total_connects      value=None   display="—"    "Total Connects"
    answered_calls      value=89.0   display="89"   "Answered Calls"
    answered_calls_rate value=34.2   display="34.2%" "Answered Calls % of Target"

Two of the three figures were right, because `_attach_bundle_columns`
re-reads the companions per row but deliberately REUSES `row["value"]`
for the primary (fetching it twice is how a row's headline and its own
column start to disagree). So the primary alone carried the null.

Everything upstream was correct: the parser named the measure, and
`primary_metric()` resolved it. The rule that had been lost is the one
`_is_population_query`'s own docstring states — a population is the
ABSENCE of a measure — which the hierarchy branch short-circuited past.

Restoring it moves these queries onto the metric path, and that path had
never applied `_apply_hierarchy_scope`, which is where `relation` and the
self-exclusion live. Both halves are needed: without the second,
"directly under" silently widened to the whole subtree and a manager
appeared among their own reports.

The fixture is a two-level org: a Zonal Head over two BCMs, one of whom
is also their own BCM, so self-exclusion is exercised rather than
assumed, and `direct` and `subtree` have deliberately different answers.
"""

import pytest

from app.database.models import (
    Advisor, Attendance, Calls, Performance, PerformancePeriod, SalesFunnel,
)
from app.llm import entity_extractor
from app.llm.ir_validator import validate_ir
from app.llm.query_compiler import compile_and_run, _is_population_query
from app.llm.query_ir import MetricRef, QueryIR, Sort, Subject

# zonal_head -> bcm -> advisor.  Zed heads the zone.
#   Zed is his own BCM for Direct One/Two   -> they are his DIRECT reports
#   Bee is a BCM under Zed                  -> Bee's people are subtree-only
#   Outsider sits under a different zone entirely
_ORG = [
    # wid, name,        zonal(portfolio_lead), bcm(management_lead), connects
    (1, "Zed",         "Zed",   "Zed",   50),
    (2, "Direct One",  "Zed",   "Zed",   10),
    (3, "Direct Two",  "Zed",   "Zed",   20),
    (4, "Bee",         "Zed",   "Bee",   30),
    (5, "Under Bee",   "Zed",   "Bee",   40),
    (6, "Outsider",    "Otto",  "Otto",  99),
]
_SUBTREE = {"Direct One", "Direct Two", "Bee", "Under Bee"}   # Zed excluded
_DIRECT = {"Direct One", "Direct Two"}


@pytest.fixture()
def org(db_session):
    for wid, name, zonal, bcm, connects in _ORG:
        db_session.add(Advisor(wid=wid, name=name, team="Team Alpha",
                               company="Acme Holdings", rm="UH One",
                               portfolio_lead=zonal, management_lead=bcm,
                               in_master_sheet=True))
        # One row per fact table, so the parametrised measures below
        # exercise DIFFERENT sources (Calls, SalesFunnel, Performance,
        # Attendance) rather than four readings of one.
        db_session.add(Calls(wid=wid, connects_mtd=connects,
                             answered_calls_mtd=connects))
        db_session.add(SalesFunnel(wid=wid, mtd_new_meeting=connects,
                                   mtd_followup_meeting=0, mtd_cr=connects,
                                   # `total_connects` is this table's
                                   # additive pair; the Calls row above
                                   # still supplies answered_calls_mtd.
                                   mtd_new_connect=connects,
                                   mtd_followup_connect=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   cleared=connects * 100))
        db_session.add(Attendance(wid=wid, biometric_mtd_ontime=connects,
                                  biometric_mtd_late=0,
                                  biometric_mtd_not_marked=0))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _ir(metric="total_connects", relation="subtree", target="advisor", **over):
    kw = dict(
        intent="filtered_list", operation="filtered_list", subject_level="advisor",
        subjects=[Subject(type="zonal_head", value="Zed", match_confidence=1.0)],
        metrics=[MetricRef(key=metric, confidence=0.95)] if metric else [],
        sort=Sort(metric=metric, direction="desc") if metric else Sort(metric=None),
        target_level=target, subject_of="zonal_head", relation=relation, limit=None,
    )
    kw.update(over)
    return QueryIR(**kw)


def _rows(db, ir):
    return compile_and_run(db, validate_ir(ir, db).ir) or []


# =====================================================================
# The measure survives
# =====================================================================
def test_connects_under_a_manager_are_populated(org):
    """THE bug, in its reported form."""
    rows = _rows(org, _ir())
    assert rows, "no rows at all"
    assert all(r["value"] is not None for r in rows), \
        [r["name"] for r in rows if r["value"] is None]
    assert {r["name"] for r in rows} == _SUBTREE


def test_the_values_are_the_right_numbers(org):
    """Populated is not enough — they must be each person's own figure."""
    got = {r["name"]: r["value"] for r in _rows(org, _ir())}
    assert got == {"Direct One": 10, "Direct Two": 20, "Bee": 30, "Under Bee": 40}


@pytest.mark.parametrize("metric", [
    "total_connects", "total_meetings", "mtd_cleared", "attendance_rate",
    "answered_calls", "client_registrations",
])
def test_any_measure_under_a_manager_is_populated(org, metric):
    """Counts, amounts and rates alike — the branch was metric-blind, so
    the fix must be too."""
    rows = _rows(org, _ir(metric=metric))
    assert rows
    assert all(r["value"] is not None for r in rows), metric


def test_a_ranked_read_is_actually_ranked(org):
    """"top advisors under X by connects" was ranked by nothing."""
    rows = _rows(org, _ir(limit=3))
    values = [r["value"] for r in rows]
    assert values == sorted(values, reverse=True)
    assert rows[0]["name"] == "Under Bee"


# =====================================================================
# What must not change
# =====================================================================
def test_a_measureless_hierarchy_read_is_still_a_population(org):
    """"who reports to X" names nothing to rank by and must keep the
    metric-free path — that path joins no fact table, so a person with no
    calls record still appears."""
    ir = _ir(metric=None)
    assert validate_ir(ir, org).ir.primary_metric() is None
    assert _is_population_query(validate_ir(ir, org).ir) is True
    rows = _rows(org, ir)
    assert {r["name"] for r in rows} == _SUBTREE
    assert all(r["value"] is None for r in rows)


def test_a_measure_bearing_read_is_not_a_population(org):
    assert _is_population_query(validate_ir(_ir(), org).ir) is False


def test_direct_and_subtree_still_mean_different_things(org):
    """The second half of the fix. Without `_apply_hierarchy_scope` on
    the metric path, the subject filter's plain column match returned the
    whole subtree for both."""
    subtree = {r["name"] for r in _rows(org, _ir(relation="subtree"))}
    direct = {r["name"] for r in _rows(org, _ir(relation="direct"))}
    assert subtree == _SUBTREE
    assert direct == _DIRECT
    assert direct < subtree


def test_the_manager_is_not_one_of_their_own_reports(org):
    """Zed is his own BCM for two advisors, so a scope that forgets to
    exclude him counts him among the people beneath him."""
    for relation in ("subtree", "direct"):
        names = {r["name"] for r in _rows(org, _ir(relation=relation))}
        assert "Zed" not in names, relation


def test_nobody_outside_the_manager_appears(org):
    for relation in ("subtree", "direct"):
        names = {r["name"] for r in _rows(org, _ir(relation=relation))}
        assert "Outsider" not in names, relation


def test_the_metric_and_population_paths_agree_on_who_is_in_scope(org):
    """The two paths must not disagree about membership — only about
    whether a measure is attached. (Modulo the fact-table join, which the
    fixture gives every advisor a row for, so here they match exactly.)"""
    with_metric = {r["name"] for r in _rows(org, _ir())}
    without = {r["name"] for r in _rows(org, _ir(metric=None))}
    assert with_metric == without


def test_a_non_hierarchy_query_is_untouched(org):
    """`_apply_hierarchy_scope` returns immediately unless the IR is a
    hierarchy read, so an ordinary ranking is unaffected."""
    ir = QueryIR(intent="leaderboard", operation="leaderboard", subject_level="advisor",
                 subjects=[], metric=MetricRef(key="total_connects", confidence=0.95),
                 sort=Sort(metric="total_connects", direction="desc"))
    names = {r["name"] for r in _rows(org, ir)}
    assert names == {n for _, n, _, _, _ in _ORG}


def test_a_bcm_level_read_scopes_to_that_level(org):
    """The target level is not always the leaf."""
    ir = _ir(target="bcm", subject_level="bcm")
    rows = _rows(org, ir)
    assert {r["name"] for r in rows} == {"Bee"}
    assert all(r["value"] is not None for r in rows)
