"""Phase 27 — an advisor who leaves the MasterSheet stops counting.

THE GAP. Loading is upsert-only, so a row is touched only when the
payload contains it. An advisor who leaves the sheet is never emitted
again and never updated — their row keeps `in_master_sheet=True` forever,
carrying the hierarchy they had on their last day. 107 such rows had
accumulated in production, and because `in_master_sheet` is the filter
the whole system reads, they still counted toward every team size: one
Unit Head reported 89 advisors where 77 remained.

It self-heals for someone still visible in an ACTIVITY tab — they are
re-emitted with the flag False and the upsert corrects them. All 107
appeared in no tab at all, which is why they survived. That self-healing
path is pinned below so the new sweep cannot replace it by accident.

DEACTIVATE, NEVER DELETE. The flag already gates every scope, so flipping
it removes these people from every answer without touching a row, a fact
table, or AdvisorHistory — and it reverses itself if they come back.

THE FLOOR is the dangerous part and gets the most tests. "Absent from the
payload" only means "gone" when the payload is COMPLETE; a truncated
fetch is indistinguishable from a mass departure, and acting on one would
empty the roster in a single run.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import Advisor, Calls, Performance, PerformancePeriod
from app.database.session import Base
from etl.load import _RECONCILE_MIN_COVERAGE, reconcile_master_sheet
from etl.validation import check_stale_master_sheet_flags


@pytest.fixture()
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _seed(db, count=10, flagged=True):
    """`count` advisors, each with a fact row, all on the sheet."""
    for wid in range(1, count + 1):
        db.add(Advisor(wid=wid, name=f"Advisor {wid}", team="Alpha",
                       company="Graana", rm="Unit One", in_master_sheet=flagged))
        db.add(Calls(wid=wid, connects_mtd=10, connects_daily=1,
                     answered_calls_mtd=5, answered_calls_daily=1))
        db.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                           target=100, cleared=50))
    db.commit()


def _payload(wids):
    """A transform() `advisors` payload listing exactly `wids` as on-sheet."""
    return [{"wid": w, "in_master_sheet": True} for w in wids]


def _flagged(db):
    return {a.wid for a in db.query(Advisor).filter(Advisor.in_master_sheet.is_(True))}


# ---------------------------------------------------------------------
# The reconciliation
# ---------------------------------------------------------------------


def test_an_advisor_absent_from_the_new_payload_is_deactivated(db):
    """Sync 1 lists 10; sync 2 lists 9. The tenth stops counting."""
    _seed(db, 10)
    deactivated = reconcile_master_sheet(db, _payload(range(1, 10)))
    db.commit()

    assert deactivated == 1
    assert _flagged(db) == set(range(1, 10))
    assert db.get(Advisor, 10).in_master_sheet is False


def test_the_advisor_row_is_kept(db):
    _seed(db, 10)
    reconcile_master_sheet(db, _payload(range(1, 10)))
    db.commit()

    departed = db.get(Advisor, 10)
    assert departed is not None
    assert departed.name == "Advisor 10"


def test_the_fact_and_history_rows_are_untouched(db):
    """Deactivation must never cost data. Their past stays inspectable."""
    _seed(db, 10)
    reconcile_master_sheet(db, _payload(range(1, 10)))
    db.commit()

    assert db.get(Calls, 10).connects_mtd == 10
    perf = db.query(Performance).filter(Performance.wid == 10).all()
    assert len(perf) == 1 and perf[0].cleared == 50


def test_the_hierarchy_fields_are_left_as_they_were(db):
    """They record where the person sat when they left, which is true.
    Only the flag is written."""
    _seed(db, 10)
    reconcile_master_sheet(db, _payload(range(1, 10)))
    db.commit()

    departed = db.get(Advisor, 10)
    assert departed.rm == "Unit One"
    assert departed.team == "Alpha"


def test_an_unchanged_roster_deactivates_nobody(db):
    _seed(db, 10)
    assert reconcile_master_sheet(db, _payload(range(1, 11))) == 0
    assert _flagged(db) == set(range(1, 11))


def test_an_advisor_who_returns_is_reactivated_by_the_upsert(db):
    """Reversibility: the flag is data, not a tombstone. The upsert
    rewrites it to True when they reappear, so this only has to not
    interfere."""
    _seed(db, 10)
    reconcile_master_sheet(db, _payload(range(1, 10)))
    db.commit()
    assert db.get(Advisor, 10).in_master_sheet is False

    db.get(Advisor, 10).in_master_sheet = True     # what the upsert does
    db.commit()
    assert reconcile_master_sheet(db, _payload(range(1, 11))) == 0
    assert db.get(Advisor, 10).in_master_sheet is True


# ---------------------------------------------------------------------
# The safety floor
# ---------------------------------------------------------------------


def test_a_truncated_payload_deactivates_nobody(db):
    """THE dangerous case. A short fetch looks exactly like everyone
    leaving; acting on it would empty the roster in one run."""
    _seed(db, 100)
    assert reconcile_master_sheet(db, _payload(range(1, 4))) == 0
    assert len(_flagged(db)) == 100


def test_an_empty_payload_deactivates_nobody(db):
    _seed(db, 100)
    assert reconcile_master_sheet(db, _payload([])) == 0
    assert len(_flagged(db)) == 100


def test_attrition_just_inside_the_floor_still_reconciles(db):
    """The floor must not be so cautious that it never fires. 100 flagged,
    85 present — 85% coverage, above the 80% floor — deactivates 15."""
    _seed(db, 100)
    assert reconcile_master_sheet(db, _payload(range(1, 86))) == 15
    assert len(_flagged(db)) == 85


def test_attrition_just_outside_the_floor_is_refused(db):
    """100 flagged, 70 present — 70% coverage, below the floor. Refused
    whole rather than partially applied."""
    _seed(db, 100)
    assert reconcile_master_sheet(db, _payload(range(1, 71))) == 0
    assert len(_flagged(db)) == 100


def test_the_floor_is_a_share_not_a_fixed_count(db):
    """A small org must reconcile too — 10 flagged, 9 present is 90%."""
    _seed(db, 10)
    assert reconcile_master_sheet(db, _payload(range(1, 10))) == 1


def test_an_empty_database_is_not_an_error(db):
    assert reconcile_master_sheet(db, _payload([1, 2, 3])) == 0


def test_the_floor_constant_is_a_meaningful_share():
    assert 0.5 < _RECONCILE_MIN_COVERAGE < 1.0


# ---------------------------------------------------------------------
# The existing self-healing path is untouched
# ---------------------------------------------------------------------


def test_an_activity_only_advisor_is_not_disturbed(db):
    """Someone never on the sheet is already flagged False by
    ensure_advisor. The sweep only looks at rows flagged True, so it has
    nothing to say about them."""
    _seed(db, 5)
    db.add(Advisor(wid=99, name="Activity Only", team="Alpha",
                   in_master_sheet=False))
    db.commit()

    reconcile_master_sheet(db, _payload(range(1, 6)))
    db.commit()
    assert db.get(Advisor, 99).in_master_sheet is False
    assert len(_flagged(db)) == 5


# ---------------------------------------------------------------------
# Scope and metric consequences
# ---------------------------------------------------------------------


def test_a_deactivated_advisor_leaves_the_unit_head_headcount(db):
    """The reported defect: a Unit Head's team size counted people who
    had left."""
    _seed(db, 10)
    before = db.query(Advisor).filter(
        Advisor.rm == "Unit One", Advisor.in_master_sheet.is_(True)).count()
    reconcile_master_sheet(db, _payload(range(1, 9)))
    db.commit()
    after = db.query(Advisor).filter(
        Advisor.rm == "Unit One", Advisor.in_master_sheet.is_(True)).count()

    assert before == 10
    assert after == 8


def test_deactivating_a_fact_less_advisor_moves_no_metric(db):
    """94 of the 107 production orphans had no fact rows, so the cleanup
    is a headcount correction and must not shift a single total. Pinned
    so a future change cannot quietly make it one."""
    _seed(db, 10)
    db.add(Advisor(wid=50, name="No Facts", team="Alpha", rm="Unit One",
                   in_master_sheet=True))
    db.commit()

    def team_connects():
        return db.query(Calls.connects_mtd).join(
            Advisor, Advisor.wid == Calls.wid).filter(
            Advisor.rm == "Unit One", Advisor.in_master_sheet.is_(True)).all()

    before = sum(v for (v,) in team_connects())
    reconcile_master_sheet(db, _payload(range(1, 11)))   # drops wid 50 only
    db.commit()

    assert db.get(Advisor, 50).in_master_sheet is False
    assert sum(v for (v,) in team_connects()) == before


# ---------------------------------------------------------------------
# The validation check
# ---------------------------------------------------------------------


def test_a_clean_sync_reports_no_stale_flags(db):
    _seed(db, 10)
    reconcile_master_sheet(db, _payload(range(1, 10)))
    db.commit()
    assert check_stale_master_sheet_flags(db, set(range(1, 10))) is None


def test_stale_flags_are_reported_when_reconciliation_was_skipped(db):
    """The floor refusing is not silent — the roster is knowingly stale
    until the next complete fetch, and the report says so."""
    _seed(db, 100)
    reconcile_master_sheet(db, _payload(range(1, 4)))     # refused
    db.commit()

    finding = check_stale_master_sheet_flags(db, set(range(1, 4)))
    assert finding is not None
    assert finding.count == 97
    assert finding.severity == "warning"
    assert finding.sample


def test_the_check_is_skipped_without_a_payload(db):
    """The health endpoint has no sheet in hand; it must not report every
    advisor as stale."""
    _seed(db, 10)
    assert check_stale_master_sheet_flags(db, None) is None
