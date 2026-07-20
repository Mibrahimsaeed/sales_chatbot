from app.database.models import AdvisorHistory, PerformancePeriod
from etl.history_snapshot import write_snapshot


def _transformed_data():
    return {
        "advisors": [{"wid": 1, "name": "Advisor One"}, {"wid": 2, "name": "Advisor Two"}],
        "performance": [
            {"wid": 1, "period": PerformancePeriod.MTD, "target": 1000, "cleared": 600, "pct": 60},
            {"wid": 1, "period": PerformancePeriod.YTD, "target": 12000, "cleared": 5000, "pct": 41},
            {"wid": 2, "period": PerformancePeriod.MTD, "target": 800, "cleared": 800, "pct": 100},
        ],
        "sales_funnel": [
            {"wid": 1, "mtd_new_connect": 10, "mtd_followup_connect": 5, "mtd_new_meeting": 3, "mtd_followup_meeting": 1},
        ],
        "pipeline": [
            {"wid": 1, "pipeline": 2000, "overdue": 300},
            {"wid": 2, "pipeline": 500, "overdue": 0},
        ],
    }


def test_writes_one_row_per_advisor_from_mtd_performance_only(db_session):
    written = write_snapshot(_transformed_data(), db=db_session)

    assert written == 2
    rows = {r.wid: r for r in db_session.query(AdvisorHistory).all()}
    assert set(rows) == {1, 2}
    # advisor 1's MTD row (cleared=600), not the YTD row (cleared=5000)
    assert rows[1].mtd_cleared == 600
    assert rows[1].mtd_target == 1000
    assert rows[1].connects == 15
    assert rows[1].meetings == 4
    assert rows[1].overdue == 300


def test_advisor_missing_from_a_source_table_gets_null_fields_not_a_crash(db_session):
    data = _transformed_data()
    # advisor 2 has no sales_funnel row at all
    written = write_snapshot(data, db=db_session)

    assert written == 2
    row = db_session.query(AdvisorHistory).filter_by(wid=2).one()
    assert row.mtd_cleared == 800
    assert row.connects is None
    assert row.meetings is None


def test_is_append_only_across_multiple_sync_runs(db_session):
    data = _transformed_data()
    write_snapshot(data, db=db_session)
    write_snapshot(data, db=db_session)

    assert db_session.query(AdvisorHistory).count() == 4
