import pytest

from app.core.exception import NotFoundError
from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.services.hierarchy_service import get_level_breakdown, get_level_flat_list, get_level_summary


def _seed(db):
    # PHASE 3: a breakdown now nests by the CHILD of the queried level,
    # so a Unit Head breaks down by Zonal Head — the structure the org
    # actually has — instead of skipping two layers to reach team.
    # Zeeshan Tariq (Unit Head) oversees two Zonal Heads, plus a
    # raw-data-only ghost that must be excluded (in_master_sheet=False).
    db.add(Advisor(wid=1, name="Advisor One", team="Blue Area", company="Graana", bm="Zeeshan Tariq", rm="Zeeshan Tariq",
                   portfolio_lead="Zonal North", in_master_sheet=True))
    db.add(SalesFunnel(wid=1, mtd_new_connect=10, mtd_followup_connect=0))
    db.add(Performance(wid=1, period=PerformancePeriod.MTD, target=1000, cleared=500))
    db.add(Performance(wid=1, period=PerformancePeriod.YTD, target=12000, cleared=8000))

    db.add(Advisor(wid=2, name="Advisor Two", team="Downtown", company="IMARAT", bm="Zeeshan Tariq", rm="Zeeshan Tariq",
                   portfolio_lead="Zonal South", in_master_sheet=True))
    db.add(SalesFunnel(wid=2, mtd_new_connect=20, mtd_followup_connect=0))
    db.add(Performance(wid=2, period=PerformancePeriod.MTD, target=2000, cleared=1000))

    db.add(Advisor(wid=3, name="Ghost", team="Downtown", company="IMARAT", bm="Zeeshan Tariq", rm="Zeeshan Tariq",
                   portfolio_lead="Zonal South", in_master_sheet=False))
    db.add(SalesFunnel(wid=3, mtd_new_connect=900, mtd_followup_connect=0))

    db.commit()


def test_get_level_summary_aggregates_across_teams(db_session):
    _seed(db_session)

    summary = get_level_summary(db_session, "unit_head", "Zeeshan Tariq")

    assert summary["level"] == "unit_head"
    assert summary["level_label"] == "Unit Head"
    assert summary["advisors"] == 2          # ghost excluded
    assert summary["connects"] == 30         # 10 + 20, not +900 ghost
    assert summary["mtd_target"] == 3000
    assert summary["mtd_cleared"] == 1500
    assert summary["ytd_target"] == 12000
    assert summary["ytd_cleared"] == 8000


def test_get_level_summary_raises_not_found_for_unknown_value(db_session):
    _seed(db_session)
    with pytest.raises(NotFoundError):
        get_level_summary(db_session, "unit_head", "Nobody Real")


def test_get_level_breakdown_nests_advisors_by_the_child_level(db_session):
    _seed(db_session)

    breakdown = get_level_breakdown(db_session, "unit_head", "Zeeshan Tariq")

    assert breakdown["advisors"] == 2
    teams_by_name = {t["team"]: t for t in breakdown["teams"]}
    assert set(teams_by_name) == {"Zonal North", "Zonal South"}
    assert teams_by_name["Zonal North"]["advisor_count"] == 1
    assert teams_by_name["Zonal North"]["advisors"][0]["name"] == "Advisor One"
    assert teams_by_name["Zonal North"]["advisors"][0]["mtd_cleared"] == 500
    assert teams_by_name["Zonal South"]["advisor_count"] == 1
    assert teams_by_name["Zonal South"]["advisors"][0]["name"] == "Advisor Two"


def test_get_level_breakdown_raises_not_found_for_unknown_value(db_session):
    _seed(db_session)
    with pytest.raises(NotFoundError):
        get_level_breakdown(db_session, "zonal_head", "Nobody Real")


def test_get_level_summary_works_generically_for_business_center(db_session):
    db_session.add(Advisor(wid=10, name="BC Advisor", team="Alpha", office="F-11 Business Center", in_master_sheet=True))
    db_session.add(SalesFunnel(wid=10, mtd_new_connect=5, mtd_followup_connect=0))
    db_session.commit()

    summary = get_level_summary(db_session, "business_center", "F-11 Business Center")
    assert summary["advisors"] == 1
    assert summary["connects"] == 5


# ---- Phase 2: flat opt-in ----

def test_get_level_flat_list_is_ungrouped(db_session):
    _seed(db_session)

    flat = get_level_flat_list(db_session, "unit_head", "Zeeshan Tariq")

    assert flat["advisors"] == 2   # top-line COUNT, same field as get_level_summary/breakdown
    assert "teams" not in flat
    advisor_rows = flat["advisor_list"]
    assert {(r["name"], r["team"]) for r in advisor_rows} == {("Advisor One", "Zonal North"), ("Advisor Two", "Zonal South")}
    assert {r["mtd_cleared"] for r in advisor_rows} == {500, 1000}


def test_get_level_flat_list_raises_not_found_for_unknown_value(db_session):
    _seed(db_session)
    with pytest.raises(NotFoundError):
        get_level_flat_list(db_session, "unit_head", "Nobody Real")
