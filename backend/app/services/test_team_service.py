from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.services.team_service import get_team_summary


def test_non_master_sheet_advisor_excluded_from_team_summary(db_session):
    db_session.add(Advisor(wid=1, name="Real", team="Alpha", company="IMARAT", in_master_sheet=True))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=10, mtd_followup_connect=0))
    db_session.add(Advisor(wid=2, name="Ghost", team="Alpha", company="IMARAT", in_master_sheet=False))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=1000, mtd_followup_connect=0))
    db_session.commit()

    summary = get_team_summary(db_session, "Alpha")

    assert summary["advisors"] == 1
    assert summary["connects"] == 10


def test_team_summary_includes_both_mtd_and_ytd_performance(db_session):
    # bug fix: team summaries never queried Performance at all before,
    # so YTD (and even MTD cleared/target) had nothing to report
    db_session.add(Advisor(wid=1, name="A", team="Alpha", in_master_sheet=True))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, target=1000, cleared=500))
    db_session.add(Performance(wid=1, period=PerformancePeriod.YTD, target=12000, cleared=8000))
    db_session.commit()

    summary = get_team_summary(db_session, "Alpha")

    assert summary["mtd_target"] == 1000
    assert summary["mtd_cleared"] == 500
    assert summary["ytd_target"] == 12000
    assert summary["ytd_cleared"] == 8000
