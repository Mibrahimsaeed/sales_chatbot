from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.services.company_service import get_company_summary


def test_non_master_sheet_advisor_excluded_from_company_summary(db_session):
    db_session.add(Advisor(wid=1, name="Real", team="Alpha", company="Graana", in_master_sheet=True))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=10, mtd_followup_connect=0))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, target=1000, cleared=500))

    db_session.add(Advisor(wid=2, name="Ghost", team="Alpha", company="Graana", in_master_sheet=False))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=1000, mtd_followup_connect=0))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, target=9000, cleared=9000))
    db_session.commit()

    summary = get_company_summary(db_session, "Graana")

    assert summary["advisors"] == 1
    assert summary["connects"] == 10
    assert summary["mtd_cleared"] == 500
    assert summary["mtd_target"] == 1000


def test_company_summary_includes_ytd_performance(db_session):
    # bug fix: this used to only ever query MTD Performance rows
    db_session.add(Advisor(wid=1, name="Real", team="Alpha", company="Graana", in_master_sheet=True))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, target=1000, cleared=500))
    db_session.add(Performance(wid=1, period=PerformancePeriod.YTD, target=12000, cleared=8000))
    db_session.commit()

    summary = get_company_summary(db_session, "Graana")

    assert summary["ytd_target"] == 12000
    assert summary["ytd_cleared"] == 8000
