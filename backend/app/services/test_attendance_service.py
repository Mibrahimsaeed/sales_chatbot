from app.database.models import Advisor, Attendance
from app.services.attendance_service import get_attendance_by_status, get_attendance_issues


def _seed(db_session):
    db_session.add(Advisor(wid=1, name="Real Late", team="Alpha", in_master_sheet=True))
    db_session.add(Attendance(wid=1, biometric_status="Late"))
    db_session.add(Advisor(wid=2, name="Ghost Late", team="Alpha", in_master_sheet=False))
    db_session.add(Attendance(wid=2, biometric_status="Late"))
    db_session.commit()


def test_get_attendance_issues_excludes_non_master_sheet(db_session):
    _seed(db_session)
    rows = get_attendance_issues(db_session)
    assert [r["name"] for r in rows] == ["Real Late"]


def test_get_attendance_by_status_excludes_non_master_sheet(db_session):
    _seed(db_session)
    rows = get_attendance_by_status(db_session, status="Late")
    assert [r["name"] for r in rows] == ["Real Late"]
