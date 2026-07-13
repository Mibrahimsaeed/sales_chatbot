# from sqlalchemy.orm import Session
# from app.database.models import Advisor, Attendance


# def get_attendance_issues(db: Session, team: str | None = None, limit: int = 15) -> list[dict]:
#     q = (
#         db.query(Advisor.wid, Advisor.name, Advisor.team, Attendance.biometric_status, Attendance.login_status)
#         .join(Attendance, Attendance.wid == Advisor.wid)
#         .filter(Attendance.biometric_status.isnot(None))
#         .filter(Attendance.biometric_status != "On Time")
#     )
#     if team:
#         q = q.filter(Advisor.team.ilike(team))
#     rows = q.limit(limit).all()
#     return [
#         {"wid": r.wid, "name": r.name, "team": r.team, "biometric_status": r.biometric_status, "login_status": r.login_status}
#         for r in rows
#     ]


from sqlalchemy.orm import Session
from app.database.models import Advisor, Attendance


def get_attendance_issues(db: Session, team: str | None = None, limit: int = 15) -> list[dict]:
    q = (
        db.query(
            Advisor.wid,
            Advisor.name,
            Advisor.team,
            Attendance.biometric_status,
            Attendance.login_status
        )
        .join(Attendance, Attendance.wid == Advisor.wid)
        .filter(Attendance.biometric_status.isnot(None))
        .filter(Attendance.biometric_status != "On Time")
    )

    if team:
        q = q.filter(Advisor.team.ilike(team))

    rows = q.limit(limit).all()

    return [
        {
            "wid": r.wid,
            "name": r.name,
            "team": r.team,
            "biometric_status": r.biometric_status,
            "login_status": r.login_status
        }
        for r in rows
    ]


def get_attendance_by_status(
    db: Session,
    team: str | None = None,
    status: str = "Not Marked",
    limit: int = 50
) -> list[dict]:

    q = (
        db.query(
            Advisor.wid,
            Advisor.name,
            Advisor.team,
            Attendance.biometric_status,
            Attendance.login_status
        )
        .join(Attendance, Attendance.wid == Advisor.wid)
        .filter(
            Attendance.biometric_status == status
        )
    )

    if team:
        q = q.filter(
            Advisor.team.ilike(team)
        )

    rows = q.limit(limit).all()

    return [
        {
            "wid": r.wid,
            "name": r.name,
            "team": r.team,
            "biometric_status": r.biometric_status,
            "login_status": r.login_status
        }
        for r in rows
    ]