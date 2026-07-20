"""Deterministic benchmark dataset. Every number is chosen so expected
query results are hand-computable — the YAML cases' expect.sql values are
derived from this table by inspection, not by running the code.

Rankings this data pins down (no ties anywhere, so first_row is stable):
- mtd_cleared desc:      Waqar 900 > Ali 700 > Bilal 500 > Sana 300 > Zara 200 > Omar 100
- ytd_cleared desc:      Ali 9000 > Waqar 8000 > Sana 6000 > Bilal 4000 > Zara 2000 > Omar 1000
- achievement_pct desc:  Waqar 90 > Ali 87.5 > Sana 75 > Bilal 62.5 > Zara 50 > Omar 25
- total_connects desc:   Waqar 30 > Ali 25 > Sana 20 > Bilal 15 > Zara 10 > Omar 5
- new_connects desc:     Waqar 20 > Ali 17 > Sana 13 > Bilal 10 > Zara 6 > Omar 3
- followup_connects:     Waqar 10 > Ali 8 > Sana 7 > Bilal 5 > Zara 4 > Omar 2
- total_meetings desc:   Waqar 12 > Ali 10 > Sana 8 > Bilal 6 > Zara 4 > Omar 2
- attendance_rate desc:  Waqar 90 > Ali 80 > Sana 70 > Bilal 60 > Zara 50 > Omar 40
- late_count desc:       Omar 6 > Zara 5 > Bilal 4 > Sana 3 > Ali 2 > Waqar 1
- overdue desc:          Omar 600 > Zara 500 > Bilal 400 > Sana 300 > Ali 200 > Waqar 100
- teams: Blue Area = Waqar+Ali, Downtown = Sana+Bilal, DHA Phase 5 = Zara+Omar
  - mtd_cleared rollup:   Blue Area 1600 > Downtown 800 > DHA Phase 5 300
  - total_meetings:       Blue Area 22 > Downtown 14 > DHA Phase 5 6
  - attendance_rate avg:  Blue Area 85 > Downtown 65 > DHA Phase 5 45
  - overdue rollup:       DHA Phase 5 1100 > Downtown 700 > Blue Area 300
  - TeamTarget achievement_pct: Blue Area 80 > Downtown 66.7 > DHA Phase 5 30
- companies: Graana = Waqar+Ali+Sana (mtd 1900), IMARAT = Bilal+Zara (700), Agency21 = Omar (100)
"""

from app.database.models import (
    Advisor, Attendance, Performance, PerformancePeriod, Pipeline, SalesFunnel, TeamTarget,
)

ADVISORS = [
    # wid, name, team, company, mtd_cleared, mtd_target, ytd_cleared,
    # new_connect, followup_connect, new_meeting, followup_meeting,
    # ontime, late, not_marked, overdue
    (1, "Waqar Haider", "Blue Area", "Graana", 900, 1000, 8000, 20, 10, 8, 4, 18, 1, 1, 100),
    (2, "Ali Raza", "Blue Area", "Graana", 700, 800, 9000, 17, 8, 6, 4, 16, 2, 2, 200),
    (3, "Sana Khan", "Downtown", "Graana", 300, 400, 6000, 13, 7, 5, 3, 14, 3, 3, 300),
    (4, "Bilal Ahmed", "Downtown", "IMARAT", 500, 800, 4000, 10, 5, 4, 2, 12, 4, 4, 400),
    (5, "Zara Malik", "DHA Phase 5", "IMARAT", 200, 400, 2000, 6, 4, 3, 1, 10, 5, 5, 500),
    (6, "Omar Farooq", "DHA Phase 5", "Agency21", 100, 400, 1000, 3, 2, 1, 1, 8, 6, 6, 600),
]

TEAM_TARGETS = [
    ("Blue Area", 2000, 1600, 80.0),
    ("Downtown", 1200, 800, 66.7),
    ("DHA Phase 5", 1000, 300, 30.0),
]


def seed(db) -> None:
    for (wid, name, team, company, mtd_cleared, mtd_target, ytd_cleared,
         new_c, fol_c, new_m, fol_m, ontime, late, not_marked, overdue) in ADVISORS:
        db.add(Advisor(wid=wid, name=name, team=team, company=company))
        db.add(Performance(
            wid=wid, period=PerformancePeriod.MTD,
            cleared=mtd_cleared, target=mtd_target,
            pct=mtd_cleared * 100.0 / mtd_target,
        ))
        db.add(Performance(wid=wid, period=PerformancePeriod.YTD, cleared=ytd_cleared, target=12000))
        db.add(SalesFunnel(
            wid=wid, mtd_new_connect=new_c, mtd_followup_connect=fol_c,
            mtd_new_meeting=new_m, mtd_followup_meeting=fol_m,
        ))
        db.add(Attendance(
            wid=wid, biometric_mtd_ontime=ontime, biometric_mtd_late=late,
            biometric_mtd_not_marked=not_marked,
            biometric_status="On Time" if late <= 2 else "Late",
        ))
        db.add(Pipeline(wid=wid, pipeline=overdue * 2, overdue=overdue))
    for team, target, achieved, pct in TEAM_TARGETS:
        db.add(TeamTarget(team=team, target=target, achieved=achieved, achievement_pct=pct))
    db.commit()
