"""
Fixtures for the golden-query suite.

DETERMINISM IS THE POINT. A regression suite that can change its answer
because a provider was reachable, out of quota, or slightly differently
tuned is not a regression suite. So every non-deterministic input is
closed off here:

  - the LLM semantic parser is forced unavailable, so understanding comes
    from the deterministic layer,
  - NLU_MODE is pinned to "rules_first" so routing does not depend on an
    env var,
  - embeddings are already disabled by the root conftest.

That means these cases pin the RULE-BASED understanding. That is the
right thing to pin: it is the layer that must work when the provider is
down, it is what every LLM answer degrades to, and it is the only layer
whose behaviour is reproducible in CI. Where the LLM would do better, the
case records what the deterministic layer does and says so.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import (
    Advisor, Attendance, Calls, Performance, PerformancePeriod, Pipeline,
    Portfolio, SalesFunnel, TeamTarget,
)
from app.database.session import Base


@pytest.fixture(autouse=True)
def _deterministic_nlu(monkeypatch):
    """No LLM, no mode ambiguity."""
    from app.core.config import settings
    from app.llm import llm_client, semantic_parser

    monkeypatch.setattr(settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(settings, "use_llm_planner", False, raising=False)
    # Both call sites, so neither a structured nor a plain call escapes.
    monkeypatch.setattr(llm_client, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _fresh_conversation_memory():
    """Each case is a first turn. Conversation memory is process-global,
    so without this a case could be read as a follow-up to whatever ran
    before it — and the order tests run in would change the answers."""
    from app.llm import conversation_memory

    conversation_memory.reset() if hasattr(conversation_memory, "reset") else None
    yield


# ---------------------------------------------------------------------
# The organisation
# ---------------------------------------------------------------------

# One coherent org covering every level of the verified chain plus the
# attributes, so entity grounding in the cases is realistic rather than
# incidental. Names are deliberately distinctive: a golden suite should
# not fail because two fixture names fuzzy-match each other.
#
# wid, name,             team,        company,   unit head,  zonal head, bcm,        office,        region
PEOPLE = [
    (1,  "Yasir Ali",     "Blue Area",  "Graana",  "Tariq Mehmood", "Fawad Hafeez", "Usman Ghani", "Beverly Center", "North/KPK"),
    (2,  "Waqar Haider",  "Blue Area",  "Graana",  "Tariq Mehmood", "Fawad Hafeez", "Usman Ghani", "Beverly Center", "North/KPK"),
    (3,  "Sana Tariq",    "Blue Area",  "Graana",  "Tariq Mehmood", "Fawad Hafeez", "Rabia Anjum", "Beverly Center", "North/KPK"),
    (4,  "Shehryar Abbasi", "Downtown", "Graana",  "Tariq Mehmood", "Fawad Hafeez", "Rabia Anjum", "Gold Crest",     "Central"),
    (5,  "Hina Malik",    "Downtown",   "Graana",  "Tariq Mehmood", "Adeel Aslam",  "Rabia Anjum", "Gold Crest",     "Central"),
    (6,  "Salman Arshad", "Gulberg",    "IMARAT",  "Sadia Rehman",  "Adeel Aslam",  "Kamran Shah", "Gold Crest",     "Central"),
    (7,  "Nadia Sheikh",  "Gulberg",    "IMARAT",  "Sadia Rehman",  "Adeel Aslam",  "Kamran Shah", "Emporium",       "South"),
    (8,  "Faisal Iqbal",  "GCC",        "IMARAT",  "Sadia Rehman",  "Adeel Aslam",  "Kamran Shah", "Emporium",       "South"),
    (9,  "Zainab Noor",   "GCC",        "IMARAT",  "Sadia Rehman",  "Adeel Aslam",  "Bilal Qadir", "Emporium",       "South"),
    (10, "Omar Farooq",   "GCC",        "IMARAT",  "Sadia Rehman",  "Adeel Aslam",  "Bilal Qadir", "Emporium",       "South"),
    # ---- deliberately AMBIGUOUS rows, for the ambiguity categories ----
    # "Kamran Shah" is already a BCM above. As an advisor name too, it
    # grounds at two hierarchy levels — the clarify_ambiguous case. Real
    # orgs do this constantly: a manager who also carries a book.
    (11, "Kamran Shah",   "Gulberg",    "IMARAT",  "Sadia Rehman",  "Adeel Aslam",  "Kamran Shah", "Emporium",       "South"),
    # Two different people with the same name — the clarify_person case.
    # Production has 238 such name groups, which is why identity is keyed
    # on wid rather than name.
    (12, "Ali Raza",      "Blue Area",  "Graana",  "Tariq Mehmood", "Fawad Hafeez", "Usman Ghani", "Beverly Center", "North/KPK"),
    (13, "Ali Raza",      "GCC",        "IMARAT",  "Sadia Rehman",  "Adeel Aslam",  "Bilal Qadir", "Emporium",       "South"),
]

# wid -> (mtd_cleared, ytd_cleared, pct, connects, cr, meetings, conversions,
#         answered_calls, pipeline, overdue, portfolio, ontime, late, status)
FACTS = {
    1:  (900, 9000, 90.0, 100, 50, 20, 5, 120, 5000, 0, 40000, 18, 2, "On Time"),
    2:  (800, 7500, 80.0,  90, 40, 18, 4, 110, 4500, 1, 35000, 17, 3, "On Time"),
    3:  (700, 6000, 70.0,  80, 30, 15, 3, 100, 4000, 2, 30000, 15, 5, "Late"),
    4:  (600, 5000, 60.0,  70, 25, 12, 3,  90, 3500, 2, 25000, 14, 6, "Late"),
    5:  (500, 4000, 50.0,  60, 20, 10, 2,  80, 3000, 3, 20000, 12, 8, "Late"),
    6:  (400, 3000, 40.0,  50, 15,  8, 2,  70, 2500, 4, 18000, 10, 10, "Absent"),
    7:  (300, 2500, 30.0,  40, 12,  6, 1,  60, 2000, 5, 15000,  9, 11, "On Time"),
    8:  (200, 2000, 20.0,  30, 10,  5, 1,  50, 1500, 6, 12000,  8, 12, "On Time"),
    9:  (150, 1500, 15.0,  20,  8,  4, 1,  40, 1000, 7, 10000,  7, 13, "Not Marked"),
    10: (100, 1000, 10.0,  10,  5,  2, 0,  30,  500, 8,  8000,  5, 15, "Not Marked"),
    11: (250, 2200, 25.0,  25,  9,  4, 1,  45, 1200, 5, 11000,  9, 11, "On Time"),
    12: (350, 3200, 35.0,  35, 11,  7, 2,  55, 1800, 3, 16000, 11,  9, "Late"),
    13: (450, 4200, 45.0,  45, 13,  9, 3,  65, 2200, 2, 19000, 13,  7, "On Time"),
}


@pytest.fixture(scope="session")
def _golden_engine():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    for wid, name, team, company, unit_head, zonal_head, bcm, office, region in PEOPLE:
        session.add(Advisor(
            wid=wid, name=name, team=team, company=company,
            rm=unit_head, portfolio_lead=zonal_head, management_lead=bcm,
            office=office, region=region, in_master_sheet=True,
        ))
        (cleared, ytd, pct, connects, cr, meetings, conversions,
         calls, pipeline, overdue, portfolio, ontime, late, status) = FACTS[wid]
        session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                target=1000, cleared=cleared, pct=pct))
        session.add(Performance(wid=wid, period=PerformancePeriod.YTD,
                                target=10000, cleared=ytd, pct=pct))
        session.add(SalesFunnel(
            wid=wid, mtd_new_connect=connects, mtd_followup_connect=0,
            mtd_cr=cr, mtd_new_meeting=meetings, mtd_followup_meeting=0,
            mtd_conversion=conversions, mtd_booking_stored=conversions,
        ))
        session.add(Pipeline(wid=wid, pipeline=pipeline, overdue=overdue))
        session.add(Portfolio(wid=wid, value=portfolio))
        session.add(Calls(wid=wid, answered_calls_mtd=calls, connects_mtd=connects))
        session.add(Attendance(
            wid=wid, biometric_mtd_ontime=ontime, biometric_mtd_late=late,
            biometric_mtd_not_marked=0, biometric_status=status,
        ))

    for team, target, achieved in (("Blue Area", 3000, 2400), ("Downtown", 2000, 1100),
                                   ("Gulberg", 1500, 700), ("GCC", 1200, 450)):
        session.add(TeamTarget(team=team, target=target, achieved=achieved,
                               achievement_pct=round(achieved / target * 100, 1)))
    session.commit()
    yield engine
    session.close()


@pytest.fixture()
def org(_golden_engine):
    """A session on the shared org, plus an expired gazetteer cache.

    The cache is process-global with a TTL, so a case could otherwise
    match against whatever org a previous test file loaded.
    """
    from app.llm import entity_extractor

    session = sessionmaker(bind=_golden_engine)()
    entity_extractor._cache["loaded_at"] = 0
    yield session
    session.close()
