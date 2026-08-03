import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.session import Base
from app.core.dependencies import get_db

# import models so their tables register on Base.metadata before create_all
import app.database.models  # noqa: F401


@pytest.fixture(autouse=True)
def _no_live_semantic_retrieval(monkeypatch):
    """Part 8: semantic_retrieval.py is a real network call site
    (embeddings), but only test_semantic_retrieval.py should ever exercise
    it (with a mocked embed_texts) — every other test's "unresolved" plan
    path must stay fully offline like it always has. test_semantic_
    retrieval.py's own fixture re-enables this per test, which runs after
    this conftest-level fixture for the same (function) scope."""
    from app.llm import semantic_retrieval

    monkeypatch.setattr(semantic_retrieval.settings, "semantic_retrieval_enabled", False)


@pytest.fixture(autouse=True)
def _no_live_entity_linking(monkeypatch):
    """Same reasoning as _no_live_semantic_retrieval above, for the
    embedding-based entity linker (Part 9) — entity_extractor.py and
    ir_validator.py both call entity_linker.semantic_candidates() as a
    fallback, a real network call site. Only test_entity_linker.py should
    exercise it for real (with a mocked embed_texts); every other test's
    fuzzy-floor-miss path must stay fully offline."""
    from app.llm import entity_linker

    monkeypatch.setattr(entity_linker.settings, "entity_linking_enabled", False)


# app/services/advisor_service.py reads the advisor_profile VIEW, which is
# created by an Alembic migration rather than by Base.metadata — so it did
# not exist in the in-memory test DB and every advisor-lookup path was
# silently untestable (any test touching it died on "no such table").
# Mirrors migrations/0006's definition, in SQLite-compatible form: the
# enum comparison is a plain string here since SQLAlchemy stores
# PerformancePeriod by value.
_ADVISOR_PROFILE_VIEW = """
CREATE VIEW advisor_profile AS
SELECT
    a.wid, a.name, a.company, a.team, a.region, a.office,
    a.portfolio_lead, a.management_lead, a.bm, a.zm, a.rm,
    sf.mtd_new_connect, sf.mtd_followup_connect, sf.system_connect,
    sf.mtd_cr, sf.mtd_new_meeting, sf.mtd_followup_meeting, sf.mtd_conversion,
    p.pipeline, p.overdue,
    att.biometric_time, att.biometric_status, att.login_time, att.login_status,
    mtd.target AS mtd_target, mtd.cleared AS mtd_cleared, mtd.pct AS mtd_pct,
    ytd.target AS ytd_target, ytd.cleared AS ytd_cleared, ytd.pct AS ytd_pct,
    port.value AS portfolio_value, port.retention_pct AS portfolio_retention_pct,
    b.confirmed AS npr_confirmed, b.expected AS npr_expected,
    c.answered_calls_mtd
FROM advisors a
LEFT JOIN sales_funnel sf ON sf.wid = a.wid
LEFT JOIN pipeline p ON p.wid = a.wid
LEFT JOIN attendance att ON att.wid = a.wid
LEFT JOIN performance mtd ON mtd.wid = a.wid AND mtd.period = 'MTD'
LEFT JOIN performance ytd ON ytd.wid = a.wid AND ytd.period = 'YTD'
LEFT JOIN portfolio port ON port.wid = a.wid
LEFT JOIN bookings b ON b.wid = a.wid
LEFT JOIN calls c ON c.wid = a.wid
WHERE a.in_master_sheet = 1;
"""


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(_ADVISOR_PROFILE_VIEW)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture()
def client(db_session):
    from fastapi.testclient import TestClient
    from app.main import app

    def _get_db_override():
        yield db_session

    app.dependency_overrides[get_db] = _get_db_override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)
