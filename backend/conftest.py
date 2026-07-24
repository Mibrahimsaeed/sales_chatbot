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


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
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
