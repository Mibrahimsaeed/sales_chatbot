"""Smoke tests for the conftest.py db_session/client fixtures themselves —
if these fail, every future test built on top of them is untrustworthy."""

from app.database.models import Advisor


def test_db_session_fixture_creates_tables_and_persists_rows(db_session):
    db_session.add(Advisor(wid=1, name="Test Advisor", team="Alpha"))
    db_session.commit()

    fetched = db_session.query(Advisor).filter_by(wid=1).one()
    assert fetched.name == "Test Advisor"


def test_client_fixture_overrides_get_db_and_serves_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
