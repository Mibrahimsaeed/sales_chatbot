"""Generic REST access for hierarchy levels (hierarchy rework phase 2) —
GET /api/hierarchy/{level}. Auth is bypassed via dependency override, same
approach the `client` fixture already uses for get_db (app.main.app has no
built-in test-auth bypass otherwise)."""

import pytest

from app.core.dependencies import get_current_user
from app.database.models import Advisor, SalesFunnel
from app.main import app


@pytest.fixture()
def auth_client(client):
    app.dependency_overrides[get_current_user] = lambda: {"sub": "test-user"}
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def _seed(db):
    db.add(Advisor(wid=1, name="Advisor One", team="Zonal North", company="Graana", bm="Zeeshan Tariq", rm="Zeeshan Tariq", portfolio_lead="Zonal North"))
    db.add(SalesFunnel(wid=1, mtd_new_connect=10, mtd_followup_connect=0))
    db.add(Advisor(wid=2, name="Advisor Two", team="Downtown", company="IMARAT", bm="Zeeshan Tariq", rm="Zeeshan Tariq", portfolio_lead="Zonal North"))
    db.add(SalesFunnel(wid=2, mtd_new_connect=20, mtd_followup_connect=0))
    db.commit()


def test_get_hierarchy_entity_returns_nested_breakdown(auth_client, db_session):
    _seed(db_session)

    response = auth_client.get("/api/hierarchy/unit_head", params={"value": "Zeeshan Tariq"})

    assert response.status_code == 200
    data = response.json()
    assert data["advisors"] == 2
    assert {t["team"] for t in data["teams"]} == {"Zonal North"}


def test_get_hierarchy_entity_flat_query_param_returns_ungrouped_list(auth_client, db_session):
    _seed(db_session)

    response = auth_client.get("/api/hierarchy/unit_head", params={"value": "Zeeshan Tariq", "flat": "true"})

    assert response.status_code == 200
    data = response.json()
    assert "teams" not in data
    assert {a["name"] for a in data["advisor_list"]} == {"Advisor One", "Advisor Two"}


def test_get_hierarchy_entity_invalid_level_400s(auth_client, db_session):
    response = auth_client.get("/api/hierarchy/not_a_real_level", params={"value": "X"})
    assert response.status_code == 400


def test_get_hierarchy_entity_advisor_level_400s(auth_client, db_session):
    """advisor lookup already has its own endpoint (GET /advisor?name=) —
    deliberately not accepted here."""
    response = auth_client.get("/api/hierarchy/advisor", params={"value": "X"})
    assert response.status_code == 400


def test_get_hierarchy_entity_unknown_value_404s(auth_client, db_session):
    _seed(db_session)
    response = auth_client.get("/api/hierarchy/unit_head", params={"value": "Nobody Real"})
    assert response.status_code == 404


def test_leaderboard_invalid_level_400s(auth_client, db_session):
    response = auth_client.get("/api/leaderboard", params={"metric": "mtd_cleared", "level": "not_a_real_level"})
    assert response.status_code == 400


def test_leaderboard_new_hierarchy_level_works(auth_client, db_session):
    _seed(db_session)
    response = auth_client.get("/api/leaderboard", params={"metric": "total_connects", "level": "unit_head"})
    assert response.status_code == 200
    rows = response.json()
    assert rows[0]["name"] == "Zeeshan Tariq"
    assert rows[0]["value"] == 30
