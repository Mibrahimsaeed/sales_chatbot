import pytest

from app.database.models import Advisor
from app.llm import entity_extractor
from app.llm.entity_extractor import extract_entities


@pytest.fixture()
def gazetteer_db(db_session):
    db_session.add_all([
        Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana"),
        Advisor(wid=2, name="Ali Raza", team="Downtown", company="IMARAT"),
        Advisor(wid=3, name="Sana Khan", team="DHA Phase 5", company="Agency21"),
    ])
    db_session.commit()
    # the module-level gazetteer cache survives between tests — reset it so
    # each test sees this fixture DB, not a previous test's
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def test_substring_company_match_gets_full_confidence(gazetteer_db):
    entities = extract_entities("show graana advisors", gazetteer_db)
    assert entities["company"] == "Graana"
    assert entities["company_matches"] == [{"value": "Graana", "score": 1.0}]


def test_typod_company_matches_fuzzily_with_score(gazetteer_db):
    entities = extract_entities("show grana advisors", gazetteer_db)
    assert entities["company"] == "Graana"
    assert entities["company_matches"][0]["score"] < 1.0


def test_typod_team_matches_fuzzily(gazetteer_db):
    entities = extract_entities("top performers in blue aera", gazetteer_db)
    assert entities["team"] == "Blue Area"
    assert entities["team_matches"][0]["score"] < 1.0


def test_multiple_teams_still_collected(gazetteer_db):
    entities = extract_entities("compare blue area with downtown", gazetteer_db)
    assert set(entities["teams"]) == {"Blue Area", "Downtown"}


def test_gibberish_matches_no_entities(gazetteer_db):
    entities = extract_entities("xyzzy plugh quux", gazetteer_db)
    assert "team" not in entities
    assert "company" not in entities


def test_advisor_fuzzy_match_populates_matches_list(gazetteer_db):
    entities = extract_entities("tell me about Waqar Hader", gazetteer_db)
    assert entities["advisor_name"] == "Waqar Haider"
    assert entities["advisor_matches"][0]["value"] == "Waqar Haider"
    assert 0 < entities["advisor_matches"][0]["score"] <= 1.0
