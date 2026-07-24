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


def test_non_master_sheet_advisor_and_team_are_excluded_from_gazetteer(gazetteer_db):
    # a raw-data-only WID (never on the MasterSheet) must not ground a
    # query, or get fuzzy-matched against, as if it were real org data
    gazetteer_db.add(Advisor(
        wid=99, name="Raw Data Ghost", team="Ghost Team", company="Ghost Co",
        in_master_sheet=False,
    ))
    gazetteer_db.commit()
    entity_extractor._cache["loaded_at"] = 0

    entities = extract_entities("show ghost team advisors from ghost co", gazetteer_db)
    assert "team" not in entities
    assert "company" not in entities

    entities = extract_entities("tell me about Raw Data Ghost", gazetteer_db)
    assert entities.get("advisor_name") != "Raw Data Ghost"


# ---- semantic fallback (Part 9) — entity_linking_enabled is forced off
# globally by conftest's autouse fixture, so these re-enable it locally
# and mock entity_linker.semantic_candidates() directly rather than the
# embeddings layer underneath it (already covered by test_entity_linker.py) ----

def test_team_falls_back_to_semantic_when_fuzzy_finds_nothing(gazetteer_db, monkeypatch):
    monkeypatch.setattr(entity_extractor.entity_linker.settings, "entity_linking_enabled", True)
    monkeypatch.setattr(
        entity_extractor.entity_linker,
        "semantic_candidates",
        lambda text, entity_type, db, **kw: [{"value": "Blue Area", "score": 0.81}] if entity_type == "team" else [],
    )
    entities = extract_entities("who's leading in the CBD zone", gazetteer_db)
    assert entities["team"] == "Blue Area"
    assert entities["team_matches"] == [{"value": "Blue Area", "score": 0.81}]


def test_semantic_fallback_not_used_when_fuzzy_already_matched(gazetteer_db, monkeypatch):
    monkeypatch.setattr(entity_extractor.entity_linker.settings, "entity_linking_enabled", True)
    calls = []
    monkeypatch.setattr(
        entity_extractor.entity_linker,
        "semantic_candidates",
        lambda text, entity_type, db, **kw: calls.append(entity_type) or [],
    )
    entities = extract_entities("show graana advisors", gazetteer_db)
    assert entities["company"] == "Graana"
    assert "company" not in calls   # exact substring already matched — semantic never consulted


def test_advisor_falls_back_to_semantic_when_fuzzy_finds_nothing(gazetteer_db, monkeypatch):
    monkeypatch.setattr(entity_extractor.entity_linker.settings, "entity_linking_enabled", True)
    monkeypatch.setattr(
        entity_extractor.entity_linker,
        "semantic_candidates",
        lambda text, entity_type, db, **kw: [{"value": "Waqar Haider", "score": 0.78}] if entity_type == "advisor" else [],
    )
    entities = extract_entities("how's that guy who closed the Graana deal doing", gazetteer_db)
    assert entities["advisor_name"] == "Waqar Haider"
    assert entities["advisor_matches"] == [{"value": "Waqar Haider", "score": 0.78}]


def test_new_entity_types_have_gazetteer_getters(gazetteer_db):
    gazetteer_db.add(Advisor(
        wid=4, name="New Guy", team="Blue Area", company="Graana",
        office="F-10 Office", portfolio_lead="Portfolio Lead A", management_lead="Management Lead A",
    ))
    gazetteer_db.commit()
    entity_extractor._cache["loaded_at"] = 0

    assert "F-10 Office" in entity_extractor.get_known_offices(gazetteer_db)
    assert "Portfolio Lead A" in entity_extractor.get_known_portfolio_leads(gazetteer_db)
    assert "Management Lead A" in entity_extractor.get_known_management_leads(gazetteer_db)


# ---- Part 12: semantic retrieval fallback for comparators & attendance status ----

def test_comparator_semantic_fallback_disabled_by_default(gazetteer_db):
    # conftest's autouse fixture forces entity_linking_enabled=False — an
    # unusual comparator phrasing must not produce a threshold at all
    entities = extract_entities("advisors north of 80", gazetteer_db)
    assert entities["thresholds"] == []


def test_comparator_semantic_fallback_classifies_unusual_phrasing(gazetteer_db, monkeypatch):
    monkeypatch.setattr(entity_extractor.entity_linker.settings, "entity_linking_enabled", True)
    # stub out the OTHER semantic path (gazetteer fields) so this test
    # can't fall through to a real, unmocked embedding call
    monkeypatch.setattr(entity_extractor.entity_linker, "semantic_candidates", lambda *a, **kw: [])
    monkeypatch.setattr(
        entity_extractor.entity_linker, "semantic_classify",
        lambda text, exemplar_type, **kw: [{"value": ">", "score": 0.9}] if exemplar_type == "comparator" else [],
    )
    entities = extract_entities("advisors north of 80", gazetteer_db)
    assert entities["thresholds"] == [{"operator": ">", "value": 80.0}]


def test_comparator_semantic_fallback_never_overrides_closed_vocabulary(gazetteer_db, monkeypatch):
    monkeypatch.setattr(entity_extractor.entity_linker.settings, "entity_linking_enabled", True)
    monkeypatch.setattr(entity_extractor.entity_linker, "semantic_candidates", lambda *a, **kw: [])
    calls = []
    monkeypatch.setattr(
        entity_extractor.entity_linker, "semantic_classify",
        lambda text, exemplar_type, **kw: calls.append(text) or [{"value": "<", "score": 0.99}],
    )
    # "more than 80" already matches the closed regex vocabulary (">") —
    # the semantic step (which would say "<" here) must never be consulted
    # for this number
    entities = extract_entities("advisors with more than 80 connects", gazetteer_db)
    assert entities["thresholds"] == [{"operator": ">", "value": 80.0}]
    assert calls == []


def test_attendance_status_semantic_fallback_disabled_by_default(gazetteer_db):
    entities = extract_entities("who walked in behind schedule today", gazetteer_db)
    assert "attendance_status" not in entities


def test_attendance_status_semantic_fallback_classifies_paraphrase(gazetteer_db, monkeypatch):
    monkeypatch.setattr(entity_extractor.entity_linker.settings, "entity_linking_enabled", True)
    monkeypatch.setattr(entity_extractor.entity_linker, "semantic_candidates", lambda *a, **kw: [])
    monkeypatch.setattr(
        entity_extractor.entity_linker, "semantic_classify",
        lambda text, exemplar_type, **kw: [{"value": "Late", "score": 0.8}] if exemplar_type == "attendance_status" else [],
    )
    entities = extract_entities("who walked in behind schedule today", gazetteer_db)
    assert entities["attendance_status"] == "Late"


def test_attendance_status_semantic_fallback_never_overrides_keyword_match(gazetteer_db, monkeypatch):
    monkeypatch.setattr(entity_extractor.entity_linker.settings, "entity_linking_enabled", True)
    monkeypatch.setattr(entity_extractor.entity_linker, "semantic_candidates", lambda *a, **kw: [])
    calls = []
    monkeypatch.setattr(
        entity_extractor.entity_linker, "semantic_classify",
        lambda text, exemplar_type, **kw: calls.append(text) or [{"value": "Absent", "score": 0.99}],
    )
    # "absent" already matches the exact keyword dict — semantic must
    # never even be consulted
    entities = extract_entities("who was absent today", gazetteer_db)
    assert entities["attendance_status"] == "Absent"
    assert calls == []


def test_attendance_status_semantic_fallback_skipped_without_a_hint(gazetteer_db, monkeypatch):
    monkeypatch.setattr(entity_extractor.entity_linker.settings, "entity_linking_enabled", True)
    monkeypatch.setattr(entity_extractor.entity_linker, "semantic_candidates", lambda *a, **kw: [])
    calls = []
    monkeypatch.setattr(
        entity_extractor.entity_linker, "semantic_classify",
        lambda text, exemplar_type, **kw: calls.append(exemplar_type) or [],
    )
    # "top 5 advisors by revenue" still has a bare number ("5"), which DOES
    # reach the comparator semantic path (a different gate, not under test
    # here) — the point of this test is narrower: no attendance-language
    # hint means the attendance_status path specifically must never fire
    extract_entities("top 5 advisors by revenue", gazetteer_db)
    assert "attendance_status" not in calls
