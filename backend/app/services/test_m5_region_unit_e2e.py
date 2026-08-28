"""Region as a first-class level, end to end (M5, updated by Phase 3).

PHASE 3 removed the `unit` LEVEL entirely — Advisor.unit has zero
production rows and the ETL never writes it — so the unit cases here are
gone. Region survives, and the alias split M5 introduced still holds:
"region" names the level, not the zonal head.

PHASE 3 also rebound the manager levels, so "unit head" now reads
Advisor.rm and "zonal head" reads Advisor.portfolio_lead. The fixture
states both, and "BM" answers from Advisor.bm under its own label.

Two things are being proved. First, that queries scoped by region and
unit now work at all — before M5 the columns held the answer and nothing
could reach them. Second, and more delicately, that the alias split took
the two ambiguous bare words WITHOUT taking any of the explicit manager
vocabulary with them: "his unit" changed meaning, "unit head" did not.

Region/Unit behaviour is asserted by EQUIVALENCE with team and company
wherever possible — a new level should behave like the levels that
already worked, not like a special case.
"""

import pytest

from app.core.config import settings
from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def db(db_session, monkeypatch):
    people = [
        (1, "Waqar Haider", "Blue Area", "Graana", "North", "Unit 1", "Kaleem Ullah", "Adeel Dogar", "Gulberg BC"),
        (2, "Sana Tariq", "Blue Area", "Graana", "North", "Unit 1", "Kaleem Ullah", "Adeel Dogar", "Gulberg BC"),
        (3, "Imran Butt", "Downtown", "Agency21", "South", "Unit 2", "Nadia Rehman", "Faisal Iqbal", "Saddar BC"),
    ]
    for wid, name, team, company, region, unit, bm, zm, office in people:
        db_session.add(Advisor(wid=wid, name=name, team=team, company=company, region=region,
                               unit=unit, bm=bm, zm=zm, office=office,
                               # PHASE 3: unit_head reads rm and zonal_head reads
                               # portfolio_lead, so the fixture states them.
                               rm=bm, portfolio_lead=zm, management_lead=f"BCM {wid}"))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=40, mtd_followup_connect=2))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000, cleared=500))
    db_session.commit()

    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "llm_first")
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    import app.llm.llm_client as llm_client
    # Patch the PROVIDER BOUNDARY, not a vendor SDK's call shape. The
    # previous form named llm_client._client.chat.completions.create —
    # four OpenAI internals — and every one of them broke when the client
    # became Ollama, erroring 146 tests that were not about the provider.
    monkeypatch.setattr(llm_client, "_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("off")))
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


# ---------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------

def test_advisors_in_a_region(db):
    r = handle_chat_message(db, "Advisors in North Region", session_id=None)
    assert r["type"] == "roster"
    assert {a["name"] for a in r["data"]["advisors"]} == {"Waqar Haider", "Sana Tariq"}


def test_top_advisors_in_a_region(db):
    r = handle_chat_message(db, "Top advisors in North Region", session_id=None)
    assert r["type"] == "leaderboard"
    assert {row["name"] for row in r["data"]} == {"Waqar Haider", "Sana Tariq"}
    assert "region = North" in r["reply"]




def test_region_scoping_matches_team_scoping_in_shape(db):
    """A new level should behave like an established one."""
    region = handle_chat_message(db, "Top advisors in North Region", session_id=None)
    team = handle_chat_message(db, "Top advisors in Blue Area", session_id=None)
    assert region["type"] == team["type"]
    assert len(region["data"]) == len(team["data"])



@pytest.mark.parametrize("query,expected", [
    ("Who is Waqar Haider's zonal head?", "Waqar Haider's Zonal Head is Adeel Dogar."),
    ("Who is Waqar Haider's zone head?", "Waqar Haider's Zonal Head is Adeel Dogar."),
    ("Who is Waqar Haider's ZM?", "Waqar Haider's ZM is Adeel Dogar."),
    ("Who is Waqar Haider's unit head?", "Waqar Haider's Unit Head is Kaleem Ullah."),
    ("Who is Waqar Haider's BM?", "Waqar Haider's BM is Kaleem Ullah."),
    ("Who is Waqar Haider's division head?", "Waqar Haider's Unit Head is Kaleem Ullah."),
    ("Who does Waqar Haider report to?", "Waqar Haider's Unit Head is Kaleem Ullah."),
])
def test_reverse_lookup_is_byte_identical(db, query, expected):
    r = handle_chat_message(db, query, session_id=None)
    assert r["type"] == "manager"
    assert r["reply"] == expected


def test_a_unit_head_ranking_still_works(db):
    """"under Kaleem Ullah" scopes by unit_head, unaffected by the split."""
    r = handle_chat_message(db, "Top advisors under Kaleem Ullah", session_id=None)
    assert {row["name"] for row in r["data"]} == {"Waqar Haider", "Sana Tariq"}


# ---------------------------------------------------------------------
# Region vs Zonal Head, Unit vs Unit Head
# ---------------------------------------------------------------------

def test_region_and_zonal_head_are_different_questions(db):
    region = handle_chat_message(db, "Advisors in North Region", session_id=None)
    zonal = handle_chat_message(db, "Who is Waqar Haider's zonal head?", session_id=None)
    assert region["type"] == "roster"
    assert zonal["type"] == "manager"
    assert "Adeel Dogar" not in region["reply"]




def test_team_company_and_business_center_are_unchanged(db):
    assert handle_chat_message(db, "How is Blue Area doing", session_id=None)["type"] == "team"
    assert handle_chat_message(db, "How is Graana doing", session_id=None)["type"] == "company"
    bc = handle_chat_message(db, "Top advisors in Gulberg BC", session_id=None)
    assert {row["name"] for row in bc["data"]} == {"Waqar Haider", "Sana Tariq"}


def test_relationship_inference_is_unchanged(db, monkeypatch):
    monkeypatch.setattr(settings, "relation_inference_enabled", True)
    monkeypatch.setattr(settings, "relation_inference_levels",
                        "team,company,office,unit_head,zonal_head,bcm")
    referred = handle_chat_message(db, "How is Waqar Haider's team doing", session_id="a")
    named = handle_chat_message(db, "How is Blue Area doing", session_id="b")
    assert referred["reply"] == named["reply"]


def test_person_lookup_unchanged(db):
    r = handle_chat_message(db, "tell me about Waqar Haider", session_id=None)
    assert r["type"] == "advisor"


# ---------------------------------------------------------------------
# Invalid and mixed
# ---------------------------------------------------------------------

def test_an_unknown_region_does_not_invent_one(db):
    r = handle_chat_message(db, "Advisors in Mars Region", session_id=None)
    assert r["type"] in ("clarification", "not_found", "unknown")
    assert "Waqar Haider" not in r["reply"]



def test_a_mixed_hierarchy_query_prefers_the_narrower_level(db):
    """Naming both a region and a team means the team — the narrower
    answer is the more informative one, per GROUP_LEVEL_ORDER."""
    r = handle_chat_message(db, "Top advisors in Blue Area in North Region", session_id=None)
    assert {row["name"] for row in r["data"]} == {"Waqar Haider", "Sana Tariq"}


