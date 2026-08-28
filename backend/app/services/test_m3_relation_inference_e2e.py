"""Inference for the newly cached relations (M3).

M3's claim is that business_center / unit_head / zonal_head become
inferable as a CONSEQUENCE of the registry — the only code written for
them was `cached=True` on three declarations and three fields on
AdvisorIdentity. Nothing in relation_resolver, reference_parser or
entity_extractor names any of them.

The routing-interaction tests at the bottom are the ones that earn their
keep. Enabling unit_head inference means "who is X's unit head" now
matches BOTH a reverse-role question (M2) and a group reference (M1) —
the hazard flagged when M2 was delivered. Reverse must still win.
"""

import pytest

from app.core.config import settings
from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def db(db_session, monkeypatch):
    people = [
        (1, "Waqar Haider", "Blue Area", "Gulberg BC", "Kaleem Ullah", "Adeel Dogar"),
        (2, "Sana Tariq", "Blue Area", "Gulberg BC", "Kaleem Ullah", "Adeel Dogar"),
        (3, "Imran Butt", "Downtown", "Saddar BC", "Nadia Rehman", "Faisal Iqbal"),
    ]
    for wid, name, team, office, bm, zm in people:
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                               office=office, bm=bm, zm=zm, rm=bm,
                               portfolio_lead=zm, management_lead=f"BCM {wid}"))
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


@pytest.fixture()
def inference_on(monkeypatch):
    monkeypatch.setattr(settings, "relation_inference_enabled", True)
    monkeypatch.setattr(
        settings, "relation_inference_levels",
        "team,company,office,unit_head,zonal_head,bcm",
    )


# ---------------------------------------------------------------------
# The projection now carries them
# ---------------------------------------------------------------------

def test_identity_carries_the_three_new_relations(db):
    advisor_resolver.refresh_cache(db, force=True)
    identity = next(i for i in advisor_resolver._cache["identities"] if i.wid == 1)
    assert identity.office == "Gulberg BC"
    assert identity.unit_head == "Kaleem Ullah"
    assert identity.zonal_head == "Adeel Dogar"


def test_fields_survive_rescoring(db):
    """The silent-drop hazard: a fuzzy match rebuilds the identity."""
    advisor_resolver.refresh_cache(db, force=True)
    identity = next(i for i in advisor_resolver._cache["identities"] if i.wid == 1)
    rescored = advisor_resolver._with_score(identity, 0.91)
    assert rescored.unit_head == "Kaleem Ullah"
    assert rescored.zonal_head == "Adeel Dogar"
    assert rescored.office == "Gulberg BC"
    assert rescored.score == 0.91


# ---------------------------------------------------------------------
# Acceptance: the three inferences
# ---------------------------------------------------------------------

@pytest.mark.parametrize("phrase,level,value", [
    ("Waqar Haider's business center", "office", "Gulberg BC"),
    ("Waqar Haider's unit head", "unit_head", "Kaleem Ullah"),
    ("Waqar Haider's zonal head", "zonal_head", "Adeel Dogar"),
])
def test_inference_resolves_the_new_relations(db, inference_on, phrase, level, value):
    entities = entity_extractor.extract_entities(phrase, db)
    assert entities[level] == value
    assert entity_extractor.provenance_of(entities, level) == "inferred:advisor:1"


def test_inferred_business_center_scopes_a_leaderboard(db, inference_on):
    """Note the phrasing: "X's centre", not "X's business center".

    The two are not interchangeable, and the split is coherent rather
    than accidental — a ROLE phrasing asks who/what holds the role, a
    bare GROUP word refers to the group. See
    test_role_phrasings_are_claimed_by_reverse_lookup below, which pins
    the boundary from the other side."""
    referred = handle_chat_message(db, "Top advisors in Waqar Haider's centre", session_id=None)
    named = handle_chat_message(db, "Top advisors in Gulberg BC", session_id=None)
    assert referred["reply"] == named["reply"]
    assert {row["name"] for row in referred["data"]} == {"Waqar Haider", "Sana Tariq"}


@pytest.mark.parametrize("query", [
    "Top advisors in Waqar Haider's business center",
    "Top advisors in Waqar Haider's branch",
])
def test_role_phrasings_are_claimed_by_reverse_lookup(db, inference_on, query):
    """KNOWN LIMITATION, characterised rather than left to be discovered.

    "business center" and "branch" are reverse-role aliases (M2), so
    `_score_reverse_hierarchy` scores 0.98, ties with leaderboard, and
    wins on scorer declaration order — the query answers "X's Business
    Center is Gulberg BC" instead of ranking that centre's advisors.

    This is NOT caused by M3: the replies are byte-identical with
    inference disabled (test_reverse_questions_are_identical_with_
    inference_off covers the same property). Resolving it means changing
    planner scoring, which M3 must not touch. Recorded so the boundary is
    visible and so a later milestone that fixes it fails here and revisits
    this deliberately."""
    r = handle_chat_message(db, query, session_id=None)
    assert r["type"] == "manager"


def test_inferred_unit_head_scopes_a_leaderboard(db, inference_on):
    """UPDATED BY M5: "X's unit" now means the UNIT level, so the unit-
    head reference uses "division" — a word that did not move in the
    alias split. The M3 property (an uncached-then-cached relation scopes
    a ranking) is unchanged."""
    referred = handle_chat_message(db, "Top advisors in Waqar Haider's division", session_id=None)
    named = handle_chat_message(db, "Top advisors under Kaleem Ullah", session_id=None)
    assert {row["name"] for row in referred["data"]} == {"Waqar Haider", "Sana Tariq"}
    assert referred["reply"] == named["reply"]


def test_ambiguous_source_still_never_infers_the_new_relations(db, inference_on):
    db.add(Advisor(wid=4, name="Yasir Ali", team="Blue Area", company="Graana",
                   office="Gulberg BC", bm="Kaleem Ullah", rm="Kaleem Ullah", zm="Adeel Dogar", portfolio_lead="Adeel Dogar"))
    db.add(Advisor(wid=5, name="Yasir Ali", team="Downtown", company="Graana",
                   office="Saddar BC", bm="Nadia Rehman", rm="Nadia Rehman", zm="Faisal Iqbal", portfolio_lead="Faisal Iqbal"))
    db.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()

    entities = entity_extractor.extract_entities("Yasir Ali's unit head", db)
    assert entities.get("unit_head") is None
    assert entities.get("advisor_ambiguous") is True


def test_advisor_with_no_value_infers_nothing(db, inference_on):
    db.add(Advisor(wid=6, name="Nadia Sheikh", team="Blue Area", company="Graana", zm=None))
    db.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()

    entities = entity_extractor.extract_entities("Nadia Sheikh's zonal head", db)
    assert entities.get("zonal_head") is None


# ---------------------------------------------------------------------
# Routing interaction — the hazard flagged at M2 delivery
# ---------------------------------------------------------------------

@pytest.mark.parametrize("query,expected_reply", [
    ("Who is Waqar Haider's unit head", "Waqar Haider's Unit Head is Kaleem Ullah."),
    ("Who is Waqar Haider\'s BM", "Waqar Haider\'s BM is Kaleem Ullah."),
    ("Who is Waqar Haider's zonal head", "Waqar Haider's Zonal Head is Adeel Dogar."),
    ("Who is Waqar Haider's business centre", "Waqar Haider's Office is Gulberg BC."),
    ("Who does Waqar Haider report to", "Waqar Haider's Unit Head is Kaleem Ullah."),
])
def test_reverse_questions_still_win_over_group_inference(db, inference_on, query, expected_reply):
    """"who is X's unit head" now matches a reverse-role question AND a
    group reference. Reverse must win — the user asked WHO the unit head
    is, not for a ranking of that unit's advisors."""
    r = handle_chat_message(db, query, session_id=None)
    assert r["type"] == "manager"
    assert r["reply"] == expected_reply


def test_reverse_questions_are_identical_with_inference_off(db, monkeypatch):
    """The same replies with the flag off, proving inference did not
    change reverse routing in either direction."""
    monkeypatch.setattr(settings, "relation_inference_enabled", False)
    off = handle_chat_message(db, "Who is Waqar Haider's unit head", session_id=None)

    monkeypatch.setattr(settings, "relation_inference_enabled", True)
    monkeypatch.setattr(settings, "relation_inference_levels",
                        "team,company,office,unit_head,zonal_head,bcm")
    on = handle_chat_message(db, "Who is Waqar Haider's unit head", session_id=None)

    assert off["type"] == on["type"] == "manager"
    assert off["reply"] == on["reply"]


# ---------------------------------------------------------------------
# M1/M2 preservation
# ---------------------------------------------------------------------

def test_team_and_company_inference_unchanged(db, inference_on):
    referred = handle_chat_message(db, "How is Waqar Haider's team doing", session_id=None)
    named = handle_chat_message(db, "How is Blue Area doing", session_id=None)
    assert referred["reply"] == named["reply"]


def test_person_lookup_unchanged(db, inference_on):
    r = handle_chat_message(db, "tell me about Waqar Haider", session_id=None)
    assert r["type"] == "advisor"


def test_uncached_relations_still_not_inferred(db, monkeypatch):
    monkeypatch.setattr(settings, "relation_inference_enabled", True)
    monkeypatch.setattr(settings, "relation_inference_levels", "portfolio_lead,management_lead")
    entities = entity_extractor.extract_entities("Waqar Haider's portfolio lead", db)
    assert entities.get("portfolio_lead") is None


def test_no_extra_database_reads_for_the_new_relations(db, monkeypatch):
    """Approach A's whole point: inference stays a getattr."""
    from app.core import tracing

    def count(enabled):
        monkeypatch.setattr(settings, "relation_inference_enabled", enabled)
        entity_extractor._cache["loaded_at"] = 0
        advisor_resolver._reset_for_tests()
        with tracing.traced("q") as trace:
            entity_extractor.extract_entities("Waqar Haider's unit head", db)
            return len(trace.sql)

    monkeypatch.setattr(settings, "relation_inference_levels",
                        "team,company,office,unit_head,zonal_head,bcm")
    assert count(True) == count(False)
