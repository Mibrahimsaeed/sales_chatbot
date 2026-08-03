"""Relationship inference end to end (M1).

The milestone's promise: a group referred to THROUGH a person behaves
exactly as it would if the user had named it. The tests are therefore
written as EQUIVALENCES against the literal-mention path wherever
possible — asserting a hand-written expected reply would only pin
whatever this milestone happened to produce, while equivalence pins the
property that actually matters.

The two guard tests (flag off, and "tell me about <person>" with the flag
ON) defend risk R1: inference must not disturb the product's most common
query shape.
"""

import pytest

from app.core.config import settings
from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid, name, team, company in (
        (1, "Waqar Haider", "Blue Area", "Graana"),
        (2, "Sana Tariq", "Blue Area", "Graana"),
        (3, "Imran Butt", "Downtown", "Agency21"),
    ):
        db_session.add(Advisor(wid=wid, name=name, team=team, company=company,
                               bm="Kaleem Ullah", zm="Adeel Dogar", office="Gulberg BC",
                               rm="Tariq Mehmood", portfolio_lead="Sana Malik",
                               management_lead="Imran Shah"))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=40, mtd_followup_connect=2))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000, cleared=500))
    # Two people share a name — an ambiguous source must never infer.
    db_session.add(Advisor(wid=4, name="Yasir Ali", team="Blue Area", company="Graana"))
    db_session.add(Advisor(wid=5, name="Yasir Ali", team="Downtown", company="Agency21"))
    db_session.commit()

    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "llm_first")
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    import app.llm.llm_client as llm_client
    monkeypatch.setattr(llm_client._client.chat.completions, "create",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("off")))
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


@pytest.fixture()
def inference_on(monkeypatch):
    monkeypatch.setattr(settings, "relation_inference_enabled", True)
    monkeypatch.setattr(settings, "relation_inference_levels", "team,company")


# ---------------------------------------------------------------------
# Risk R1 — the change must not disturb what already works
# ---------------------------------------------------------------------

def test_flag_off_leaves_everything_exactly_as_it_was(db):
    """Default configuration: the milestone is dark."""
    r = handle_chat_message(db, "Tell me about Waqar Haider's team", session_id=None)
    assert r["type"] == "advisor"


def test_plain_person_lookup_is_unaffected_with_inference_on(db, inference_on):
    """The highest-traffic query shape. No possessive, no reference, so
    nothing is inferred and advisor_profile still wins."""
    r = handle_chat_message(db, "tell me about Waqar Haider", session_id=None)
    assert r["type"] == "advisor"
    assert "Waqar Haider has 42 MTD connects" in r["reply"]


def test_a_bare_level_word_does_not_trigger_inference(db, inference_on):
    """"team performance" names a topic, not a relationship."""
    entities = entity_extractor.extract_entities("Waqar Haider team performance", db)
    assert entities.get("team") is None


# ---------------------------------------------------------------------
# The milestone's actual promise — equivalence with the named path
# ---------------------------------------------------------------------

def test_possessive_team_behaves_like_naming_the_team(db, inference_on):
    referred = handle_chat_message(db, "How is Waqar Haider's team doing", session_id=None)
    named = handle_chat_message(db, "How is Blue Area doing", session_id=None)
    assert referred["type"] == named["type"]
    assert referred["reply"] == named["reply"]


def test_leaderboard_scope_is_applied_from_the_referred_team(db, inference_on):
    """The acceptance criterion: the ranking is confined to the team, and
    the reply SAYS so — previously it silently ranked everyone."""
    r = handle_chat_message(db, "Top 5 advisors in Waqar Haider's team", session_id=None)
    assert r["type"] == "leaderboard"
    assert "filtered by team = Blue Area" in r["reply"]
    assert {row["name"] for row in r["data"]} == {"Waqar Haider", "Sana Tariq"}


def test_leaderboard_scope_matches_the_named_equivalent(db, inference_on):
    referred = handle_chat_message(db, "Top 5 advisors in Waqar Haider's team", session_id=None)
    named = handle_chat_message(db, "Top 5 advisors in Blue Area", session_id=None)
    assert referred["reply"] == named["reply"]


def test_company_reference_scopes_a_leaderboard(db, inference_on):
    """P9. The company is inferred and applied as a filter, so the
    ranking is confined to it — Agency21's advisor is excluded."""
    referred = handle_chat_message(db, "Top advisors in Waqar Haider's company", session_id=None)
    named = handle_chat_message(db, "Top advisors in Graana", session_id=None)

    assert "filtered by company = Graana" in referred["reply"]
    assert {row["name"] for row in referred["data"]} == {"Waqar Haider", "Sana Tariq"}
    assert referred["reply"] == named["reply"]


def test_company_as_a_SUBJECT_is_not_routed_yet(db, inference_on):
    """KNOWN LIMITATION, characterised deliberately rather than left to
    be discovered.

    Inference does its job here — entities["company"] is populated — but
    the PLANNER still answers with the person's profile, because
    `RELATIONAL_RE` recognises "'s team" as a relational phrase and has
    no equivalent for "'s company". So `_score_hierarchy` fires for team
    and nothing group-ward fires for company, leaving advisor_profile
    (0.50) to beat entity_summary (0.40).

    Closing this means editing the planner's trigger vocabulary, which
    M1 explicitly does not own. This test documents the boundary; when a
    later milestone widens that vocabulary, it will fail and force this
    to be revisited.
    """
    entities = entity_extractor.extract_entities("How is Waqar Haider's company doing", db)
    assert entities["company"] == "Graana"          # inference worked
    assert entity_extractor.provenance_of(entities, "company") == "inferred:advisor:1"

    r = handle_chat_message(db, "How is Waqar Haider's company doing", session_id=None)
    assert r["type"] == "advisor"                   # ...but routing did not follow


# ---------------------------------------------------------------------
# Risk R2 — an ambiguous source must never infer
# ---------------------------------------------------------------------

def test_ambiguous_source_does_not_infer_a_team(db, inference_on):
    """Two people named Yasir Ali, on different teams. Inferring from
    whichever sorted first would answer confidently about the wrong
    person's team."""
    entities = entity_extractor.extract_entities("Yasir Ali's team", db)
    assert entities.get("team") is None
    assert entities.get("advisor_ambiguous") is True


def test_ambiguous_source_still_asks_which_person(db, inference_on):
    r = handle_chat_message(db, "How is Yasir Ali's team doing", session_id=None)
    assert r["type"] == "clarification"


# ---------------------------------------------------------------------
# Precedence, provenance, and absent data
# ---------------------------------------------------------------------

def test_an_explicitly_named_entity_is_never_overwritten(db, inference_on):
    """The user said Downtown; we must not replace it with Waqar's team."""
    entities = entity_extractor.extract_entities("Waqar Haider's team vs Downtown", db)
    assert entities["team"] == "Downtown"
    assert entity_extractor.provenance_of(entities, "team") == entity_extractor.PROVENANCE_EXPLICIT


def test_inferred_entities_carry_source_provenance(db, inference_on):
    entities = entity_extractor.extract_entities("Waqar Haider's team", db)
    assert entities["team"] == "Blue Area"
    assert entity_extractor.provenance_of(entities, "team") == "inferred:advisor:1"
    assert entity_extractor.provenance_of(entities, "teams") == "inferred:advisor:1"
    assert entity_extractor.provenance_of(entities, "advisor_name") == entity_extractor.PROVENANCE_EXPLICIT


def test_inferred_entity_uses_the_same_key_shape_as_a_gazetteer_hit(db, inference_on):
    inferred = entity_extractor.extract_entities("Waqar Haider's team", db)
    named = entity_extractor.extract_entities("Blue Area", db)
    assert inferred["team"] == named["team"]
    assert inferred["teams"] == named["teams"]
    assert set(inferred["team_matches"][0]) == set(named["team_matches"][0])


def test_a_level_outside_the_enabled_set_is_not_inferred(db, monkeypatch):
    """Per-relation granularity: enabling team must not enable company."""
    monkeypatch.setattr(settings, "relation_inference_enabled", True)
    monkeypatch.setattr(settings, "relation_inference_levels", "team")
    entities = entity_extractor.extract_entities("Waqar Haider's company", db)
    assert entities.get("company") is None


def test_uncached_relations_are_not_resolved(db, monkeypatch):
    """The PROPERTY: a relation the identity cache does not carry is
    never resolved, so inference cannot start issuing per-reference
    database reads by accident.

    UPDATED BY M3, which moved unit_head into the cache — this test now
    uses portfolio_lead, which remains uncached. The property is
    unchanged; only the example of an uncached relation moved."""
    monkeypatch.setattr(settings, "relation_inference_enabled", True)
    monkeypatch.setattr(settings, "relation_inference_levels", "team,company,portfolio_lead")
    entities = entity_extractor.extract_entities("Waqar Haider's portfolio lead", db)
    assert entities.get("portfolio_lead") is None


def test_reverse_manager_questions_are_untouched(db, inference_on):
    """The one advisor->X capability that already works."""
    r = handle_chat_message(db, "Who is Waqar Haider's BM", session_id=None)
    assert r["type"] == "manager"
    assert "Kaleem Ullah" in r["reply"]


def test_inference_adds_no_database_reads(db, monkeypatch):
    """The acceptance criterion for M1's cost: team and company are
    already on AdvisorIdentity, so inferring them is a getattr. If a
    future change starts issuing a query per reference, this fails."""
    from app.core import tracing

    def count_statements(enabled: bool) -> int:
        monkeypatch.setattr(settings, "relation_inference_enabled", enabled)
        entity_extractor._cache["loaded_at"] = 0
        advisor_resolver._reset_for_tests()
        with tracing.traced("Waqar Haider's team") as trace:
            entity_extractor.extract_entities("Waqar Haider's team", db)
            return len(trace.sql)

    monkeypatch.setattr(settings, "relation_inference_levels", "team,company")
    assert count_statements(True) == count_statements(False)


def test_advisor_with_no_team_on_file_infers_nothing(db, inference_on):
    db.add(Advisor(wid=9, name="Nadia Rehman", team=None, company="Graana"))
    db.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()

    entities = entity_extractor.extract_entities("Nadia Rehman's team", db)
    assert entities.get("team") is None
