"""End-to-end coverage for the hierarchy rework's new capabilities through
the real chat pipeline (nlu_pipeline.resolve -> chat_service dispatch),
mirroring test_chat_service_multi.py's rules_first + no-LLM setup so these
stay deterministic and fully offline."""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def hierarchy_db(db_session, monkeypatch):
    db_session.add(Advisor(wid=1, name="Advisor One", team="Blue Area", company="Graana", bm="Zeeshan Tariq", rm="Zeeshan Tariq",
                           portfolio_lead="Zonal North"))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=10, mtd_followup_connect=0))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, target=1000, cleared=500))

    db_session.add(Advisor(wid=2, name="Advisor Two", team="Downtown", company="IMARAT", bm="Zeeshan Tariq", rm="Zeeshan Tariq",
                           portfolio_lead="Zonal South"))
    db_session.add(SalesFunnel(wid=2, mtd_new_connect=20, mtd_followup_connect=0))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, target=2000, cleared=1000))

    db_session.add(Advisor(wid=3, name="Advisor Three", team="Gamma", company="Graana", bm="Someone Else", rm="Someone Else",
                           portfolio_lead="Zonal Other"))
    db_session.add(SalesFunnel(wid=3, mtd_new_connect=900, mtd_followup_connect=0))
    db_session.commit()

    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def test_bare_unit_head_mention_returns_nested_breakdown(hierarchy_db):
    response = handle_chat_message(hierarchy_db, "tell me about unit head Zeeshan Tariq", session_id="h-1")

    assert response["type"] == "breakdown"
    assert response["data"]["advisors"] == 2
    team_names = {t["team"] for t in response["data"]["teams"]}
    # PHASE 3: a Unit Head breakdown nests by Zonal Head, its child
    # in the verified chain, not by team.
    assert team_names == {"Zonal North", "Zonal South"}
    assert "Zonal North" in response["reply"]
    assert "Advisor One" in response["reply"]


def test_top_unit_heads_leaderboard_ranks_by_rollup(hierarchy_db):
    response = handle_chat_message(hierarchy_db, "top 2 unit heads by connects", session_id="h-3")

    assert response["type"] == "leaderboard"
    names = [row["name"] for row in response["data"]]
    assert "Zeeshan Tariq" in names
    assert "Someone Else" in names
    by_name = {row["name"]: row["value"] for row in response["data"]}
    assert by_name["Zeeshan Tariq"] == 30   # 10 + 20, rolled up from two teams
    assert by_name["Someone Else"] == 900


def test_flat_phrase_returns_ungrouped_advisor_list(hierarchy_db):
    response = handle_chat_message(
        hierarchy_db, "give me a flat list of unit head Zeeshan Tariq's advisors", session_id="h-4"
    )

    assert response["type"] == "breakdown"
    assert "teams" not in response["data"]
    assert {a["name"] for a in response["data"]["advisor_list"]} == {"Advisor One", "Advisor Two"}


# ---- Phase 2: intent="breakdown" reachable from the LLM/IR pipeline too,
# not just the rule-based bare-mention path — dispatched directly here
# (rather than mocking a full LLM round trip) since chat_service._dispatch
# routes purely on resolution.kind/ir.intent, independent of which pipeline
# produced the IR. ----

def test_ir_breakdown_intent_dispatches_to_nested_breakdown(hierarchy_db):
    from app.llm.nlu_pipeline import Resolution
    from app.llm.query_ir import QueryIR, Subject
    from app.services.chat_service import _dispatch

    ir = QueryIR(
        intent="breakdown",
        subject_level="unit_head",
        subjects=[Subject(type="unit_head", value="Zeeshan Tariq", resolved_id="Zeeshan Tariq", match_confidence=1.0)],
    )
    response = _dispatch(hierarchy_db, Resolution(kind="ir", ir=ir, entities={}))

    assert response["type"] == "breakdown"
    assert response["data"]["advisors"] == 2
    assert {t["team"] for t in response["data"]["teams"]} == {"Zonal North", "Zonal South"}


def test_ir_breakdown_intent_respects_flat_field(hierarchy_db):
    from app.llm.nlu_pipeline import Resolution
    from app.llm.query_ir import QueryIR, Subject
    from app.services.chat_service import _dispatch

    ir = QueryIR(
        intent="breakdown",
        subject_level="unit_head",
        subjects=[Subject(type="unit_head", value="Zeeshan Tariq", resolved_id="Zeeshan Tariq", match_confidence=1.0)],
        flat=True,
    )
    response = _dispatch(hierarchy_db, Resolution(kind="ir", ir=ir, entities={}))

    assert response["type"] == "breakdown"
    assert "teams" not in response["data"]
    assert "advisor_list" in response["data"]


def test_ir_breakdown_intent_not_found_is_graceful(hierarchy_db):
    from app.llm.nlu_pipeline import Resolution
    from app.llm.query_ir import QueryIR, Subject
    from app.services.chat_service import _dispatch

    ir = QueryIR(
        intent="breakdown",
        subject_level="unit_head",
        subjects=[Subject(type="unit_head", value="Totally Unknown", resolved_id="Totally Unknown", match_confidence=1.0)],
    )
    response = _dispatch(hierarchy_db, Resolution(kind="ir", ir=ir, entities={}))

    assert response["type"] == "not_found"


# ---- Phase 2: cross-level ambiguity, end to end through the real pipeline ----

def test_a_manager_who_is_also_an_advisor_gets_their_own_profile(hierarchy_db):
    """"Zeeshan Tariq" is grounded as a unit head AND, once added here, as
    a real advisor name.

    This asserted a clarification — "the Unit Head or the Advisor?" — and
    that expectation is what changed. Those are not two entities; they are
    one person and their job. A question that names neither a measure nor
    their team is about the person, and answering it with their own
    profile is what a single-role advisor already gets for these words.

    The clarification itself is not gone: a name that is also a TEAM or a
    COMPANY still asks, because those genuinely are different things
    sharing a spelling. That case is pinned in the golden corpus
    ("how is Nashit Raza doing") and in test_person_profile_ambiguity.py.
    """
    hierarchy_db.add(Advisor(wid=99, name="Zeeshan Tariq", team="Gamma", company="Graana"))
    hierarchy_db.commit()
    from app.llm import entity_extractor
    entity_extractor._cache["loaded_at"] = 0

    response = handle_chat_message(hierarchy_db, "tell me about Zeeshan Tariq", session_id="h-5")

    assert response["type"] == "advisor"
    assert "Zeeshan Tariq" in response["reply"]
