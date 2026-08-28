"""Reverse-role lookup end to end (M2).

The acceptance check: "who is X's portfolio lead" and "who is X's
management lead" now answer with the person, and every reverse phrasing
that already worked still answers identically.

Nothing downstream of the vocabulary changed — no planner, no service, no
formatter — so these tests are really asking whether routing now reaches
machinery that was already correct and previously unreachable.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def db(db_session, monkeypatch):
    db_session.add(Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana",
                           bm="Kaleem Ullah", zm="Adeel Dogar", office="Gulberg BC",
                           rm="Tariq Mehmood", portfolio_lead="Sana Malik",
                           management_lead="Imran Shah"))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=40, mtd_followup_connect=2))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, target=1000, cleared=500))
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
# The acceptance criteria
# ---------------------------------------------------------------------

def test_portfolio_lead_now_resolves(db):
    r = handle_chat_message(db, "Who is Waqar Haider's portfolio lead", session_id=None)
    assert r["type"] == "manager"
    assert "Sana Malik" in r["reply"]


def test_management_lead_now_resolves(db):
    r = handle_chat_message(db, "Who is Waqar Haider's management lead", session_id=None)
    assert r["type"] == "manager"
    assert "Imran Shah" in r["reply"]


# ---------------------------------------------------------------------
# Preservation — every reverse role that already worked
# ---------------------------------------------------------------------

# REBOUND BY PHASE 3. "Unit Head" now reads Advisor.rm and "Zonal Head"
# reads Advisor.portfolio_lead, so RM and Unit Head are the SAME level —
# the verified chain has one level there, not two. `bm`/`zm` keep their
# own labels and answer from their own columns.
@pytest.mark.parametrize("query,expected", [
    ("Who is Waqar Haider's BM", "Waqar Haider's BM is Kaleem Ullah."),
    ("Who is Waqar Haider's ZM", "Waqar Haider's ZM is Adeel Dogar."),
    ("Who is Waqar Haider's unit head", "Waqar Haider's Unit Head is Tariq Mehmood."),
    ("Who is Waqar Haider's RM", "Waqar Haider's Unit Head is Tariq Mehmood."),
    ("Who is Waqar Haider's zonal head", "Waqar Haider's Zonal Head is Sana Malik."),
    ("Who is Waqar Haider's portfolio lead", "Waqar Haider's Zonal Head is Sana Malik."),
    ("Who does Waqar Haider report to", "Waqar Haider's Unit Head is Tariq Mehmood."),
    ("Who is Waqar Haider's manager", "Waqar Haider's Unit Head is Tariq Mehmood."),
])
def test_previously_working_reverse_queries_are_byte_identical(db, query, expected):
    r = handle_chat_message(db, query, session_id=None)
    assert r["type"] == "manager"
    assert r["reply"] == expected


def test_newly_recognised_roles_route_to_the_right_column(db):
    """These phrasings were understood by level detection but never
    reached it. Each must now answer from its own column."""
    for query, expected in (
        ("Who is Waqar Haider's division head", "Tariq Mehmood"),  # -> unit_head (rm)
        ("Who is Waqar Haider's zone head", "Sana Malik"),         # -> zonal_head (portfolio_lead)
        ("Who is Waqar Haider's regional head", "Tariq Mehmood"),  # -> unit_head (rm)
        ("Who is Waqar Haider's business centre", "Gulberg BC"),   # -> office
        ("Who is Waqar Haider's management lead", "Imran Shah"),   # -> bcm
    ):
        r = handle_chat_message(db, query, session_id=None)
        assert r["type"] == "manager", query
        assert expected in r["reply"], query


# ---------------------------------------------------------------------
# Guards — the boundaries M2 must not cross
# ---------------------------------------------------------------------

def test_person_lookup_is_untouched(db):
    r = handle_chat_message(db, "tell me about Waqar Haider", session_id=None)
    assert r["type"] == "advisor"


def test_forward_hierarchy_is_untouched(db):
    """Scoped by the Unit Head value (Advisor.rm) after the rebind —
    "Kaleem Ullah" is now the BM column, which is not a chain level."""
    r = handle_chat_message(db, "Who works under Tariq Mehmood", session_id=None)
    assert r["type"] == "roster"


def test_m1_group_reference_is_untouched(db):
    """"X's team" must remain a group reference, not become a reverse
    question — team carries no role aliases precisely so this holds."""
    r = handle_chat_message(db, "Tell me about Waqar Haider's team", session_id=None)
    assert r["type"] != "manager"


def test_absent_role_says_so_rather_than_answering_differently(db):
    db.add(Advisor(wid=2, name="Nadia Rehman", team="Blue Area", company="Graana",
                   bm="Kaleem Ullah", rm="Kaleem Ullah", portfolio_lead=None))
    db.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()

    r = handle_chat_message(db, "Who is Nadia Rehman's portfolio lead", session_id=None)
    assert r["type"] == "not_found"
    # "portfolio lead" is the Zonal Head level after the Phase 3 rebind.
    assert "Zonal Head" in r["reply"]
