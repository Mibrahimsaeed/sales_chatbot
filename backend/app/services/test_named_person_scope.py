"""A question about a person answers about THAT PERSON.

    "connects of Naina"   ->  Naina's own figure, at her senior role
                              NOT the eleven people underneath her

THE BUG. The model routinely returns a HIERARCHY READ for a bare name —
target_level="advisor", subject_of="unit_head" — and since the repair
that nulled it was removed, the reply came back as a population of
Naina's advisors, each with their own connects. Her own number was absent
from a reply that looked entirely reasonable.

A read enumerates a LEVEL, and "connects of Naina" names none. That is
the same evidence (`level_word`) the highest-role promotion already runs
on, so the two decisions are made together in one place and cannot
disagree about what the question was.

TEAM SIZE FOLLOWS THE PERSON'S ROLE, not the code path that served the
reply. An advisor has no team of their own, so a headcount beside their
connects would read as though the figure covered one.
"""

import pytest

from app.database.models import Advisor, Calls, SalesFunnel
from app.llm import conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message

# Naina manages the whole unit; Zara a zone within it; Bilal a business
# centre; Sana manages nobody. Every manager also has an advisor row,
# which is what makes the junior reading always available.
PEOPLE = [
    # name,          rm,            portfolio_lead,  management_lead, connects
    ("Sana Iqbal",   "Naina Shah",  "Zara Qureshi",  "Bilal Anwar",   5),
    ("Bilal Anwar",  "Naina Shah",  "Zara Qureshi",  "Bilal Anwar",   7),
    ("Zara Qureshi", "Naina Shah",  "Zara Qureshi",  "Zara Qureshi",  11),
    ("Naina Shah",   "Naina Shah",  "Naina Shah",    "Naina Shah",    13),
]


@pytest.fixture()
def org(db_session, monkeypatch):
    for wid, (name, rm, pl, ml, connects) in enumerate(PEOPLE, start=1):
        db_session.add(Advisor(wid=wid, name=name, team="Alpha", company="Graana",
                               rm=rm, portfolio_lead=pl, management_lead=ml,
                               in_master_sheet=True))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=connects,
                                   mtd_followup_connect=0, mtd_cr=connects))
        db_session.add(Calls(wid=wid, connects_mtd=connects, connects_daily=0,
                             answered_calls_mtd=connects, answered_calls_daily=0))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "llm_first")
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()


def _model_returns_a_read(monkeypatch, name, metric="total_connects", level="advisor"):
    """What the model actually returns for a bare name: a hierarchy read
    under the person. Every test below starts from this, because it is
    the parse the fix has to cope with."""
    payload = {
        "intent": "filtered_list", "operation": "group_metric",
        "subject_level": level,
        "subjects": [{"type": level, "value": name, "match_confidence": 1.0}],
        "metric": {"key": metric, "confidence": 0.9},
        "metrics": [], "filters": [], "filter_tree": None,
        "time_range": {"mode": "snapshot", "period": "MTD",
                       "compare_to": None, "confidence": 0.9},
        "sort": {"metric": metric, "direction": "desc"},
        "limit": None, "group_by": None,
        "target_level": "advisor", "subject_of": "unit_head", "relation": "subtree",
        "overall_confidence": 0.95, "intent_confidence": 0.95,
    }
    monkeypatch.setattr(semantic_parser, "call_llm_structured",
                        lambda p, s, schema_name=None: payload)


def _model_returns_no_read(monkeypatch, name, metric="total_connects", level="advisor"):
    """A clean parse, for the cases where the query names a level and the
    model has no reason to invent a traversal."""
    payload = {
        "intent": "filtered_list", "operation": "group_metric",
        "subject_level": level,
        "subjects": [{"type": level, "value": name, "match_confidence": 1.0}],
        "metric": {"key": metric, "confidence": 0.9},
        "metrics": [], "filters": [], "filter_tree": None,
        "time_range": {"mode": "snapshot", "period": "MTD",
                       "compare_to": None, "confidence": 0.9},
        "sort": {"metric": metric, "direction": "desc"},
        "limit": None, "group_by": None,
        "target_level": None, "subject_of": None, "relation": "subtree",
        "overall_confidence": 0.95, "intent_confidence": 0.95,
    }
    monkeypatch.setattr(semantic_parser, "call_llm_structured",
                        lambda p, s, schema_name=None: payload)


def _ask(db, text):
    conversation_memory._store.clear()
    return handle_chat_message(db, text, session_id=None)


# ---------------------------------------------------------------------
# 1-4: a named person answers about themselves, at their own level
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", ["Naina Shah", "Zara Qureshi", "Bilal Anwar"])
def test_a_manager_named_without_a_level_is_not_expanded(name, org, monkeypatch):
    """One row — theirs. Not a list of the people underneath them."""
    _model_returns_a_read(monkeypatch, name)

    result = _ask(org, f"connects of {name}")

    assert len(result.get("data") or []) == 1, \
        f"{name}: expanded into {len(result['data'])} rows"
    assert name in result["reply"]


def test_a_team_named_without_a_level_is_not_expanded(org, monkeypatch):
    _model_returns_no_read(monkeypatch, "Alpha", level="team")

    assert len(_ask(org, "connects of Alpha").get("data") or []) == 1


# ---------------------------------------------------------------------
# 5-6: Team Size follows the person's role
# ---------------------------------------------------------------------

def test_an_advisor_only_person_shows_no_team_size(org, monkeypatch):
    """Sana manages nobody, so a headcount beside her connects would read
    as though the number covered a team."""
    _model_returns_a_read(monkeypatch, "Sana Iqbal")

    assert "Team Size" not in _ask(org, "connects of Sana Iqbal")["reply"]


@pytest.mark.parametrize("name", ["Naina Shah", "Zara Qureshi", "Bilal Anwar"])
def test_a_manager_shows_a_team_size(name, org, monkeypatch):
    _model_returns_a_read(monkeypatch, name)

    assert "Team Size" in _ask(org, f"connects of {name}")["reply"], name


# ---------------------------------------------------------------------
# 7-8: an explicitly stated level still wins
# ---------------------------------------------------------------------

def test_as_an_advisor_answers_at_advisor_level_with_no_team_size(org, monkeypatch):
    _model_returns_no_read(monkeypatch, "Naina Shah")

    reply = _ask(org, "connects of Naina Shah as an advisor")["reply"]

    assert "Naina Shah" in reply
    assert "Team Size" not in reply


def test_as_unit_head_answers_at_unit_head_level_with_a_team_size(org, monkeypatch):
    _model_returns_no_read(monkeypatch, "Naina Shah", level="unit_head")

    reply = _ask(org, "connects of unit head Naina Shah")["reply"]

    assert "Team Size" in reply


# ---------------------------------------------------------------------
# 9-10: CR follows the same rule; leaderboards are untouched
# ---------------------------------------------------------------------

def test_client_registrations_show_no_team_size_for_anyone(org, monkeypatch):
    """CR HAS NO BLOCK TO CARRY ONE, and that is a deliberate earlier
    decision rather than an oversight here.

    CR's rate reaches the reader through `companion`, which states it in
    the sentence. It was briefly given a bundle so it could carry a Team
    Size line, and that was reverted because it printed the rate twice —
    once in the sentence, once as a bullet. With no block, there is
    nowhere for a Team Size to go, for a manager or an advisor.

    So this pins the agreed behaviour rather than the requested one: a
    manager's CR answer shows no Team Size. Restoring it means choosing
    between the duplicate rate and a CR-only Team Size line.
    """
    for name in ("Sana Iqbal", "Naina Shah"):
        _model_returns_a_read(monkeypatch, name, metric="client_registrations")
        reply = _ask(org, f"client registrations of {name}")["reply"]
        assert "Team Size" not in reply, name
        assert name in reply, "the figure itself is unaffected"


def test_a_genuine_hierarchy_read_still_enumerates(org, monkeypatch):
    """The guard is the absence of a level WORD. "advisors under Naina"
    names one and keeps its read — the fix must not have removed the
    ability to list a team."""
    _model_returns_a_read(monkeypatch, "Naina Shah", level="unit_head")

    rows = _ask(org, "advisors under Naina Shah").get("data") or []

    assert len(rows) > 1, "a real read must still enumerate"


def test_a_leaderboard_is_unchanged(org, monkeypatch):
    """Nothing here may touch a ranking: it names no single person and has
    its own rows by definition."""
    payload = {
        "intent": "leaderboard", "operation": "leaderboard",
        "subject_level": "advisor", "subjects": [],
        "metric": {"key": "total_connects", "confidence": 0.9},
        "metrics": [], "filters": [], "filter_tree": None,
        "time_range": {"mode": "snapshot", "period": "MTD",
                       "compare_to": None, "confidence": 0.9},
        "sort": {"metric": "total_connects", "direction": "desc"},
        "limit": 10, "group_by": None,
        "target_level": None, "subject_of": None, "relation": "subtree",
        "overall_confidence": 0.95, "intent_confidence": 0.95,
    }
    monkeypatch.setattr(semantic_parser, "call_llm_structured",
                        lambda p, s, schema_name=None: payload)

    rows = _ask(org, "top advisors by connects").get("data") or []

    assert len(rows) == len(PEOPLE)
