"""The "did you mean the BCM or the Advisor?" clarification.

A name that grounds at more than one hierarchy level was already
detected, and the question was already asked. What was missing is the
record that a question is OUTSTANDING: the branch returned without
storing anything, so the answer arrived on the next turn as a bare,
contextless "BCM", grounded to nothing, and fell through to "I'm not
tracking that one".

The fixture makes the two readings return DIFFERENT numbers on purpose.
As a BCM the name means the group reporting to him (30); as an advisor it
means his own row (77). A fixture where both were 30 would pass whether
or not the chosen level was actually applied.
"""

import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import (
    Advisor, Attendance, Calls, Performance, PerformancePeriod, Pipeline,
    Portfolio, SalesFunnel, TeamTarget,
)
from app.database.session import Base
from app.llm import conversation_memory, entity_extractor, nlu_pipeline
from app.services.chat_service import handle_chat_message

NAME = "Khurram Ishaq Quraishi"
AS_BCM = 30      # 10 + 20, the two advisors reporting to him
AS_ADVISOR = 77  # his own row


@pytest.fixture(autouse=True)
def _clean_store():
    conversation_memory._store.clear()
    yield
    conversation_memory._store.clear()


@pytest.fixture()
def org(monkeypatch):
    from conftest import _ADVISOR_PROFILE_VIEW
    from app.core.config import settings
    from app.llm import llm_client, narrative, semantic_parser

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    # wids 1-2 report to him; wid 3 IS him, as an advisor in his own right.
    for wid, name, bcm, cr in ((1, "Yasir Ali", NAME, 10),
                               (2, "Waqar Haider", NAME, 20),
                               (3, NAME, "Usman Ghani", AS_ADVISOR)):
        s.add(Advisor(wid=wid, name=name, team="Blue Area", company="Graana",
                      management_lead=bcm, rm="Tariq Mehmood",
                      portfolio_lead="Fawad Hafeez", office="Beverly Center",
                      region="North/KPK", unit="A", in_master_sheet=True))
        s.add(SalesFunnel(wid=wid, mtd_cr=cr, mtd_new_connect=10,
                          mtd_followup_connect=0, mtd_new_meeting=1,
                          mtd_followup_meeting=0, mtd_conversion=1,
                          mtd_booking_stored=1))
        s.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=100,
                          cleared=50, pct=50))
        s.add(Pipeline(wid=wid, pipeline=100, overdue=1))
        s.add(Portfolio(wid=wid, value=500))
        s.add(Calls(wid=wid, answered_calls_mtd=20, connects_mtd=10))
        s.add(Attendance(wid=wid, biometric_mtd_ontime=15, biometric_mtd_late=5,
                         biometric_mtd_not_marked=0, login_mtd_ontime=18,
                         login_mtd_late=2, login_mtd_not_marked=0))
    s.add(TeamTarget(team="Blue Area", target=2000, achieved=1000,
                     achievement_pct=50.0))
    s.commit()
    with engine.begin() as conn:
        conn.exec_driver_sql(_ADVISOR_PROFILE_VIEW)

    monkeypatch.setattr(settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(settings, "use_llm_planner", False, raising=False)
    monkeypatch.setattr(llm_client, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False, raising=False)
    entity_extractor._cache["loaded_at"] = 0
    yield s
    s.close()


def _ask(org, session="s"):
    """Turn 1: the ambiguous question."""
    return handle_chat_message(org, f"CR of {NAME}", session_id=session)


# ---------------------------------------------------------------------
# Turn 1 records the outstanding question
# ---------------------------------------------------------------------


def test_the_question_is_asked(org):
    response = _ask(org)
    assert response["type"] == "clarification"
    assert response["options"] == ["BCM", "Advisor"]


def test_the_outstanding_question_is_stored(org):
    """The whole bug: this branch asked and stored nothing, so the answer
    arrived next turn with no record that anything was pending."""
    _ask(org)
    pending = conversation_memory.get_pending_level("s")

    assert pending is not None
    assert pending.value == NAME
    assert pending.levels == ["bcm", "advisor"]
    assert pending.original_text == f"CR of {NAME}"


def test_the_question_survives_the_session_state_rebuild(org):
    """set() reconstructs SessionState and carries only named fields —
    dropping this one would lose the question between the turn that asks
    it and the turn that answers it."""
    _ask(org)
    from app.llm.query_ir import MetricRef, QueryIR, Sort

    conversation_memory.set("s", QueryIR(
        intent="leaderboard", subject_level="advisor",
        metric=MetricRef(key="mtd_cleared"), sort=Sort(metric="mtd_cleared"),
        limit=10))
    assert conversation_memory.get_pending_level("s") is not None


# ---------------------------------------------------------------------
# Turn 2 answers it
# ---------------------------------------------------------------------


def test_answering_bcm_reads_the_name_as_the_bcm(org):
    _ask(org)
    response = handle_chat_message(org, "BCM", session_id="s")

    assert response["type"] != "clarification"
    assert str(AS_BCM) in response["reply"]
    assert str(AS_ADVISOR) not in response["reply"]


def test_answering_advisor_reads_the_name_as_the_advisor(org):
    _ask(org)
    response = handle_chat_message(org, "Advisor", session_id="s")

    assert response["type"] != "clarification"
    assert str(AS_ADVISOR) in response["reply"]


def test_the_two_answers_give_different_numbers(org):
    """Proof the choice is APPLIED rather than merely accepted."""
    _ask(org, session="b")
    as_bcm = handle_chat_message(org, "BCM", session_id="b")["reply"]
    _ask(org, session="a")
    as_advisor = handle_chat_message(org, "Advisor", session_id="a")["reply"]

    assert str(AS_BCM) in as_bcm
    assert str(AS_ADVISOR) in as_advisor
    assert as_bcm != as_advisor


def test_the_original_question_is_re_run_not_retyped(org):
    """The user asked for CR. Answering "BCM" must answer THAT, not
    return a bare profile of the BCM."""
    _ask(org)
    reply = handle_chat_message(org, "BCM", session_id="s")["reply"]
    assert "Client Registration" in reply


def test_answering_clears_the_pending_question(org):
    _ask(org)
    handle_chat_message(org, "BCM", session_id="s")
    assert conversation_memory.get_pending_level("s") is None


@pytest.mark.parametrize("answer", ["BCM", "bcm", "Advisor", "advisor"])
def test_the_answer_is_case_insensitive(org, answer):
    _ask(org)
    assert handle_chat_message(org, answer, session_id="s")["type"] != "clarification"


# ---------------------------------------------------------------------
# Answers that are not choices
# ---------------------------------------------------------------------


def test_an_unrecognised_short_answer_re_asks(org):
    _ask(org)
    response = handle_chat_message(org, "purple", session_id="s")

    assert response["type"] == "clarification"
    assert "BCM" in response["reply"]


def test_a_second_unrecognised_answer_gives_up_with_advice(org):
    """MAX_CLARIFY_ATTEMPTS — re-asking forever is its own failure."""
    _ask(org)
    handle_chat_message(org, "purple", session_id="s")
    response = handle_chat_message(org, "purple", session_id="s")

    assert response["type"] == "clarification"
    assert conversation_memory.get_pending_level("s") is None


def test_a_new_question_abandons_the_pending_one(org):
    """A message that stands on its own is the user moving on, not a
    malformed answer."""
    _ask(org)
    response = handle_chat_message(org, "Show me the top advisors by revenue",
                                   session_id="s")

    assert response["type"] == "leaderboard"
    assert conversation_memory.get_pending_level("s") is None


def test_naming_both_levels_is_not_a_choice(org):
    _ask(org)
    response = handle_chat_message(org, "bcm or advisor", session_id="s")
    assert response["type"] == "clarification"


# ---------------------------------------------------------------------
# Expiry and isolation
# ---------------------------------------------------------------------


def test_the_question_expires(org):
    """An answer to a question asked five minutes ago is not an answer —
    the same TTL every other pending state uses."""
    _ask(org)
    conversation_memory._store["s"].pending_level.asked_at = (
        time.time() - conversation_memory._PENDING_TTL_SECONDS - 1)

    assert conversation_memory.get_pending_level("s") is None


def test_an_expired_question_does_not_capture_the_next_message(org):
    _ask(org)
    conversation_memory._store["s"].pending_level.asked_at = (
        time.time() - conversation_memory._PENDING_TTL_SECONDS - 1)
    response = handle_chat_message(org, "BCM", session_id="s")

    assert str(AS_BCM) not in response["reply"]


def test_another_session_does_not_inherit_the_question(org):
    _ask(org, session="one")
    assert conversation_memory.get_pending_level("two") is None

    response = handle_chat_message(org, "BCM", session_id="two")
    assert str(AS_BCM) not in response["reply"]


def test_no_session_id_stores_nothing(org):
    handle_chat_message(org, f"CR of {NAME}", session_id=None)
    assert conversation_memory.get_pending_level(None) is None


# ---------------------------------------------------------------------
# The pin itself
# ---------------------------------------------------------------------


def test_pinning_removes_only_the_losing_groundings(org):
    from app.llm.entity_extractor import extract_entities

    entities = extract_entities(f"CR of {NAME}", org)
    assert entities.get("ambiguous_entity")

    pinned = nlu_pipeline._pin_level(entities, NAME, "bcm")
    assert pinned.get("bcm") == NAME
    assert not pinned.get("advisor_name")
    assert "ambiguous_entity" not in pinned


def test_pinning_leaves_an_unrelated_entity_alone(org):
    from app.llm.entity_extractor import extract_entities

    entities = extract_entities(f"CR of {NAME} in Blue Area", org)
    pinned = nlu_pipeline._pin_level(entities, NAME, "bcm")
    assert pinned.get("team") == "Blue Area"
