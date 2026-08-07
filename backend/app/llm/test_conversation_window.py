"""The conversation window: recent MESSAGES sent to the LLM.

This complements the structured layer rather than replacing it. The
deterministic path (conversation_memory.last_ir, ir_patcher,
conversation_context.merge, cross_turn_resolver) already resolves almost
every follow-up on its own, and the tests at the bottom pin that it still
does. The window exists for references that layer cannot see — wording
that appeared only in a reply, or a subject that never grounded.

Every limit is asserted, because an unbounded window is the standard way
this feature goes wrong: it grows until it crowds out the schema the
prompt needs, and does so silently.
"""

import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.database.models import (
    Advisor, Attendance, Calls, Performance, PerformancePeriod, Pipeline,
    Portfolio, SalesFunnel, TeamTarget,
)
from app.database.session import Base
from app.llm import conversation_memory, entity_extractor
from app.llm.prompt_builder import build_ir_prompt
from app.services.chat_service import handle_chat_message


@pytest.fixture(autouse=True)
def _clean_store():
    conversation_memory._store.clear()
    yield
    conversation_memory._store.clear()


@pytest.fixture()
def org(monkeypatch):
    from conftest import _ADVISOR_PROFILE_VIEW
    from app.llm import llm_client, semantic_parser

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    people = [(1, "Ahmed Khan", "Downtown", "Graana"),
              (2, "Sara Ali", "Downtown", "IMARAT"),
              (3, "Yasir Ali", "Blue Area", "Graana")]
    for wid, name, team, company in people:
        s.add(Advisor(wid=wid, name=name, team=team, company=company,
                      rm="Tariq Mehmood", portfolio_lead="Fawad Hafeez",
                      management_lead="Usman Ghani", office="Beverly Center",
                      region="North/KPK", unit="A", in_master_sheet=True))
        s.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000,
                          cleared=1000 * wid, pct=10.0 * wid))
        s.add(Performance(wid=wid, period=PerformancePeriod.YTD, target=10000,
                          cleared=10000 * wid, pct=10.0 * wid))
        s.add(SalesFunnel(wid=wid, mtd_new_connect=10 * wid, mtd_followup_connect=0,
                          mtd_cr=5 * wid, mtd_new_meeting=wid,
                          mtd_followup_meeting=0, mtd_conversion=wid,
                          mtd_booking_stored=wid))
        s.add(Pipeline(wid=wid, pipeline=1000 * wid, overdue=wid))
        s.add(Portfolio(wid=wid, value=5000 * wid))
        s.add(Calls(wid=wid, answered_calls_mtd=20 * wid, connects_mtd=10 * wid))
        s.add(Attendance(wid=wid, biometric_mtd_ontime=15, biometric_mtd_late=5,
                         biometric_mtd_not_marked=0, login_mtd_ontime=18,
                         login_mtd_late=2, login_mtd_not_marked=0))
    for team in ("Downtown", "Blue Area"):
        s.add(TeamTarget(team=team, target=2000, achieved=1000, achievement_pct=50.0))
    s.commit()
    with engine.begin() as conn:
        conn.exec_driver_sql(_ADVISOR_PROFILE_VIEW)

    monkeypatch.setattr(settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(settings, "use_llm_planner", False, raising=False)
    monkeypatch.setattr(llm_client, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    # The narrative polish step is a separate LLM call site; off here so
    # these tests never reach a provider.
    from app.llm import narrative
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False, raising=False)
    entity_extractor._cache["loaded_at"] = 0
    yield s
    s.close()


# ---------------------------------------------------------------------
# The window itself
# ---------------------------------------------------------------------


def test_a_turn_is_recorded_and_read_back():
    conversation_memory.record_turn("w1", "user", "hello")
    conversation_memory.record_turn("w1", "assistant", "hi")
    assert conversation_memory.recent_turns("w1") == [
        ("user", "hello"), ("assistant", "hi")]


def test_history_is_bounded_by_the_configured_turn_count():
    """Unbounded growth is the standard failure of this feature."""
    for i in range(50):
        conversation_memory.record_turn("w2", "user", f"message {i}")

    kept = conversation_memory.recent_turns("w2")
    assert len(kept) <= settings.conversation_window_turns * 2
    # the SURVIVORS are the most recent ones
    assert kept[-1] == ("user", "message 49")


def test_history_is_bounded_by_the_character_budget():
    for i in range(20):
        conversation_memory.record_turn("w3", "assistant", "x" * 500)

    rendered = sum(len(t) for _role, t in conversation_memory.recent_turns("w3"))
    assert rendered <= settings.conversation_window_chars


def test_a_single_oversized_turn_is_truncated_not_dropped():
    """Dropping it loses the turn a follow-up most likely refers to;
    passing it whole crowds out the schema."""
    conversation_memory.record_turn("w4", "assistant", "y" * 9000)
    kept = conversation_memory.recent_turns("w4")

    assert len(kept) == 1
    assert len(kept[0][1]) <= settings.conversation_window_chars
    assert kept[0][1].endswith("…")


def test_the_newest_turns_survive_the_character_budget():
    conversation_memory.record_turn("w5", "user", "z" * 1100)
    conversation_memory.record_turn("w5", "user", "the newest message")

    kept = conversation_memory.recent_turns("w5")
    assert kept[-1] == ("user", "the newest message")


def test_a_new_conversation_inherits_nothing():
    conversation_memory.record_turn("a", "user", "Ahmed's revenue")
    assert conversation_memory.recent_turns("b") == []


def test_an_expired_conversation_returns_nothing():
    conversation_memory.record_turn("w6", "user", "hello")
    conversation_memory._store["w6"].saved_at = (
        time.time() - conversation_memory._TTL_SECONDS - 1)
    assert conversation_memory.recent_turns("w6") == []


def test_recording_a_turn_does_not_revive_an_expired_conversation():
    """The TTL means "this long later isn't a follow-up". Appending a
    message is not evidence against that — refreshing saved_at here let a
    new message bind a pronoun to someone named half an hour earlier."""
    conversation_memory.record_turn("w7", "user", "Tell me about Ahmed Khan")
    expired_at = time.time() - conversation_memory._TTL_SECONDS - 1
    conversation_memory._store["w7"].saved_at = expired_at

    conversation_memory.record_turn("w7", "user", "and his pipeline?")
    assert conversation_memory.recent_turns("w7") == [("user", "and his pipeline?")]


def test_no_session_id_records_nothing():
    conversation_memory.record_turn(None, "user", "hello")
    assert conversation_memory.recent_turns(None) == []


def test_an_empty_message_is_not_recorded():
    conversation_memory.record_turn("w8", "assistant", "")
    assert conversation_memory.recent_turns("w8") == []


# ---------------------------------------------------------------------
# The window reaches the prompt
# ---------------------------------------------------------------------


def test_the_prompt_carries_the_recent_turns():
    prompt = build_ir_prompt(
        "what about his pipeline?", ["Downtown"], ["Graana"],
        grounded_entities={},
        recent_turns=[("user", "Show me Ahmed Khan's revenue"),
                      ("assistant", "Ahmed Khan has 1,000 MTD revenue cleared.")],
    )
    assert "Recent conversation" in prompt
    assert "Ahmed Khan's revenue" in prompt


def test_the_prompt_omits_the_block_when_there_is_no_history():
    prompt = build_ir_prompt("top advisors by revenue", ["Downtown"], ["Graana"],
                             grounded_entities={}, recent_turns=[])
    assert "Recent conversation" not in prompt


def test_the_structured_prior_ir_is_still_sent_alongside():
    """The window complements the IR; it does not replace it."""
    prompt = build_ir_prompt(
        "what about pipeline?", ["Downtown"], ["Graana"], grounded_entities={},
        prior_ir_json='{"intent":"leaderboard"}',
        recent_turns=[("user", "Downtown revenue")],
    )
    assert "Previous turn's resolved query" in prompt
    assert "Recent conversation" in prompt


# ---------------------------------------------------------------------
# End to end — the window fills as the conversation runs
# ---------------------------------------------------------------------


def test_the_conversation_is_recorded_as_it_happens(org):
    handle_chat_message(org, "Show me Ahmed Khan's revenue", session_id="e1")
    kept = conversation_memory.recent_turns("e1")

    assert kept[0] == ("user", "Show me Ahmed Khan's revenue")
    assert kept[1][0] == "assistant" and kept[1][1]


def test_a_multi_turn_conversation_keeps_only_the_recent_window(org):
    for _ in range(6):
        handle_chat_message(org, "Top advisors by revenue", session_id="e2")

    assert len(conversation_memory.recent_turns("e2")) <= (
        settings.conversation_window_turns * 2)


def test_two_sessions_do_not_share_a_window(org):
    handle_chat_message(org, "Show me Ahmed Khan's revenue", session_id="e3a")
    handle_chat_message(org, "Top advisors by revenue", session_id="e3b")

    # Asserted on the USER turns. A leaderboard reply legitimately names
    # Ahmed Khan — he is in the data — so scanning the assistant text
    # would fail on correct behaviour rather than on context bleed.
    asked = [text for role, text in conversation_memory.recent_turns("e3b")
             if role == "user"]
    assert asked == ["Top advisors by revenue"]


# ---------------------------------------------------------------------
# The deterministic layer still owns resolution
# ---------------------------------------------------------------------


def test_a_pronoun_on_a_person_still_resolves_without_the_llm(org):
    """The window is additional context for the LLM. With the LLM off,
    the structured layer must resolve this exactly as before."""
    handle_chat_message(org, "Show me Ahmed Khan's revenue", session_id="d1")
    response = handle_chat_message(org, "What about his pipeline?", session_id="d1")

    assert "Ahmed Khan" in response["reply"]
    assert "pipeline" in response["reply"].lower()


def test_an_ellipsis_filter_still_resolves_without_the_llm(org):
    handle_chat_message(org, "Show Downtown pipeline", session_id="d2")
    response = handle_chat_message(org, "Now only Graana", session_id="d2")

    assert "Downtown" in response["reply"]
    assert "Graana" in response["reply"]


def test_a_metric_follow_up_still_resolves_without_the_llm(org):
    handle_chat_message(org, "Blue Area revenue", session_id="d3")
    response = handle_chat_message(org, "what about connects?", session_id="d3")

    assert "Blue Area" in response["reply"]
    assert "Connects" in response["reply"]


def test_a_period_follow_up_still_resolves_without_the_llm(org):
    handle_chat_message(org, "Blue Area revenue", session_id="d4")
    response = handle_chat_message(org, "year to date", session_id="d4")

    assert "YTD" in response["reply"]
