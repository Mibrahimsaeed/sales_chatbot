"""The rule-based execution trace, end to end through handle_chat_message.

These tests pin the trace to the thing it exists to do: make the FIRST
stage that dropped a query's requirement identifiable from the log alone.
The canonical case is "tell me about X's team" — the relational keyword
IS detected, the hierarchy scorer nonetheless declines, and the advisor
profile answers instead. Every link in that chain has to be readable.
"""

import pytest

from app.core import audit
from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def audited_db(db_session, monkeypatch, tmp_path):
    db_session.add(Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana",
                           bm="Aimal Khan", rm="Aimal Khan", portfolio_lead="Kaleem Ullah"))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=40, mtd_followup_connect=2))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, target=1000, cleared=500))
    db_session.add(Advisor(wid=2, name="Sana Tariq", team="Blue Area", company="Graana",
                           bm="Aimal Khan", rm="Aimal Khan", portfolio_lead="Kaleem Ullah"))
    db_session.commit()

    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "llm_first")
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    monkeypatch.setattr(audit.settings, "chat_audit_debug", True)
    monkeypatch.setattr(audit.settings, "chat_audit_console", False)
    monkeypatch.setattr(audit.settings, "chat_audit_dir", str(tmp_path))
    monkeypatch.setattr(audit, "_file_logger", None)
    monkeypatch.setattr(audit, "_file_logger_path", None)

    yield db_session

    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


def _audit_text() -> str:
    return audit.log_path().read_text()


def test_every_requested_field_is_present(audited_db):
    handle_chat_message(audited_db, "tell me about Waqar Haider", session_id="s")
    text = _audit_text()

    for field in ("RULE-BASED AUDIT", "User Query:", "Routing Decision:",
                  "Detected Intent:", "Confidence:", "Extracted Entities:",
                  "Extracted Keywords:", "Resolved Advisor:", "Generated QueryIR:",
                  "Selected Planner Function(s):", "Generated SQL/API Calls",
                  "Retrieved Data Summary:", "Formatter Selected:", "Final Response:"):
        assert field in text, f"missing section: {field}"


def test_routing_decision_states_why_the_llm_was_never_called(audited_db):
    handle_chat_message(audited_db, "tell me about Waqar Haider", session_id="s")
    text = _audit_text()

    assert "routing: rule_based_path" in text
    assert "_RULE_BASED_ACTIONS" in text
    assert "NO LLM call is made" in text


def test_the_teams_question_trace_shows_where_the_requirement_was_dropped(audited_db):
    """The whole objective, as one assertion chain: the keyword was seen,
    the intent that would have honoured it never scored, and a formatter
    with no team section produced the answer."""
    handle_chat_message(audited_db, "tell me about Waqar Haider's team", session_id="s")
    text = _audit_text()

    # 1. the signal WAS detected — so the loss is not in keyword extraction
    assert "relational: DETECTED" in text
    # 2. but the intent that acts on it never became a candidate
    assert "declined _score_hierarchy" in text
    # 3. and the person-profile intent won instead
    assert "Detected Intent: lookup" in text
    assert "advisor_profile" in text
    # 4. answered by a formatter that has no team section at all
    assert "Formatter Selected: format_advisor_reply" in text


def test_sql_is_captured_while_the_trace_is_still_open(audited_db):
    """Regression: the audit context wraps the tracing context, so reading
    SQL at format time yielded zero — indistinguishable from a query that
    ran none."""
    handle_chat_message(audited_db, "tell me about Waqar Haider", session_id="s")
    text = _audit_text()

    assert "Generated SQL/API Calls (0)" not in text
    assert "SELECT" in text


def test_resolved_advisor_records_name_and_id(audited_db):
    handle_chat_message(audited_db, "tell me about Waqar Haider", session_id="s")
    text = _audit_text()
    assert "name='Waqar Haider'  wid=1" in text


def test_rule_path_queries_report_no_queryir(audited_db):
    """The rule-based path builds no IR — stated explicitly, since an
    empty field would read as "the log failed to capture it"."""
    handle_chat_message(audited_db, "tell me about Waqar Haider", session_id="s")
    assert "no QueryIR is built" in _audit_text()


def test_shortcut_queries_are_traced_too(audited_db):
    """A greeting is equally LLM-free; it must not vanish from the trace
    just because it never reached the planner.

    Phase 1 (routing refactor) inverted the last assertion. It used to
    read "neither the planner nor entity [extraction runs]", which was
    true when classify_intent() ran FIRST and was handed a hardcoded
    empty dict. That ordering is the P1 defect: it made
    `attendance_rate` and `login_rate` unreachable for any person-scoped
    question. Entity extraction now runs BEFORE the shortcut check and
    the shortcut is a fallback, so the old wording would document
    behaviour the system no longer has.
    """
    handle_chat_message(audited_db, "hello there", session_id="s")
    text = _audit_text()
    assert "RULE-BASED AUDIT" in text
    assert "shortcut:greeting" in text
    assert "routing allowed it" in text.lower()


def test_no_rule_block_when_the_audit_flag_is_off(audited_db, monkeypatch):
    monkeypatch.setattr(audit.settings, "chat_audit_debug", False)
    handle_chat_message(audited_db, "tell me about Waqar Haider", session_id="s")
    assert not audit.log_path().exists()


def test_business_logic_is_unchanged_by_the_audit(audited_db, monkeypatch):
    """The reply must be byte-identical with auditing on and off — the
    trace is an observer, and a diagnostic that changes the answer would
    invalidate every conclusion drawn from it."""
    with_audit = handle_chat_message(audited_db, "tell me about Waqar Haider", session_id="a")

    monkeypatch.setattr(audit.settings, "chat_audit_debug", False)
    without_audit = handle_chat_message(audited_db, "tell me about Waqar Haider", session_id="b")

    assert with_audit["reply"] == without_audit["reply"]
    assert with_audit["type"] == without_audit["type"]
