"""Regression test for a reported bug: 'who was not marked today' was
returning late arrivals too. Root cause was intent_detector._rule_attendance
routing ANY attendance-related query without a team straight to the
generic attendance_check shortcut (get_attendance_issues(), which only
excludes 'On Time' and mixes every other status together), never reaching
query_planner's attendance_filter action — which calls
get_attendance_by_status() and filters to the exact status asked about."""

import pytest

from app.database.models import Advisor, Attendance
from app.llm import conversation_memory, entity_extractor
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def attendance_db(db_session):
    db_session.add_all([
        Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana"),
        Advisor(wid=2, name="Ali Raza", team="Downtown", company="IMARAT"),
        Advisor(wid=3, name="Sana Khan", team="Downtown", company="IMARAT"),
    ])
    db_session.add(Attendance(wid=1, biometric_status="Not Marked"))
    db_session.add(Attendance(wid=2, biometric_status="Late"))
    db_session.add(Attendance(wid=3, biometric_status="On Time"))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def test_not_marked_query_excludes_late_arrivals(attendance_db):
    response = handle_chat_message(attendance_db, "who was not marked today", session_id="att-1")
    names = {row["name"] for row in response["data"]}
    assert names == {"Waqar Haider"}
    assert "Ali Raza" not in names


def test_late_query_excludes_not_marked(attendance_db):
    response = handle_chat_message(attendance_db, "who was late today", session_id="att-2")
    names = {row["name"] for row in response["data"]}
    assert names == {"Ali Raza"}
    assert "Waqar Haider" not in names


def test_generic_attendance_query_still_returns_all_issues(attendance_db):
    # no specific status named -> the broad "any issue" shortcut still
    # applies, on purpose
    response = handle_chat_message(attendance_db, "show attendance issues today", session_id="att-3")
    names = {row["name"] for row in response["data"]}
    assert names == {"Waqar Haider", "Ali Raza"}
