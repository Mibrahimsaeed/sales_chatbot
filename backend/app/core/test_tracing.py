"""Phase 7 request tracing.

The property under test is the audit's M1 finding: a production failure
must be REPRODUCIBLE from what was recorded. Every wrong-person bug found
in the audit had to be diagnosed by hand-instrumenting a REPL, because
none of the decisive information — which identity candidates were
considered, what the planner decided, what SQL ran — was written down.
"""

import pytest

from app.core import tracing
from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def traced_db(db_session, monkeypatch):
    db_session.add(Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana", bm="Aimal Khan", rm="Aimal Khan"))
    db_session.add(SalesFunnel(wid=1, mtd_new_connect=40, mtd_followup_connect=2))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, target=1000, cleared=500))
    db_session.add(Advisor(wid=2, name="Yasir Ali", team="North/KPK", company="Agency21"))
    db_session.add(Advisor(wid=3, name="Yasir Ali", team="Downtown", company="IMARAT"))
    db_session.commit()

    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


@pytest.fixture()
def captured(monkeypatch):
    """Intercepts the emitted trace instead of parsing log output."""
    seen = []
    monkeypatch.setattr(tracing, "_emit", lambda trace: seen.append(trace.to_dict()))
    return seen


# ---- the trace context ----

def test_no_active_trace_makes_recording_a_no_op():
    """Population helpers are called from deep in the pipeline, which also
    runs outside a chat request (ETL, tests, startup). They must never
    require a guard at the call site."""
    assert tracing.current() is None
    tracing.record_entities({"team": "Alpha"})       # must not raise
    tracing.record_sql("SELECT 1")
    tracing.record_response({"type": "text"})


def test_trace_is_emitted_even_when_the_request_raises(captured):
    """A trace for a request that blew up is the most valuable one."""
    with pytest.raises(ValueError):
        with tracing.traced("boom", session_id="s"):
            raise ValueError("boom")
    assert len(captured) == 1
    assert captured[0]["query"] == "boom"


def test_trace_id_is_unique_per_request(captured):
    with tracing.traced("a"):
        pass
    with tracing.traced("b"):
        pass
    assert captured[0]["trace_id"] != captured[1]["trace_id"]


def test_population_failure_never_breaks_the_request(captured):
    class Exploding:
        @property
        def status(self):
            raise RuntimeError("nope")

    with tracing.traced("q"):
        tracing.record_identity(Exploding())   # must be swallowed
    assert len(captured) == 1


# ---- the nine required fields ----

def test_trace_captures_the_full_decision_chain(traced_db, captured):
    handle_chat_message(traced_db, "tell me about Waqar Haider", session_id="t1")
    trace = captured[-1]

    assert trace["query"] == "tell me about Waqar Haider"          # user query
    assert trace["entities"]["advisor_name"] == "Waqar Haider"      # extracted entities
    assert trace["identity"]["resolved_wid"] == 1                   # resolved WID
    assert trace["identity"]["candidates"][0]["wid"] == 1           # candidate advisors
    assert trace["plan"]["action"] == "lookup"                      # planner decision
    assert len(trace["sql"]) > 0                                    # generated SQL
    assert trace["row_count"] is not None                           # returned row count
    assert trace["response_type"] == "advisor"                      # final response
    assert "Waqar Haider" in trace["response_preview"]
    assert trace["duration_ms"] is not None


def test_ambiguous_request_records_every_candidate_considered(traced_db, captured):
    """The candidate list is what makes a "you gave me the wrong person"
    report diagnosable — it shows what the alternatives were."""
    handle_chat_message(traced_db, "tell me about Yasir Ali", session_id="t2")
    trace = captured[-1]

    assert trace["identity"]["status"] == "ambiguous"
    assert trace["identity"]["resolved_wid"] is None
    assert {c["wid"] for c in trace["identity"]["candidates"]} == {2, 3}
    assert trace["plan"]["action"] == "clarify_person"
    assert trace["plan"]["candidate_count"] == 2


def test_planner_decision_is_recorded_for_non_plan_resolutions(traced_db, captured):
    """Regression: the plan was recorded from Resolution in chat_service,
    which only carries it for kind=="plan" — so clarifications and
    IR-routed queries (the two shapes most likely to be reported as
    wrong) logged plan=null, hiding the routing decision."""
    handle_chat_message(traced_db, "tell me about Yasir Ali", session_id="t3")
    assert captured[-1]["plan"] is not None


def test_trace_records_the_matched_name_span(traced_db, captured):
    """Which words were treated as the name — the question that was
    unanswerable when the whole sentence went to the matcher."""
    handle_chat_message(traced_db, "tell me about Waqar Haider", session_id="t4")
    assert captured[-1]["identity"]["matched_span"] == "waqar haider"


def test_leaderboard_trace_records_ir_and_row_count(traced_db, captured):
    """The query INTENT and the response MODE are recorded separately.

    Phase 3 changed the last assertion from "leaderboard" to
    "metric_value". `response_type` used to be `ir.intent` passed
    straight through, so a ranking that matched exactly one advisor — see
    row_count below — was still reported and rendered as a leaderboard,
    header and all. The intent is still "leaderboard": the user did ask
    for a ranking. The ANSWER is one subject's figure.
    """
    handle_chat_message(traced_db, "top 3 advisors by connects", session_id="t5")
    trace = captured[-1]
    assert trace["ir"]["intent"] == "leaderboard"
    assert trace["ir"]["sort"]["metric"] == "total_connects"
    assert trace["row_count"] == 1
    assert trace["response_type"] == "metric_value"


def test_sql_is_captured_with_statements(traced_db, captured):
    handle_chat_message(traced_db, "top 3 advisors by connects", session_id="t6")
    statements = [e["statement"] for e in captured[-1]["sql"]]
    assert any("SELECT" in s.upper() for s in statements)


# ---- bounds ----

def test_sql_capture_is_bounded(captured):
    with tracing.traced("q"):
        for i in range(tracing.MAX_SQL_STATEMENTS + 20):
            tracing.record_sql(f"SELECT {i}")
    assert len(captured[-1]["sql"]) == tracing.MAX_SQL_STATEMENTS


def test_reply_preview_is_bounded(captured):
    with tracing.traced("q"):
        tracing.record_response({"type": "text", "reply": "x" * 5000, "data": None})
    assert len(captured[-1]["response_preview"]) == tracing.MAX_REPLY_CHARS


def test_entities_are_filtered_to_the_traceable_keys(captured):
    """The raw dict carries resolution objects and embedding candidate
    lists that aren't JSON-friendly and don't explain a routing decision."""
    with tracing.traced("q"):
        tracing.record_entities({
            "advisor_name": "X", "team": "Alpha",
            "advisor_resolution": object(), "advisor_matches": [{"huge": "blob"}],
        })
    entities = captured[-1]["entities"]
    assert entities == {"advisor_name": "X", "team": "Alpha"}
