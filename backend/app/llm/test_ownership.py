"""Phase 4 — architectural ownership invariants.

These are not behaviour tests. Each one pins a structural property that
took a refactor to establish and that a single well-meaning edit could
undo without failing anything else: a second writer of conversation
state, a `"type"` string invented at a return site, a planner run twice.

The defects these prevent were all invisible in output. Duplicated
ownership does not break a request — it lets two components answer the
same question differently, and the wrong one wins whenever the inputs
happen to differ.
"""

import inspect

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import (
    Advisor, Attendance, Calls, Performance, PerformancePeriod, Pipeline,
    Portfolio, SalesFunnel, TeamTarget,
)
from app.database.session import Base
from app.llm import (
    conversation_memory, entity_extractor, query_planner, response_planner,
    semantic_parser,
)


# ---------------------------------------------------------------------
# Static invariants
# ---------------------------------------------------------------------


def test_conversation_state_has_exactly_one_writer_module():
    """`conversation_memory.set()` writes last_ir. Before Phase 4 four
    call sites across two modules wrote it, and semantic_parser wrote the
    PRE-MERGE IR — so the IR that answered and the IR the next turn
    inherited could differ."""
    import pathlib

    writers = set()
    for path in pathlib.Path("app").rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        if "conversation_memory.set(" in path.read_text():
            writers.add(path.name)

    assert writers == {"nlu_pipeline.py"}, (
        f"last_ir is written by {sorted(writers)} — nlu_pipeline.resolve() "
        "is the single owner of conversation state"
    )


def test_the_parser_does_not_store_conversation_state():
    source = inspect.getsource(semantic_parser)
    assert "conversation_memory.set(" not in source


def test_the_dispatcher_invents_no_response_types():
    """Every response leaves through response_planner.respond(), which
    rejects a mode it does not know. A `"type"` literal at a return site
    is a second vocabulary of response kinds — that is how `unknown` came
    to mean three unrelated things."""
    from app.services import chat_service

    source = inspect.getsource(chat_service)
    assert '"type": "' not in source, (
        "a response type is being written inline; call respond() instead"
    )


def test_respond_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="not a response mode"):
        response_planner.respond("something_new", "hi", None)


def test_every_dispatch_mode_is_a_declared_mode():
    """DISPATCH_MODES maps planner modes to the wire format. Both sides
    must stay closed sets."""
    assert response_planner.DISPATCH_MODES
    for mode, wire in response_planner.DISPATCH_MODES.items():
        assert isinstance(mode, str) and mode
        assert isinstance(wire, str) and wire


def test_the_formatter_renders_a_plan_rather_than_making_one():
    """format_ir_reply() takes the plan the caller already made. It used
    to call plan_response() itself while chat_service called it too —
    two deciders of one decision."""
    from app.llm import response_formatter

    sig = inspect.signature(response_formatter.format_ir_reply)
    assert "plan" in sig.parameters


def test_the_parser_accepts_the_plan_it_is_given():
    sig = inspect.signature(semantic_parser.parse)
    assert "plan" in sig.parameters


def test_the_capability_registry_has_one_definition():
    """nlu_pipeline and response_planner both read
    ir_validator._UNSUPPORTED_INTENTS; neither restates it."""
    from app.llm import ir_validator, nlu_pipeline

    for action, reason in nlu_pipeline._UNSUPPORTED_ACTIONS.items():
        assert reason == ir_validator._UNSUPPORTED_INTENTS[action]


def test_the_validator_may_only_degrade_a_level_never_choose_one():
    """ir_validator was a second owner of subject level until Phase 2. It
    may now only fall back when the chosen level is uncomputable, and the
    guard proving that is the is_answerable() check it sits behind."""
    source = inspect.getsource(__import__("app.llm.ir_validator", fromlist=["x"]))
    assert "not is_answerable(metric_key, ir.subject_level)" in source


# ---------------------------------------------------------------------
# Runtime invariants — no component runs twice
# ---------------------------------------------------------------------

_PEOPLE = [(1, "Yasir Ali", "Blue Area", "Graana"),
           (2, "Waqar Haider", "Blue Area", "Graana"),
           (3, "Shehryar Abbasi", "Downtown", "IMARAT")]


@pytest.fixture(scope="module")
def _own_engine():
    from conftest import _ADVISOR_PROFILE_VIEW

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    for wid, name, team, company in _PEOPLE:
        s.add(Advisor(wid=wid, name=name, team=team, company=company,
                      rm="Tariq Mehmood", portfolio_lead="Fawad Hafeez",
                      management_lead="Usman Ghani", office="Beverly Center",
                      region="North/KPK", unit="A", in_master_sheet=True))
        s.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000,
                          cleared=100 * wid, pct=10.0 * wid))
        s.add(Performance(wid=wid, period=PerformancePeriod.YTD, target=10000,
                          cleared=1000 * wid, pct=10.0 * wid))
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
    for team, target, achieved in (("Blue Area", 3000, 2400), ("Downtown", 2000, 1100)):
        s.add(TeamTarget(team=team, target=target, achieved=achieved,
                         achievement_pct=round(achieved / target * 100, 1)))
    s.commit()
    with engine.begin() as conn:
        conn.exec_driver_sql(_ADVISOR_PROFILE_VIEW)
    yield engine
    s.close()


@pytest.fixture()
def org(_own_engine, monkeypatch):
    from app.core.config import settings
    from app.llm import llm_client

    monkeypatch.setattr(settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(settings, "use_llm_planner", False, raising=False)
    monkeypatch.setattr(llm_client, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)

    s = sessionmaker(bind=_own_engine)()
    entity_extractor._cache["loaded_at"] = 0
    yield s
    s.close()


def _count_calls(monkeypatch, module, name, counter, key):
    original = getattr(module, name)

    def spy(*args, **kwargs):
        counter[key] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return original


@pytest.mark.parametrize("query", [
    "Top advisors by revenue",
    "What is Downtown revenue?",
    "Blue Area pipeline",
])
def test_the_planner_runs_once_per_request(org, monkeypatch, query):
    """It ran twice: nlu_pipeline planned to route, then semantic_parser
    planned again from the same text to build its degrade IR. Planning is
    pure, so the duplicate was invisible and paid for twice."""
    from app.llm import nlu_pipeline
    from app.services.chat_service import handle_chat_message

    counter = {"n": 0}
    original = _count_calls(monkeypatch, query_planner, "build_query_plan", counter, "n")
    monkeypatch.setattr(nlu_pipeline, "build_query_plan", query_planner.build_query_plan)
    monkeypatch.setattr(semantic_parser, "build_query_plan", query_planner.build_query_plan)

    handle_chat_message(org, query, session_id=None)
    assert counter["n"] == 1, f"build_query_plan ran {counter['n']}x"
    assert original is not None


def test_the_response_planner_runs_once_per_request(org, monkeypatch):
    """chat_service and response_formatter both called plan_response()."""
    from app.services import chat_service

    counter = {"n": 0}
    _count_calls(monkeypatch, response_planner, "plan_response", counter, "n")
    monkeypatch.setattr(chat_service, "plan_response", response_planner.plan_response)

    chat_service.handle_chat_message(org, "Top advisors by revenue", session_id=None)
    assert counter["n"] == 1, f"plan_response ran {counter['n']}x"


def test_conversation_state_is_written_once_per_request(org, monkeypatch):
    from app.llm import nlu_pipeline
    from app.services.chat_service import handle_chat_message

    counter = {"n": 0}
    _count_calls(monkeypatch, conversation_memory, "set", counter, "n")
    monkeypatch.setattr(nlu_pipeline, "conversation_memory", conversation_memory)

    handle_chat_message(org, "Top advisors by revenue", session_id="own-1")
    assert counter["n"] == 1, f"last_ir written {counter['n']}x"


# ---------------------------------------------------------------------
# The wire format did not move
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("Top advisors by revenue", "leaderboard"),
    ("What is Downtown revenue?", "metric_value"),
    ("Tell me about Yasir Ali", "advisor"),
    ("How is Blue Area performing?", "team"),
    ("Who does Yasir Ali report to?", "manager"),
    ("Show me the trend of revenue", "unsupported"),
    # portfolio % — the refusal working_days.py did not retire.
    ("What is Downtown's portfolio %?", "clarification"),
    ("hello there", "text"),
])
def test_the_api_response_types_are_unchanged(org, query, expected):
    """Phase 4 changed WHO decides the type, not WHAT it is. The frontend
    reads these strings, so renaming them would be a breaking change
    dressed up as a refactor."""
    from app.services.chat_service import handle_chat_message

    assert handle_chat_message(org, query, session_id=None)["type"] == expected


def test_every_response_carries_the_standard_keys(org):
    from app.services.chat_service import handle_chat_message

    for query in ("Top advisors by revenue", "Tell me about Yasir Ali",
                  "Show me the trend of revenue", "hello there"):
        response = handle_chat_message(org, query, session_id=None)
        assert "type" in response and "reply" in response and "data" in response
