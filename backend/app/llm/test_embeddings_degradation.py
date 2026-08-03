"""
Graceful degradation when embeddings are unavailable.

Embeddings are an OPTIONAL widening tier: semantic entity linking and
semantic metric retrieval both sit behind exact and fuzzy matching, and
only run when the deterministic tiers found nothing. So an unusable
provider must cost a little recall on paraphrased queries and NOTHING
else — the backend keeps starting, keeps serving, and keeps answering via
exact/fuzzy/WID resolution.

What these tests pin down:
  - every expected provider failure is caught and classified
  - the subsystem gives up ONCE and then stops calling (no retry loop)
  - the actionable WARNING is logged once, not per call
  - the chatbot answers real queries with embeddings disabled
"""

import logging

import openai
import pytest

from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, embeddings, entity_extractor
from app.llm import entity_linker, llm_client, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture(autouse=True)
def _reset_embeddings(monkeypatch):
    embeddings._reset_for_tests()
    # the subsystem is only "enabled" if a feature wants it
    monkeypatch.setattr(embeddings.settings, "entity_linking_enabled", True)
    monkeypatch.setattr(embeddings.settings, "semantic_retrieval_enabled", True)
    yield
    embeddings._reset_for_tests()


def _openai_error(cls, message="boom", **kw):
    """Construct a real openai exception. Their constructors differ, so
    fall back to __new__ when a signature doesn't cooperate — the point
    is the TYPE, which is what classification keys on."""
    try:
        if cls is openai.APIConnectionError:
            return cls(request=None)
        if cls is openai.APITimeoutError:
            return cls(request=None)
        response = type("R", (), {"status_code": 429, "headers": {}, "request": None})()
        return cls(message=message, response=response, body=kw.get("body"))
    except Exception:
        exc = cls.__new__(cls)
        Exception.__init__(exc, message)
        if "body" in kw:
            exc.body = kw["body"]
        return exc


def _raise(exc):
    def _fn(texts):
        raise exc
    return _fn


# =====================================================================
# 4. Expected API failures are caught and classified
# =====================================================================

def test_quota_exceeded_is_classified_and_disables(monkeypatch):
    exc = _openai_error(openai.RateLimitError, "exceeded quota",
                        body={"error": {"code": "insufficient_quota"}})
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(exc))

    assert embeddings.embed_texts(["x"]) is None
    status = embeddings.status()
    assert status.ready is False
    assert status.reason == embeddings.REASON_QUOTA


def test_rate_limit_is_distinguished_from_exhausted_quota(monkeypatch):
    """Worth telling apart: a rate limit is transient and worth a
    restart-retry; an exhausted quota needs a human to add credit."""
    exc = _openai_error(openai.RateLimitError, "slow down", body={"error": {"code": "rate_limit_exceeded"}})
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(exc))

    embeddings.embed_texts(["x"])
    assert embeddings.status().reason == embeddings.REASON_RATE_LIMIT


def test_invalid_api_key_is_classified(monkeypatch):
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(_openai_error(openai.AuthenticationError)))
    assert embeddings.embed_texts(["x"]) is None
    assert embeddings.status().reason == embeddings.REASON_AUTH


def test_network_failure_is_classified(monkeypatch):
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(_openai_error(openai.APIConnectionError)))
    assert embeddings.embed_texts(["x"]) is None
    assert embeddings.status().reason == embeddings.REASON_CONNECTION


def test_timeout_is_classified(monkeypatch):
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(_openai_error(openai.APITimeoutError)))
    assert embeddings.embed_texts(["x"]) is None
    assert embeddings.status().reason == embeddings.REASON_TIMEOUT


def test_generic_api_error_is_classified(monkeypatch):
    exc = openai.APIError("server exploded", request=None, body=None)
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(exc))
    assert embeddings.embed_texts(["x"]) is None
    assert embeddings.status().reason == embeddings.REASON_API_ERROR


def test_unexpected_error_never_escapes(monkeypatch):
    """Anything not in the provider's hierarchy must still be absorbed —
    an embedding failure must never reach the chat endpoint."""
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(ValueError("something odd")))
    assert embeddings.embed_texts(["x"]) is None
    assert embeddings.status().reason == embeddings.REASON_UNEXPECTED


def test_empty_provider_response_is_treated_as_unavailable(monkeypatch):
    monkeypatch.setattr(llm_client, "create_embeddings", lambda texts: [])
    assert embeddings.embed_texts(["x"]) is None
    assert embeddings.status().reason == embeddings.REASON_EMPTY


# =====================================================================
# 2. No retry loop — give up once, then stop calling
# =====================================================================

def test_provider_is_not_called_again_after_giving_up(monkeypatch):
    calls = []

    def _fail(texts):
        calls.append(texts)
        raise _openai_error(openai.AuthenticationError)

    monkeypatch.setattr(llm_client, "create_embeddings", _fail)

    for _ in range(20):
        assert embeddings.embed_texts(["x"]) is None

    assert len(calls) == 1, "a broken key must cost ONE failed request per process, not one per query"


def test_the_warning_is_logged_once_not_per_call(monkeypatch, caplog):
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(_openai_error(openai.AuthenticationError)))

    with caplog.at_level(logging.WARNING, logger="llm.embeddings"):
        for _ in range(10):
            embeddings.embed_texts(["x"])

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Embeddings unavailable" in warnings[0].getMessage()
    assert "invalid_api_key" in warnings[0].getMessage()


def test_no_traceback_at_warning_level(monkeypatch, caplog):
    """The reported symptom was a ~400-line traceback every minute. Detail
    belongs at DEBUG; the WARNING carries the actionable part."""
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(_openai_error(openai.AuthenticationError)))

    with caplog.at_level(logging.WARNING, logger="llm.embeddings"):
        embeddings.embed_texts(["x"])

    assert all(r.exc_info is None for r in caplog.records if r.levelno == logging.WARNING)


def test_rebuild_is_the_explicit_way_back(monkeypatch):
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(_openai_error(openai.AuthenticationError)))
    embeddings.embed_texts(["x"])
    assert embeddings.status().ready is False

    monkeypatch.setattr(llm_client, "create_embeddings", lambda texts: [[1.0, 0.0] for _ in texts])
    status = embeddings.rebuild()
    assert status.ready is True
    assert status.reason is None
    assert embeddings.embed_texts(["x"]) == [[1.0, 0.0]]


# =====================================================================
# 1/5. Startup + health status
# =====================================================================

def test_probe_never_raises_and_reports_status(monkeypatch):
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(_openai_error(openai.AuthenticationError)))
    status = embeddings.probe()          # must not raise
    assert status.enabled is True
    assert status.ready is False
    assert status.provider == "openai"
    assert status.reason == embeddings.REASON_AUTH


def test_status_shape_matches_the_health_contract(monkeypatch):
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(_openai_error(openai.AuthenticationError)))
    embeddings.probe()
    payload = embeddings.status().to_dict()
    assert set(payload) == {"enabled", "ready", "provider", "reason", "checked_at"}


def test_disabled_by_config_reports_cleanly(monkeypatch):
    monkeypatch.setattr(embeddings.settings, "entity_linking_enabled", False)
    monkeypatch.setattr(embeddings.settings, "semantic_retrieval_enabled", False)
    called = []
    monkeypatch.setattr(llm_client, "create_embeddings", lambda texts: called.append(texts))

    status = embeddings.probe()
    assert status.enabled is False and status.ready is False
    assert status.reason == embeddings.REASON_DISABLED
    assert called == [], "config-disabled must not touch the provider at all"


def test_health_endpoint_exposes_embedding_status(client, monkeypatch):
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(_openai_error(openai.AuthenticationError)))
    embeddings.probe()

    body = client.get("/health").json()
    assert body["status"] == "ok", "an optional tier being down must not mark the service unhealthy"
    assert body["embeddings"]["ready"] is False
    assert body["embeddings"]["provider"] == "openai"
    assert body["embeddings"]["reason"] == "invalid_api_key"


def test_rebuild_endpoint(client, monkeypatch):
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(_openai_error(openai.AuthenticationError)))
    embeddings.probe()
    assert client.get("/health").json()["embeddings"]["ready"] is False

    monkeypatch.setattr(llm_client, "create_embeddings", lambda texts: [[1.0, 0.0] for _ in texts])
    body = client.post("/health/embeddings/rebuild").json()
    assert body["embeddings"]["ready"] is True


# =====================================================================
# 3/7. The chatbot keeps working with embeddings unavailable
# =====================================================================

@pytest.fixture()
def degraded_db(db_session, monkeypatch):
    """A realistic dataset with embeddings hard-down (quota exhausted)."""
    def advisor(wid, name, team, **kw):
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana", **kw))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=10, mtd_followup_connect=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000, cleared=500))

    advisor(1, "Reportee One", "Blue Area", bm="Adeel Dogar", rm="Adeel Dogar")
    advisor(2, "Reportee Two", "Downtown", bm="Adeel Dogar", rm="Adeel Dogar")
    advisor(10, "Yasir Ali", "North/KPK")
    advisor(11, "Yasir Ali", "Downtown")
    advisor(30, "Ali Murtaza", "North/KPK", bm="Musab Sial", rm="Musab Sial")
    advisor(31, "Murtaza Reportee", "North/KPK", bm="Ali Murtaza", rm="Ali Murtaza")
    db_session.commit()

    quota = _openai_error(openai.RateLimitError, "quota", body={"error": {"code": "insufficient_quota"}})
    monkeypatch.setattr(llm_client, "create_embeddings", _raise(quota))
    embeddings.probe()
    assert embeddings.status().ready is False   # precondition: genuinely degraded

    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


def test_hierarchy_query_works_without_embeddings(degraded_db):
    response = handle_chat_message(degraded_db, "Show Adeel Dogar's team", session_id=None)
    assert response["type"] == "breakdown"
    assert response["data"]["advisors"] == 2


def test_reverse_hierarchy_works_without_embeddings(degraded_db):
    response = handle_chat_message(degraded_db, "Who is BM of Ali Murtaza?", session_id=None)
    assert response["type"] == "manager"
    assert response["data"]["manager"] == "Musab Sial"


def test_who_reports_to_works_without_embeddings(degraded_db):
    response = handle_chat_message(degraded_db, "Who reports to Adeel Dogar?", session_id=None)
    assert response["type"] == "breakdown"


def test_duplicate_name_clarification_works_without_embeddings(degraded_db):
    response = handle_chat_message(degraded_db, "Tell me about Yasir Ali", session_id=None)
    assert response["type"] == "clarification"
    assert len(response["options"]) == 2


def test_leaderboard_works_without_embeddings(degraded_db):
    response = handle_chat_message(degraded_db, "Top 5 advisors by connects", session_id=None)
    assert response["type"] == "leaderboard"
    assert len(response["data"]) > 0


def test_serving_many_queries_never_retries_the_provider(degraded_db, monkeypatch):
    """The integration form of the no-retry-loop guarantee: a whole
    conversation costs zero further embedding calls."""
    calls = []
    monkeypatch.setattr(
        llm_client, "create_embeddings",
        lambda texts: calls.append(texts) or (_ for _ in ()).throw(RuntimeError("must not be called")),
    )

    for text in ["Show Adeel Dogar's team", "Tell me about Yasir Ali",
                 "Top 5 advisors by connects", "Who is BM of Ali Murtaza?"]:
        handle_chat_message(degraded_db, text, session_id=None)

    assert calls == []
