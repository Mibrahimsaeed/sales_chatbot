"""Startup must not wait on the network.

THE BUG. All of the optional initialisation — an LLM ping, an embeddings
ping, and an entity-index build over ~930 gazetteer strings — ran before
the lifespan's `yield`. uvicorn prints "Application startup complete."
only when the lifespan reaches that yield, so the server sat at "Waiting
for application startup." until every one of them returned. Measured
here, the LLM probe alone took 123.0s — and SUCCEEDED, so this was never
an error path, just a slow one nothing should have been waiting for. It
is also unpredictable: three fresh processes gave 123s, 126s and 2.1s, so
no timeout could have distinguished "slow today" from "broken".

They were also synchronous calls inside an `async` function, which means
they blocked the event loop: even had readiness been signalled, the
process could not have answered /health.

None of it is a dependency. call_llm_json never raises, the probes are
diagnostics, and entity_linker.build_index is already called lazily by
semantic_candidates with its own TTL.
"""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture()
def slow_warmup(monkeypatch):
    """Stands in for the real thing: a warm-up step that takes far longer
    than any acceptable startup."""
    state = {"started": threading.Event(), "finished": threading.Event()}

    def slow():
        state["started"].set()
        time.sleep(3)
        state["finished"].set()

    monkeypatch.setattr(main, "_probe_llm", slow)
    monkeypatch.setattr(main, "_warm_entity_index", lambda: None)
    return state


def test_the_app_becomes_ready_without_waiting_for_the_warm_up(slow_warmup):
    """The requirement, stated directly: readiness must not be behind a
    three-second (or two-minute) network call."""
    started_at = time.perf_counter()

    with TestClient(main.app):
        ready_after = time.perf_counter() - started_at

    assert ready_after < 1.5, (
        f"startup waited {ready_after:.1f}s on optional initialisation")


def test_requests_are_served_while_the_warm_up_is_still_running(slow_warmup):
    """Readiness that cannot serve a request is not readiness. This also
    pins that the warm-up is off the event loop — a blocking call on the
    loop would make this request wait for it."""
    with TestClient(main.app) as client:
        assert slow_warmup["started"].wait(2), "the warm-up did not start"
        assert not slow_warmup["finished"].is_set(), "it should still be running"

        began = time.perf_counter()
        response = client.get("/health")
        took = time.perf_counter() - began

    assert response.status_code == 200
    assert took < 1.0, f"/health blocked for {took:.1f}s behind the warm-up"


def test_the_warm_up_still_runs(slow_warmup):
    """The work is deferred, not dropped — the LLM and embedding layers
    are still initialised, just not on the critical path."""
    with TestClient(main.app):
        assert slow_warmup["started"].wait(2)


def test_a_failing_warm_up_does_not_break_startup(monkeypatch):
    """Every step is best-effort. A broken probe must cost diagnostics,
    never availability."""
    def boom():
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(main, "_probe_llm", boom)
    monkeypatch.setattr(main, "_warm_entity_index", boom)

    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200


def test_shutdown_does_not_wait_for_a_slow_warm_up(slow_warmup):
    """Cancellation propagates, so stopping the server during a probe
    returns immediately instead of blocking on its timeout."""
    with TestClient(main.app):
        assert slow_warmup["started"].wait(2)
        began = time.perf_counter()
    took = time.perf_counter() - began

    assert took < 1.5, f"shutdown waited {took:.1f}s for the warm-up"


def test_authentication_is_untouched():
    """The refactor moved initialisation; it must not have moved the
    security boundary."""
    with TestClient(main.app) as client:
        assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
