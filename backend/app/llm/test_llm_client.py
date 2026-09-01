"""llm_client.py unit tests — the provider is monkeypatched throughout, so
no real network call and no API key is involved.

Verifies the FAIL-SOFT CONTRACT (any exception -> None) and that each
function wires the right provider parameters, not real model output
quality. Those contracts are unchanged across both provider migrations;
only the call shape has moved.

Two seams are used deliberately:

  `_chat`    the provider boundary every inference call goes through.
             Tests patch THIS rather than the vendor client's own method,
             because an earlier form reached into
             `_client.chat.completions.create` — four SDK details, none
             of them this project's — and all four broke on a provider
             change, erroring 146 tests that had nothing to do with it.

  `_openai`  patched only where a test asserts on the raw client call
             (embeddings), which has no separate boundary of its own.
"""

from types import SimpleNamespace

import pytest

from app.llm import llm_client
from app.llm.llm_client import call_llm_json, call_llm_structured, create_embeddings


def _fake_chat(content, raise_error=None):
    """Stands in for llm_client._chat.

    Returns the provider's chat shape — `response.choices[0].message
    .content` — and records every call's kwargs on `.calls` for
    assertions.
    """
    calls = []

    def _chat(*, messages, fmt, purpose="unknown"):
        calls.append({"messages": messages, "fmt": fmt, "purpose": purpose})
        if raise_error:
            raise raise_error
        return _provider_response(content)

    _chat.calls = calls
    return _chat


def _provider_response(content='{"ok": true}', **usage):
    """The chat-completions shape: choices[0].message.content, plus the
    `usage` block the telemetry line reads its token counts from."""
    fields = dict(prompt_tokens=9363, completion_tokens=120, total_tokens=9483)
    fields.update(usage)
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, refusal=None))],
        usage=SimpleNamespace(**fields),
    )


def _fake_embed(vectors, raise_error=None):
    """Stands in for the client's `embeddings.create(model=, input=)`."""
    calls = []

    def create(*, model, input):
        calls.append({"model": model, "input": input})
        if raise_error:
            raise raise_error
        return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    client.calls = calls
    return client


# ---- call_llm_structured ----

def test_structured_call_returns_parsed_json(monkeypatch):
    fake = _fake_chat('{"intent": "leaderboard"}')
    monkeypatch.setattr(llm_client, "_chat", fake)

    result = call_llm_structured("prompt", {"type": "object"}, "query_ir")

    assert result == {"intent": "leaderboard"}
    # The SCHEMA itself is passed as Ollama's `format`, which constrains
    # decoding — the equivalent of OpenAI's strict json_schema block.
    assert fake.calls[0]["fmt"] == {"type": "object"}
    assert fake.calls[0]["messages"][0]["content"] == "prompt"


def test_the_boundary_wires_the_configured_model(monkeypatch):
    """`_chat` is the one place the model is named, so this is where that
    wiring is checked."""
    seen = _capture_chat(monkeypatch)

    llm_client._chat(messages=[{"role": "user", "content": "x"}], fmt="json")

    assert seen["model"] == llm_client.settings.openai_model
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["temperature"] == 0.0


def test_structured_call_fails_soft_on_malformed_json(monkeypatch):
    monkeypatch.setattr(llm_client, "_chat", _fake_chat("not json at all"))
    assert call_llm_structured("prompt", {"type": "object"}, "query_ir") is None


def test_structured_call_fails_soft_on_connection_error(monkeypatch):
    monkeypatch.setattr(
        llm_client, "_chat", _fake_chat(None, raise_error=RuntimeError("provider down")))
    assert call_llm_structured("prompt", {"type": "object"}, "query_ir") is None


def test_structured_call_fails_soft_on_empty_response(monkeypatch):
    monkeypatch.setattr(llm_client, "_chat", _fake_chat(""))
    assert call_llm_structured("prompt", {"type": "object"}, "query_ir") is None


def test_structured_call_rejects_a_non_object(monkeypatch):
    """A bare list is valid JSON but not a QueryIR."""
    monkeypatch.setattr(llm_client, "_chat", _fake_chat('[1, 2, 3]'))
    assert call_llm_structured("prompt", {"type": "object"}, "query_ir") is None


# ---- call_llm_json ----

def test_json_call_uses_loose_json_object_format(monkeypatch):
    fake = _fake_chat('{"summary": "ok"}')
    monkeypatch.setattr(llm_client, "_chat", fake)

    assert call_llm_json("prompt") == {"summary": "ok"}
    # No schema — free-form object output, which is all the narrative
    # polish needs.
    assert fake.calls[0]["fmt"] == "json"


def test_json_call_fails_soft_on_error(monkeypatch):
    monkeypatch.setattr(
        llm_client, "_chat", _fake_chat(None, raise_error=RuntimeError("provider down")))
    assert call_llm_json("prompt") is None


# ---- create_embeddings ----

def test_create_embeddings_returns_vectors(monkeypatch):
    fake = _fake_embed([[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr(llm_client, "_openai", lambda: fake)
    monkeypatch.setattr(llm_client.settings, "openai_embedding_model", "text-embedding-3-small")

    assert create_embeddings(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert fake.calls[0]["model"] == "text-embedding-3-small"
    assert fake.calls[0]["input"] == ["a", "b"]


def test_create_embeddings_empty_input_short_circuits_without_calling_provider(monkeypatch):
    fake = _fake_embed([[0.1]])
    monkeypatch.setattr(llm_client, "_openai", lambda: fake)
    monkeypatch.setattr(llm_client.settings, "openai_embedding_model", "text-embedding-3-small")

    assert create_embeddings([]) == []
    assert fake.calls == []


def test_create_embeddings_raises_so_the_policy_layer_can_classify(monkeypatch):
    """embeddings.py owns availability policy: it treats an exception as
    the signal to disable the tier for the process."""
    monkeypatch.setattr(
        llm_client, "_openai", lambda: _fake_embed([], raise_error=RuntimeError("boom")))
    monkeypatch.setattr(llm_client.settings, "openai_embedding_model", "text-embedding-3-small")

    with pytest.raises(RuntimeError):
        create_embeddings(["a"])


def test_create_embeddings_refuses_when_no_model_is_configured(monkeypatch):
    """The migration's actual breakage: embeddings had no working model
    but were still called on every query. Now it says so, in a type
    embeddings.classify_error() maps to `not_configured`."""
    monkeypatch.setattr(llm_client.settings, "openai_embedding_model", "")

    with pytest.raises(llm_client.EmbeddingsNotConfigured):
        create_embeddings(["a"])


def test_the_not_configured_error_is_classified_distinctly():
    from app.llm import embeddings

    reason = embeddings.classify_error(
        llm_client.EmbeddingsNotConfigured("no model"))
    assert reason == embeddings.REASON_NOT_CONFIGURED


# ---- latency instrumentation (task 1) ----
#
# The audit could not say whether latency came from prefill, generation,
# model loading or the network, because nothing timed the call. These pin
# that the boundary now reports it — and that the numbers are the
# PROVIDER'S, never inferred from string lengths.

def _fake_client(response=None, raise_error=None):
    # **kwargs, not a fixed signature: the boundary's parameter set is
    # config-driven and can grow, and a fake that pins today's exact set
    # breaks on every addition without testing anything.
    def create(**kwargs):
        if raise_error:
            raise raise_error
        return response
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _llm_call_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("llm_call ")]


def _fields(line):
    return dict(part.split("=", 1) for part in line.removeprefix("llm_call ").split(" "))


def test_a_successful_call_logs_provider_metadata(monkeypatch, caplog):
    monkeypatch.setattr(llm_client, "_openai", lambda: _fake_client(_provider_response()))

    with caplog.at_level("INFO", logger="llm.client"):
        llm_client._chat(messages=[{"role": "user", "content": "x"}],
                         fmt="json", purpose="structured:query_ir")

    lines = _llm_call_lines(caplog)
    assert len(lines) == 1
    f = _fields(lines[0])

    assert f["purpose"] == "structured:query_ir"
    assert f["success"] == "True"
    assert f["model"] == llm_client.settings.openai_model
    assert f["provider"] == llm_client.PROVIDER
    # Token counts are the provider's own, not derived from characters.
    assert f["prompt_tokens"] == "9363"
    assert f["output_tokens"] == "120"
    assert f["total_tokens"] == "9483"
    # Measured at the boundary, so it is present whatever the provider
    # reports about itself.
    assert float(f["duration_ms"]) >= 0


def test_a_failed_call_is_timed_and_logged_too(monkeypatch, caplog):
    """A 60-second timeout is the most expensive thing this function can
    do; leaving it unlogged would hide it from the measurement."""
    monkeypatch.setattr(
        llm_client, "_openai",
        lambda: _fake_client(raise_error=RuntimeError("provider down")))

    with caplog.at_level("INFO", logger="llm.client"):
        with pytest.raises(RuntimeError):
            llm_client._chat(messages=[], fmt="json", purpose="narrative")

    f = _fields(_llm_call_lines(caplog)[0])
    assert f["success"] == "False"
    assert f["error"] == "RuntimeError"
    assert f["purpose"] == "narrative"
    assert float(f["duration_ms"]) >= 0
    # Nothing invented for counts the provider never sent.
    assert f["prompt_tokens"] == "None"
    assert f["output_tokens"] == "None"


def test_absent_usage_is_recorded_as_none_not_zero(monkeypatch, caplog):
    """A response without a usage block must not read as "zero tokens"."""
    monkeypatch.setattr(llm_client, "_openai", lambda: _fake_client(
        SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="{}", refusal=None))], usage=None)))

    with caplog.at_level("INFO", logger="llm.client"):
        llm_client._chat(messages=[], fmt="json", purpose="narrative")

    f = _fields(_llm_call_lines(caplog)[0])
    assert f["prompt_tokens"] == "None"
    assert f["output_tokens"] == "None"
    assert f["success"] == "True"


def test_the_real_call_sites_label_their_purpose(monkeypatch, caplog):
    """Both purposes must be distinguishable in the log, since the whole
    question is which of the two calls costs what."""
    monkeypatch.setattr(llm_client, "_openai", lambda: _fake_client(_provider_response()))

    with caplog.at_level("INFO", logger="llm.client"):
        call_llm_structured("prompt", {"type": "object"}, "query_ir")
        call_llm_json("prompt")

    purposes = [_fields(line)["purpose"] for line in _llm_call_lines(caplog)]
    assert purposes == ["structured:query_ir", "narrative"]


# ---- inference configuration ----
#
# The point of these is that every knob is EXPLICIT and comes from config
# rather than a literal, so a change can be measured rather than guessed
# at. The Ollama-only knobs (num_ctx, num_predict, keep_alive, think) went
# with the Ollama transport; what remains is what this provider accepts.

def _capture_chat(monkeypatch):
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return _provider_response("{}")

    monkeypatch.setattr(llm_client, "_openai", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    return seen


def test_every_configured_option_is_sent(monkeypatch):
    seen = _capture_chat(monkeypatch)
    llm_client._chat(messages=[], fmt="json")

    assert seen["model"] == llm_client.settings.openai_model
    assert seen["temperature"] == llm_client.settings.openai_temperature
    assert seen["max_completion_tokens"] == llm_client.settings.openai_max_output_tokens


def test_the_values_come_from_config_not_literals(monkeypatch):
    """Environment-configurable is the requirement; a literal in the
    client would defeat it."""
    monkeypatch.setattr(llm_client.settings, "openai_model", "some-other-model")
    monkeypatch.setattr(llm_client.settings, "openai_temperature", 0.7)
    monkeypatch.setattr(llm_client.settings, "openai_max_output_tokens", 256)

    seen = _capture_chat(monkeypatch)
    llm_client._chat(messages=[], fmt="json")

    assert seen["model"] == "some-other-model"
    assert seen["temperature"] == 0.7
    assert seen["max_completion_tokens"] == 256


def test_temperature_defaults_to_deterministic():
    """Both call sites are deterministic transforms — text to QueryIR
    under a schema, and a copy-edit rejected if it introduces a number.
    Sampling adds variance to a task with one right answer and makes a
    wrong parse unreproducible."""
    assert llm_client.settings.openai_temperature == 0.0


def test_the_output_ceiling_clears_the_largest_real_output():
    """A maximal valid QueryIR serialises to ~347 tokens and the
    narrative reply is 60-120. The ceiling is a runaway guard, so it must
    not be tight enough to truncate a legitimate answer."""
    assert llm_client.settings.openai_max_output_tokens >= 512


def test_the_options_reach_both_real_call_sites(monkeypatch):
    """One boundary, so neither caller needs to know about any of this."""
    seen = _capture_chat(monkeypatch)

    call_llm_structured("prompt", {"type": "object"}, "query_ir")
    assert seen["max_completion_tokens"] == llm_client.settings.openai_max_output_tokens
    seen.clear()

    call_llm_json("prompt")
    assert seen["max_completion_tokens"] == llm_client.settings.openai_max_output_tokens
