"""llm_client.py unit tests — the provider is monkeypatched throughout, so
no real network call and no running Ollama is involved.

Verifies the FAIL-SOFT CONTRACT (any exception -> None) and that each
function wires the right provider parameters, not real model output
quality. The contracts are unchanged from the OpenAI version of this
file; only the call shape moved.

Two seams are used deliberately:

  `_chat`    the provider boundary every inference call goes through.
             Tests patch THIS rather than the vendor client's own method,
             because the previous form reached into
             `_client.chat.completions.create` — four OpenAI SDK details
             — and all four broke on the migration, erroring 146 tests
             that had nothing to do with the provider.

  `_ollama`  patched only where a test asserts on the raw client call
             (embeddings), which has no separate boundary of its own.
"""

from types import SimpleNamespace

import pytest

from app.llm import llm_client
from app.llm.llm_client import call_llm_json, call_llm_structured, create_embeddings


def _fake_chat(content, raise_error=None):
    """Stands in for llm_client._chat.

    Returns the Ollama chat shape — `response.message.content` — and
    records every call's kwargs on `.calls` for assertions.
    """
    calls = []

    def _chat(*, messages, fmt, purpose="unknown"):
        calls.append({"messages": messages, "fmt": fmt, "purpose": purpose})
        if raise_error:
            raise raise_error
        return SimpleNamespace(message=SimpleNamespace(content=content))

    _chat.calls = calls
    return _chat


def _fake_ollama_embed(vectors, raise_error=None):
    """Stands in for the Ollama client's `embed(model=, input=)`."""
    calls = []

    def embed(*, model, input):
        calls.append({"model": model, "input": input})
        if raise_error:
            raise raise_error
        return SimpleNamespace(embeddings=vectors)

    return SimpleNamespace(embed=embed, calls=calls)


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
    seen = {}

    def fake_client_chat(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content="{}"))

    monkeypatch.setattr(llm_client, "_ollama", SimpleNamespace(chat=fake_client_chat))

    llm_client._chat(messages=[{"role": "user", "content": "x"}], fmt="json")

    assert seen["model"] == llm_client.settings.ollama_model
    assert seen["format"] == "json"
    assert seen["options"]["temperature"] == 0.0


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
    fake = _fake_ollama_embed([[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr(llm_client, "_ollama", fake)
    monkeypatch.setattr(llm_client.settings, "ollama_embedding_model", "nomic-embed-text")

    assert create_embeddings(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert fake.calls[0]["model"] == "nomic-embed-text"
    assert fake.calls[0]["input"] == ["a", "b"]


def test_create_embeddings_empty_input_short_circuits_without_calling_provider(monkeypatch):
    fake = _fake_ollama_embed([[0.1]])
    monkeypatch.setattr(llm_client, "_ollama", fake)
    monkeypatch.setattr(llm_client.settings, "ollama_embedding_model", "nomic-embed-text")

    assert create_embeddings([]) == []
    assert fake.calls == []


def test_create_embeddings_raises_so_the_policy_layer_can_classify(monkeypatch):
    """embeddings.py owns availability policy: it treats an exception as
    the signal to disable the tier for the process."""
    monkeypatch.setattr(
        llm_client, "_ollama", _fake_ollama_embed(None, raise_error=RuntimeError("boom")))
    monkeypatch.setattr(llm_client.settings, "ollama_embedding_model", "nomic-embed-text")

    with pytest.raises(RuntimeError):
        create_embeddings(["a"])


def test_create_embeddings_refuses_when_no_model_is_configured(monkeypatch):
    """The migration's actual breakage: embeddings had no working model
    but were still called on every query. Now it says so, in a type
    embeddings.classify_error() maps to `not_configured`."""
    monkeypatch.setattr(llm_client.settings, "ollama_embedding_model", "")

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

def _ollama_response(content='{"ok": true}', **metadata):
    """A ChatResponse-shaped fake carrying Ollama's real metadata field
    names. Durations are NANOSECONDS, as ollama._types documents."""
    fields = dict(
        total_duration=2_000_000_000,
        load_duration=500_000_000,
        prompt_eval_count=9363,
        prompt_eval_duration=1_000_000_000,
        eval_count=120,
        eval_duration=400_000_000,
    )
    fields.update(metadata)
    return SimpleNamespace(message=SimpleNamespace(content=content), **fields)


def _fake_client(response=None, raise_error=None):
    # **kwargs, not a fixed signature: the boundary's parameter set is
    # config-driven and grows (keep_alive, think), and a fake that pins
    # today's exact set breaks on every addition without testing anything.
    def chat(**kwargs):
        if raise_error:
            raise raise_error
        return response
    return SimpleNamespace(chat=chat)


def _llm_call_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("llm_call ")]


def _fields(line):
    return dict(part.split("=", 1) for part in line.removeprefix("llm_call ").split(" "))


def test_a_successful_call_logs_provider_metadata(monkeypatch, caplog):
    monkeypatch.setattr(llm_client, "_ollama", _fake_client(_ollama_response()))

    with caplog.at_level("INFO", logger="llm.client"):
        llm_client._chat(messages=[{"role": "user", "content": "x"}],
                         fmt="json", purpose="structured:query_ir")

    lines = _llm_call_lines(caplog)
    assert len(lines) == 1
    f = _fields(lines[0])

    assert f["purpose"] == "structured:query_ir"
    assert f["success"] == "True"
    assert f["model"] == llm_client.settings.ollama_model
    assert f["provider"] == llm_client.settings.llm_provider
    # Token counts are the provider's own, not derived from characters.
    assert f["prompt_tokens"] == "9363"
    assert f["output_tokens"] == "120"
    # Nanoseconds converted to milliseconds.
    assert f["load_duration_ms"] == "500.0"
    assert f["prompt_eval_duration_ms"] == "1000.0"
    assert f["eval_duration_ms"] == "400.0"
    assert f["total_duration_ms"] == "2000.0"
    # 120 tokens / 0.4s
    assert f["eval_tokens_per_second"] == "300.0"
    assert float(f["duration_ms"]) >= 0


def test_a_failed_call_is_timed_and_logged_too(monkeypatch, caplog):
    """A 60-second timeout is the most expensive thing this function can
    do; leaving it unlogged would hide it from the measurement."""
    monkeypatch.setattr(
        llm_client, "_ollama", _fake_client(raise_error=RuntimeError("provider down")))

    with caplog.at_level("INFO", logger="llm.client"):
        with pytest.raises(RuntimeError):
            llm_client._chat(messages=[], fmt="json", purpose="narrative")

    f = _fields(_llm_call_lines(caplog)[0])
    assert f["success"] == "False"
    assert f["error"] == "RuntimeError"
    assert f["purpose"] == "narrative"
    assert float(f["duration_ms"]) >= 0
    # Nothing invented for metadata the provider never sent.
    assert f["prompt_tokens"] == "None"
    assert f["eval_tokens_per_second"] == "None"


def test_absent_metadata_is_recorded_as_none_not_zero(monkeypatch, caplog):
    """An older Ollama, or a response without timings, must not read as
    "instant"."""
    monkeypatch.setattr(llm_client, "_ollama", _fake_client(
        SimpleNamespace(message=SimpleNamespace(content="{}"))))

    with caplog.at_level("INFO", logger="llm.client"):
        llm_client._chat(messages=[], fmt="json", purpose="narrative")

    f = _fields(_llm_call_lines(caplog)[0])
    assert f["load_duration_ms"] == "None"
    assert f["prompt_tokens"] == "None"
    assert f["success"] == "True"


def test_throughput_is_none_when_it_cannot_be_computed():
    assert llm_client._tokens_per_second(None, 1_000_000_000) is None
    assert llm_client._tokens_per_second(100, None) is None
    # A zero eval_duration must not divide by zero.
    assert llm_client._tokens_per_second(100, 0) is None
    assert llm_client._tokens_per_second(100, 1_000_000_000) == 100.0


def test_nanoseconds_convert_to_milliseconds():
    assert llm_client._ns_to_ms(None) is None
    assert llm_client._ns_to_ms(0) == 0.0
    assert llm_client._ns_to_ms(1_500_000_000) == 1500.0


def test_the_real_call_sites_label_their_purpose(monkeypatch, caplog):
    """Both purposes must be distinguishable in the log, since the whole
    question is which of the two calls costs what."""
    monkeypatch.setattr(llm_client, "_ollama", _fake_client(_ollama_response()))

    with caplog.at_level("INFO", logger="llm.client"):
        call_llm_structured("prompt", {"type": "object"}, "query_ir")
        call_llm_json("prompt")

    purposes = [_fields(line)["purpose"] for line in _llm_call_lines(caplog)]
    assert purposes == ["structured:query_ir", "narrative"]


# ---- inference configuration (task 2) ----
#
# All four options were previously unset, so every call silently inherited
# an Ollama default. These pin that each is now sent, and that each comes
# from config rather than a literal — the point of the change is that the
# configuration is explicit and therefore measurable.

def _capture_chat(monkeypatch):
    seen = {}

    def chat(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content="{}"))

    monkeypatch.setattr(llm_client, "_ollama", SimpleNamespace(chat=chat))
    return seen


def test_all_four_options_are_sent(monkeypatch):
    seen = _capture_chat(monkeypatch)
    llm_client._chat(messages=[], fmt="json")

    assert seen["think"] is False
    assert seen["keep_alive"] == llm_client.settings.ollama_keep_alive
    assert seen["options"]["num_ctx"] == llm_client.settings.ollama_num_ctx
    assert seen["options"]["num_predict"] == llm_client.settings.ollama_num_predict


def test_the_values_come_from_config_not_literals(monkeypatch):
    """Environment-configurable is the requirement; a literal in the
    client would defeat it."""
    monkeypatch.setattr(llm_client.settings, "ollama_keep_alive", "5m")
    monkeypatch.setattr(llm_client.settings, "ollama_num_ctx", 32768)
    monkeypatch.setattr(llm_client.settings, "ollama_num_predict", 256)
    monkeypatch.setattr(llm_client.settings, "ollama_think", True)

    seen = _capture_chat(monkeypatch)
    llm_client._chat(messages=[], fmt="json")

    assert seen["keep_alive"] == "5m"
    assert seen["options"]["num_ctx"] == 32768
    assert seen["options"]["num_predict"] == 256
    assert seen["think"] is True


def test_think_is_disabled_by_default():
    """qwen3 reasons by default. Both calls are constrained
    transformations, so thinking is pure latency on the interactive
    path."""
    assert llm_client.settings.ollama_think is False


def test_think_can_be_omitted_entirely(monkeypatch):
    """None is the escape hatch for a model that rejects the parameter —
    config alone, no code change."""
    monkeypatch.setattr(llm_client.settings, "ollama_think", None)
    seen = _capture_chat(monkeypatch)
    llm_client._chat(messages=[], fmt="json")

    assert "think" not in seen


def test_num_ctx_holds_the_measured_worst_case_prompt():
    """The largest observed prompt is ~9,623 tokens by a chars/4 estimate
    that UNDER-counts dense JSON and name lists. The window must clear
    that plus generation plus growth — and must never be 4096, because
    the user's question sits near the END of the prompt, so a truncating
    window loses the question itself."""
    settings = llm_client.settings
    assert settings.ollama_num_ctx >= 16384
    assert settings.ollama_num_ctx > 9623 + settings.ollama_num_predict


def test_num_predict_clears_the_largest_real_output():
    """A large realistic QueryIR serialises to ~331 tokens; the narrative
    reply is 60-120. The ceiling is a runaway guard, so it must not be
    tight enough to truncate a legitimate answer."""
    assert llm_client.settings.ollama_num_predict >= 512


def test_temperature_is_unchanged(monkeypatch):
    """Task 2 changes configuration, not sampling."""
    seen = _capture_chat(monkeypatch)
    llm_client._chat(messages=[], fmt="json")
    assert seen["options"]["temperature"] == 0.0


def test_the_options_reach_both_real_call_sites(monkeypatch):
    """One boundary, so neither caller needs to know about any of this."""
    seen = _capture_chat(monkeypatch)

    call_llm_structured("prompt", {"type": "object"}, "query_ir")
    assert seen["options"]["num_ctx"] == llm_client.settings.ollama_num_ctx
    seen.clear()

    call_llm_json("prompt")
    assert seen["options"]["num_ctx"] == llm_client.settings.ollama_num_ctx
