"""llm_client.py unit tests — the OpenAI client is monkeypatched throughout,
no real network call involved. Verifies the fail-soft contract (any
exception -> None) and that each function wires the right OpenAI
parameters (response_format/model), not real model output quality."""

from types import SimpleNamespace

from app.llm import llm_client
import pytest

from app.llm.llm_client import call_llm_json, call_llm_structured, create_embeddings


def _fake_chat_client(content, raise_error=None):
    """Mirrors client.chat.completions.create(**kwargs) -> response with
    response.choices[0].message.content, the OpenAI SDK shape llm_client.py
    calls into. `.calls` (attached directly to the returned object, same
    spot the old Ollama-client fakes exposed it) records every kwargs dict
    passed in, for assertions."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if raise_error:
            raise raise_error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), calls=calls)


def _fake_embed_client(vectors, raise_error=None):
    """Mirrors client.embeddings.create(**kwargs) -> response with
    response.data[i].embedding."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if raise_error:
            raise raise_error
        return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])

    return SimpleNamespace(embeddings=SimpleNamespace(create=create), calls=calls)


# ---- call_llm_structured ----

def test_structured_call_returns_parsed_json(monkeypatch):
    fake = _fake_chat_client('{"intent": "leaderboard"}')
    monkeypatch.setattr(llm_client, "_client", fake)
    result = call_llm_structured("prompt", {"type": "object"}, "query_ir")
    assert result == {"intent": "leaderboard"}
    assert fake.calls[0]["model"] == llm_client.settings.openai_model
    response_format = fake.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "query_ir"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == {"type": "object"}


def test_structured_call_fails_soft_on_malformed_json(monkeypatch):
    fake = _fake_chat_client("not json")
    monkeypatch.setattr(llm_client, "_client", fake)
    assert call_llm_structured("prompt", {}, "query_ir") is None


def test_structured_call_fails_soft_on_connection_error(monkeypatch):
    fake = _fake_chat_client(None, raise_error=ConnectionError("network unreachable"))
    monkeypatch.setattr(llm_client, "_client", fake)
    assert call_llm_structured("prompt", {}, "query_ir") is None


# ---- call_llm_json ----

def test_json_call_uses_loose_json_object_format(monkeypatch):
    fake = _fake_chat_client('{"summary": "ok"}')
    monkeypatch.setattr(llm_client, "_client", fake)
    result = call_llm_json("prompt")
    assert result == {"summary": "ok"}
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_json_call_fails_soft_on_error(monkeypatch):
    fake = _fake_chat_client(None, raise_error=RuntimeError("no quota"))
    monkeypatch.setattr(llm_client, "_client", fake)
    assert call_llm_json("prompt") is None


# ---- create_embeddings (RAW — the one function here that does NOT fail soft) ----

def test_create_embeddings_returns_vectors(monkeypatch):
    fake = _fake_embed_client([[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr(llm_client, "_client", fake)
    result = create_embeddings(["a", "b"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert fake.calls[0]["model"] == llm_client.settings.openai_embedding_model


def test_create_embeddings_empty_input_short_circuits_without_calling_client(monkeypatch):
    fake = _fake_embed_client([])
    monkeypatch.setattr(llm_client, "_client", fake)
    assert create_embeddings([]) == []
    assert fake.calls == []


def test_create_embeddings_raises_so_the_policy_layer_can_classify(monkeypatch):
    """Deliberately NOT fail-soft, unlike every other function here:
    app/llm/embeddings.py needs the exception to tell "no quota" from
    "network blip". Swallowing it would collapse both into an
    indistinguishable None, and both would then be retried forever."""
    fake = _fake_embed_client(None, raise_error=RuntimeError("no quota"))
    monkeypatch.setattr(llm_client, "_client", fake)
    with pytest.raises(RuntimeError):
        create_embeddings(["a"])
