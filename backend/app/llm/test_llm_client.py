"""llm_client.py unit tests — ollama.Client is monkeypatched throughout,
no daemon or network involved. Verifies the fail-soft contract (any
exception -> None) and that each function wires the right Ollama
parameters (format/think/model), not real model output quality."""

from types import SimpleNamespace

from app.llm import llm_client
from app.llm.llm_client import call_llm_json, call_llm_structured, embed_texts


class _FakeChatClient:
    def __init__(self, content, raise_error=None):
        self._content = content
        self._raise_error = raise_error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_error:
            raise self._raise_error
        return SimpleNamespace(message=SimpleNamespace(content=self._content))


class _FakeEmbedClient:
    def __init__(self, vectors, raise_error=None):
        self._vectors = vectors
        self._raise_error = raise_error
        self.calls = []

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_error:
            raise self._raise_error
        return SimpleNamespace(embeddings=self._vectors)


# ---- call_llm_structured ----

def test_structured_call_returns_parsed_json(monkeypatch):
    fake = _FakeChatClient('{"intent": "leaderboard"}')
    monkeypatch.setattr(llm_client, "_client", fake)
    result = call_llm_structured("prompt", {"type": "object"}, "query_ir")
    assert result == {"intent": "leaderboard"}
    assert fake.calls[0]["format"] == {"type": "object"}
    assert fake.calls[0]["think"] is False
    assert fake.calls[0]["model"] == llm_client.settings.ollama_model


def test_structured_call_fails_soft_on_malformed_json(monkeypatch):
    fake = _FakeChatClient("not json")
    monkeypatch.setattr(llm_client, "_client", fake)
    assert call_llm_structured("prompt", {}, "query_ir") is None


def test_structured_call_fails_soft_on_connection_error(monkeypatch):
    fake = _FakeChatClient(None, raise_error=ConnectionError("daemon down"))
    monkeypatch.setattr(llm_client, "_client", fake)
    assert call_llm_structured("prompt", {}, "query_ir") is None


# ---- call_llm_json ----

def test_json_call_uses_loose_json_format(monkeypatch):
    fake = _FakeChatClient('{"summary": "ok"}')
    monkeypatch.setattr(llm_client, "_client", fake)
    result = call_llm_json("prompt")
    assert result == {"summary": "ok"}
    assert fake.calls[0]["format"] == "json"


def test_json_call_fails_soft_on_error(monkeypatch):
    fake = _FakeChatClient(None, raise_error=RuntimeError("model not pulled"))
    monkeypatch.setattr(llm_client, "_client", fake)
    assert call_llm_json("prompt") is None


# ---- embed_texts ----

def test_embed_texts_returns_vectors(monkeypatch):
    fake = _FakeEmbedClient([[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr(llm_client, "_client", fake)
    result = embed_texts(["a", "b"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert fake.calls[0]["model"] == llm_client.settings.ollama_embedding_model


def test_embed_texts_empty_input_short_circuits_without_calling_client(monkeypatch):
    fake = _FakeEmbedClient([])
    monkeypatch.setattr(llm_client, "_client", fake)
    assert embed_texts([]) == []
    assert fake.calls == []


def test_embed_texts_fails_soft_on_error(monkeypatch):
    fake = _FakeEmbedClient(None, raise_error=RuntimeError("embedding model not pulled"))
    monkeypatch.setattr(llm_client, "_client", fake)
    assert embed_texts(["a"]) is None
