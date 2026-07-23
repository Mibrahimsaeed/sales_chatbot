"""Unit tests for semantic_retrieval.py — embed_texts is monkeypatched
throughout, no API key or network involved. This module CANNOT be
verified against the live OpenAI API right now (429 insufficient_quota at
the time it was written); these tests lock in the fail-soft contract and
the matching logic against a deterministic fake embedding space."""

import pytest

from app.llm import semantic_retrieval
from app.llm.semantic_retrieval import retrieve_metric, _cosine


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    semantic_retrieval._reset_cache_for_tests()
    monkeypatch.setattr(semantic_retrieval.settings, "semantic_retrieval_enabled", True)
    yield
    semantic_retrieval._reset_cache_for_tests()


def _fake_embed_factory(target_phrase: str):
    """A tiny fake embedding space: the target phrase (and anything meant
    to match it) gets vector [1, 0]; everything else gets [0, 1] — so
    cosine similarity is 1.0 for a real match and 0.0 otherwise, with no
    real embedding model involved."""
    def _embed(texts):
        return [[1.0, 0.0] if t == target_phrase or t == "query" else [0.0, 1.0] for t in texts]
    return _embed


def test_disabled_by_flag_returns_none_without_calling_embed(monkeypatch):
    monkeypatch.setattr(semantic_retrieval.settings, "semantic_retrieval_enabled", False)
    called = []
    monkeypatch.setattr(semantic_retrieval, "embed_texts", lambda texts: called.append(texts))
    assert retrieve_metric("anything") is None
    assert called == []


def test_embed_failure_returns_none(monkeypatch):
    monkeypatch.setattr(semantic_retrieval, "embed_texts", lambda texts: None)
    assert retrieve_metric("who is crushing it") is None


def test_embed_failure_is_cached_and_not_retried(monkeypatch):
    calls = []

    def _embed(texts):
        calls.append(texts)
        return None

    monkeypatch.setattr(semantic_retrieval, "embed_texts", _embed)
    assert retrieve_metric("first") is None
    assert retrieve_metric("second") is None
    # only one attempt to embed the exemplar corpus — the second call
    # short-circuits on the cached _unavailable flag instead of retrying
    assert len(calls) == 1


def test_close_match_above_floor_returns_metric_key(monkeypatch):
    # "top performer" is a real achievement_pct synonym in the ontology
    monkeypatch.setattr(semantic_retrieval, "embed_texts", _fake_embed_factory("top performer"))

    def _embed_query(texts):
        # the query text always gets the "matches" vector in this fake space
        return [[1.0, 0.0] for _ in texts]

    # first call embeds the exemplar corpus (uses the phrase-matching fake),
    # second call embeds the query — patch per-call via a stateful wrapper
    calls = {"n": 0}
    corpus_fn = _fake_embed_factory("top performer")

    def _embed(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            return corpus_fn(texts)
        return _embed_query(texts)

    monkeypatch.setattr(semantic_retrieval, "embed_texts", _embed)
    assert retrieve_metric("who is the star this month") == "achievement_pct"


def test_no_close_match_below_floor_returns_none(monkeypatch):
    # exemplar corpus gets one vector, the query gets an orthogonal one —
    # cosine similarity is 0.0 against every exemplar, well below the floor
    calls = {"n": 0}

    def _embed(texts):
        calls["n"] += 1
        vector = [0.0, 1.0] if calls["n"] == 1 else [1.0, 0.0]
        return [vector for _ in texts]

    monkeypatch.setattr(semantic_retrieval, "embed_texts", _embed)
    assert retrieve_metric("completely unrelated gibberish") is None


def test_cosine_similarity_basic_cases():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
