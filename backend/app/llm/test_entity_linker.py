"""Unit tests for entity_linker.py — embed_texts is monkeypatched
throughout, no Ollama daemon or network involved."""

import pytest

from app.llm import entity_linker
from app.llm.entity_linker import (
    build_index,
    register_entity_type,
    register_exemplar_type,
    semantic_candidates,
    semantic_classify,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    entity_linker._reset_for_tests()
    monkeypatch.setattr(entity_linker.settings, "entity_linking_enabled", True)
    yield
    entity_linker._reset_for_tests()


@pytest.fixture()
def widget_type(monkeypatch):
    """A throwaway entity type so tests don't depend on entity_extractor's
    real registrations (advisor/team/company/...) and don't leak into
    them either."""
    register_entity_type("widget", lambda db: ["Alpha Team", "Beta Team", "Gamma Team"], kind="team")
    yield "widget"


def _fake_embed_factory(target: str):
    """[1, 0] for the target string (and the literal query text "query"),
    [0, 1] for everything else — deterministic cosine similarity without a
    real embedding model."""
    def _embed(texts):
        return [[1.0, 0.0] if t == target or t == "query" else [0.0, 1.0] for t in texts]
    return _embed


def test_disabled_by_flag_returns_empty_without_calling_embed(monkeypatch, widget_type):
    monkeypatch.setattr(entity_linker.settings, "entity_linking_enabled", False)
    called = []
    monkeypatch.setattr(entity_linker, "embed_texts", lambda texts: called.append(texts))
    assert semantic_candidates("anything", "widget", db=None) == []
    assert called == []


def test_unregistered_entity_type_returns_empty(monkeypatch):
    monkeypatch.setattr(entity_linker, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])
    assert semantic_candidates("anything", "no_such_type", db=None) == []


def test_embed_failure_returns_empty(monkeypatch, widget_type):
    monkeypatch.setattr(entity_linker, "embed_texts", lambda texts: None)
    assert semantic_candidates("anything", "widget", db=None) == []


def test_close_match_above_floor_is_returned_with_score(monkeypatch, widget_type):
    calls = {"n": 0}
    corpus_fn = _fake_embed_factory("Alpha Team")

    def _embed(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            return corpus_fn(texts)
        return [[1.0, 0.0] for _ in texts]  # query embedding

    monkeypatch.setattr(entity_linker, "embed_texts", _embed)
    results = semantic_candidates("query", "widget", db=None)
    assert results[0]["value"] == "Alpha Team"
    assert results[0]["score"] == pytest.approx(1.0)


def test_no_close_match_below_floor_returns_empty(monkeypatch, widget_type):
    calls = {"n": 0}

    def _embed(texts):
        calls["n"] += 1
        vector = [0.0, 1.0] if calls["n"] == 1 else [1.0, 0.0]
        return [vector for _ in texts]

    monkeypatch.setattr(entity_linker, "embed_texts", _embed)
    assert semantic_candidates("completely unrelated gibberish", "widget", db=None) == []


def test_top_k_limits_and_orders_results(monkeypatch):
    register_entity_type("scored_widget", lambda db: ["A", "B", "C"], kind="team")

    def _embed(texts):
        if texts == ["A", "B", "C"]:
            return [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(entity_linker, "embed_texts", _embed)
    results = semantic_candidates("query", "scored_widget", db=None, top_k=1, floor=0.5)
    assert len(results) == 1
    assert results[0]["value"] == "A"


def test_build_index_skips_rebuild_within_ttl(monkeypatch, widget_type):
    # other tests in this module register their own ad-hoc types, which
    # persist in the module-global registry for the rest of the session —
    # count only calls for THIS test's own candidate list so the count is
    # robust to that registry accumulation, matching entity_extractor.py's
    # real types (advisor/team/company/...) also being registered and
    # swept into every build_index() call.
    calls = []

    def _embed(texts):
        calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(entity_linker, "embed_texts", _embed)
    build_index(db=None)
    build_index(db=None)  # second call within TTL — no re-embed
    widget_calls = [c for c in calls if c == ["Alpha Team", "Beta Team", "Gamma Team"]]
    assert len(widget_calls) == 1


def test_build_index_force_rebuilds(monkeypatch, widget_type):
    calls = []

    def _embed(texts):
        calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(entity_linker, "embed_texts", _embed)
    build_index(db=None)
    build_index(db=None, force=True)
    widget_calls = [c for c in calls if c == ["Alpha Team", "Beta Team", "Gamma Team"]]
    assert len(widget_calls) == 2


def test_registered_types_reflects_registrations(widget_type):
    assert "widget" in entity_linker.registered_types()


# ---- exemplar-corpus retrieval (Part 12: semantic_classify / register_exemplar_type) ----

@pytest.fixture()
def mood_type():
    """A throwaway exemplar type: many phrases -> few canonical labels."""
    corpus = [
        ("hi", "greeting"), ("hello there", "greeting"),
        ("thanks a lot", "thanks"), ("appreciate it", "thanks"),
    ]
    register_exemplar_type("mood", lambda: corpus)
    yield "mood"


def test_semantic_classify_disabled_by_flag_returns_empty(monkeypatch, mood_type):
    monkeypatch.setattr(entity_linker.settings, "entity_linking_enabled", False)
    called = []
    monkeypatch.setattr(entity_linker, "embed_texts", lambda texts: called.append(texts))
    assert semantic_classify("hey", "mood") == []
    assert called == []


def test_semantic_classify_unregistered_type_returns_empty(monkeypatch):
    monkeypatch.setattr(entity_linker, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])
    assert semantic_classify("anything", "no_such_exemplar_type") == []


def test_semantic_classify_embed_failure_returns_empty(monkeypatch, mood_type):
    monkeypatch.setattr(entity_linker, "embed_texts", lambda texts: None)
    assert semantic_classify("hey", "mood") == []


def test_semantic_classify_returns_canonical_label_not_phrase(monkeypatch, mood_type):
    # "hi" and "hello there" both map to "greeting" — the query should
    # match whichever phrase scores best, but the RETURNED value is the
    # label, never the matched phrase itself
    calls = {"n": 0}

    def _embed(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            # corpus embed call — "hello there" gets the "matches" vector
            return [[1.0, 0.0] if t == "hello there" else [0.0, 1.0] for t in texts]
        return [[1.0, 0.0] for _ in texts]  # query embedding

    monkeypatch.setattr(entity_linker, "embed_texts", _embed)
    results = semantic_classify("yo there", "mood")
    assert results[0]["value"] == "greeting"
    assert results[0]["score"] == pytest.approx(1.0)


def test_semantic_classify_dedupes_by_label_best_score_wins(monkeypatch, mood_type):
    # BOTH greeting phrases score highly — only one "greeting" entry
    # should come back, not two near-duplicates
    def _embed(texts):
        if texts == ["hi", "hello there", "thanks a lot", "appreciate it"]:
            return [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.0, 1.0]]
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(entity_linker, "embed_texts", _embed)
    results = semantic_classify("hey", "mood", floor=0.5)
    labels = [r["value"] for r in results]
    assert labels.count("greeting") == 1
    assert results[0]["score"] == pytest.approx(1.0)  # best of the two greeting phrases


def test_semantic_classify_below_floor_returns_empty(monkeypatch, mood_type):
    def _embed(texts):
        return [[0.0, 1.0] for _ in texts] if len(texts) > 1 else [[1.0, 0.0]]

    monkeypatch.setattr(entity_linker, "embed_texts", _embed)
    assert semantic_classify("completely unrelated gibberish", "mood") == []


def test_registered_exemplar_types_reflects_registrations(mood_type):
    assert "mood" in entity_linker.registered_exemplar_types()
