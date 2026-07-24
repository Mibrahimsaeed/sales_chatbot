"""
Hybrid semantic entity linking (Part 9) — a third widening step behind
exact substring and RapidFuzz fuzzy matching in entity_extractor.py /
ir_validator.py, for names that are unseen, paraphrased, abbreviated, or
too badly mangled for edit-distance scoring to recover (e.g. "the Islamabad
lead" for a portfolio_lead name that shares no tokens with the query, or
"WH" for "Waqar Haider").

The resolution cascade and its confidence tiers are, in order:
  1. exact   — case-insensitive substring/equality match, score 1.0
  2. fuzzy   — RapidFuzz (fuzzy_match.py), score is an edit-distance ratio
  3. semantic — cosine similarity over Ollama embeddings, score is that
               similarity

Each tier only runs if the previous one found nothing, and every tier
returns its own score so a caller can tell which floor a match cleared. If
nothing clears the semantic floor either, the caller gets [] back and must
ask for clarification rather than guess — the same contract fuzzy_match.py
and semantic_retrieval.py already use.

Pluggable by design: adding a new linkable entity type (a future "zone",
"region", whatever) is one register_entity_type() call with a candidate-
fetching function — nothing else in this module changes. entity_extractor.py
is the sole place that currently calls register_entity_type(); see its
gazetteer cache for the fetch functions passed in.

Fail-soft exactly like semantic_retrieval.py: no Ollama, no embedding model
pulled, a network error, or ENTITY_LINKING_ENABLED=False all just mean
semantic_candidates() returns [] and the caller falls back to its next-best
option (fuzzy already failed, so that means a clarifying question) instead
of raising or guessing.

Part 12 (semantic retrieval expansion) added a SECOND registration mode,
register_exemplar_type()/semantic_classify(), for ontology-shaped
categories where MANY natural-language phrasings map to FEW canonical
labels — intents, temporal buckets, comparison operators, attendance-
status filters (see intent_detector.py, temporal_parser.py,
entity_extractor.py) — as opposed to register_entity_type()'s DB-gazetteer
categories where each candidate string IS the value itself. Same
pluggable/fail-soft/floor-gated contract either way; this module is now
the one shared engine behind "semantic retrieval" everywhere in the NLU
pipeline except metrics, which predates this module (semantic_retrieval.py)
and is left as its own thing to avoid churning a working, independently
tested piece for the sake of consolidation alone.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import get_logger
from app.llm.llm_client import embed_texts

log = get_logger("llm.entity_linker")

# Same spirit as fuzzy_match.STRONG_FLOOR: a false-positive entity guess
# silently narrows or misdirects a query, which is worse than asking. This
# sits below STRONG_FLOOR (0.80) because embedding similarity and edit-
# distance ratio aren't the same scale — semantic hits for genuine
# paraphrases ("the Islamabad lead") often land lower than a typo'd exact
# name would on RapidFuzz.
SEMANTIC_FLOOR = 0.75

# Index candidates are re-embedded at most this often — mirrors
# entity_extractor._CACHE_TTL_SECONDS so a newly onboarded advisor/team
# becomes semantically findable on roughly the same cadence it becomes
# fuzzy/exact-findable, without re-embedding the whole gazetteer on every
# single query.
_INDEX_TTL_SECONDS = 300


@dataclass
class EntityTypeSpec:
    name: str
    fetch_fn: Callable[[Session], list[str]]
    kind: str = "team"  # fuzzy_match.py scorer kind, kept alongside for consistency


@dataclass
class _Index:
    values: list[str] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    built_at: float = 0.0


_REGISTRY: dict[str, EntityTypeSpec] = {}
_INDEXES: dict[str, _Index] = {}


def register_entity_type(name: str, fetch_fn: Callable[[Session], list[str]], kind: str = "team") -> None:
    """The one step required to make a new entity type semantically
    linkable. `fetch_fn(db)` should return the current list of known
    values for that type (typically a thin wrapper around a gazetteer
    cache, like entity_extractor.get_known_teams)."""
    _REGISTRY[name] = EntityTypeSpec(name=name, fetch_fn=fetch_fn, kind=kind)
    _INDEXES.setdefault(name, _Index())


def registered_types() -> list[str]:
    return list(_REGISTRY.keys())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_one(name: str, db: Session) -> None:
    spec = _REGISTRY[name]
    idx = _INDEXES[name]
    try:
        values = sorted({v for v in spec.fetch_fn(db) if v})
    except Exception:
        # one type's candidate fetch failing (bad session, transient DB
        # error) must not stop every other registered type from building —
        # same fail-soft contract as an embeddings-layer failure below.
        log.exception(f"Entity index build skipped for '{name}' — candidate fetch failed")
        return

    if not values:
        idx.values, idx.vectors, idx.built_at = [], [], time.time()
        return

    vectors = embed_texts(values)
    if vectors is None:
        # leave built_at untouched so the next call retries instead of
        # treating this type as permanently dead — Ollama coming back
        # (or getting pulled) should self-heal without a restart.
        log.warning(f"Entity index build skipped for '{name}' — embeddings unavailable")
        return

    idx.values, idx.vectors, idx.built_at = values, vectors, time.time()


def build_index(db: Session, force: bool = False) -> None:
    """Embeds every registered entity type's current candidate list.
    Best-effort — called eagerly at app startup (main.py) and refreshed
    lazily here on a TTL so index staleness tracks gazetteer staleness.
    Safe to call repeatedly; a type whose TTL hasn't expired is a no-op."""
    now = time.time()
    for name, idx in _INDEXES.items():
        if not force and idx.built_at and now - idx.built_at < _INDEX_TTL_SECONDS:
            continue
        _build_one(name, db)


def semantic_candidates(
    text: str, entity_type: str, db: Session, top_k: int = 3, floor: float = SEMANTIC_FLOOR
) -> list[dict]:
    """Top-k {value, score} candidates for `text` against `entity_type`'s
    embedding index, above `floor`, best score first. [] whenever the
    feature is off, the type isn't registered, embeddings are unavailable,
    or nothing clears the floor — every case the caller should treat as
    "ask for clarification", never as "assume no match was possible"."""
    if not settings.entity_linking_enabled or entity_type not in _REGISTRY:
        return []

    build_index(db)
    idx = _INDEXES[entity_type]
    if not idx.values or not idx.vectors:
        return []

    query_vectors = embed_texts([text])
    if not query_vectors:
        return []
    query_vector = query_vectors[0]

    scored = [
        {"value": value, "score": round(_cosine(query_vector, vector), 2)}
        for value, vector in zip(idx.values, idx.vectors)
    ]
    scored = [c for c in scored if c["score"] >= floor]
    scored.sort(key=lambda c: -c["score"])
    return scored[:top_k]


# ---------------------------------------------------------------------
# Exemplar-corpus retrieval (Part 12) — MANY phrases -> FEW canonical
# labels, for intents/temporal buckets/comparison operators/attendance
# statuses. Parallel to the entity-type registry above; independent
# storage since the shapes differ (labels aren't unique candidates).
# ---------------------------------------------------------------------

@dataclass
class ExemplarTypeSpec:
    name: str
    corpus_fn: Callable[[], list[tuple[str, str]]]  # () -> [(phrase, canonical_label), ...]


@dataclass
class _ExemplarIndex:
    phrases: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    built_at: float = 0.0


_EXEMPLAR_REGISTRY: dict[str, ExemplarTypeSpec] = {}
_EXEMPLAR_INDEXES: dict[str, _ExemplarIndex] = {}


def register_exemplar_type(name: str, corpus_fn: Callable[[], list[tuple[str, str]]]) -> None:
    """The one step required to make a new ontology-shaped category
    semantically classifiable. `corpus_fn()` returns a static list of
    (example phrase, canonical label) pairs — e.g. [("hey there",
    "greeting"), ("yo", "greeting"), ("thanks a lot", "thanks")]. No `db`
    parameter: unlike register_entity_type()'s gazetteers, this corpus is
    static ontology content, not something pulled from the database."""
    _EXEMPLAR_REGISTRY[name] = ExemplarTypeSpec(name=name, corpus_fn=corpus_fn)
    _EXEMPLAR_INDEXES.setdefault(name, _ExemplarIndex())


def registered_exemplar_types() -> list[str]:
    return list(_EXEMPLAR_REGISTRY.keys())


def _build_exemplar_index(name: str) -> None:
    spec = _EXEMPLAR_REGISTRY[name]
    idx = _EXEMPLAR_INDEXES[name]
    corpus = spec.corpus_fn()
    if not corpus:
        idx.phrases, idx.labels, idx.vectors, idx.built_at = [], [], [], time.time()
        return

    phrases = [phrase for phrase, _label in corpus]
    vectors = embed_texts(phrases)
    if vectors is None:
        log.warning(f"Exemplar index build skipped for '{name}' — embeddings unavailable")
        return

    idx.phrases = phrases
    idx.labels = [label for _phrase, label in corpus]
    idx.vectors = vectors
    idx.built_at = time.time()


def semantic_classify(
    text: str, exemplar_type: str, top_k: int = 3, floor: float = SEMANTIC_FLOOR
) -> list[dict]:
    """Top-k {value, score} CANONICAL LABELS for `text` against
    `exemplar_type`'s exemplar corpus — the best-scoring phrase per label
    wins (so a label with ten phrasings doesn't crowd out the ranking with
    ten near-duplicate entries), sorted best first. [] whenever the
    feature is off, the type isn't registered, embeddings are unavailable,
    or nothing clears the floor — same contract as semantic_candidates().
    No `db` parameter — exemplar corpora are static, so there's nothing to
    query; callers with a DB session (entity_extractor.py) simply don't
    pass one through."""
    if not settings.entity_linking_enabled or exemplar_type not in _EXEMPLAR_REGISTRY:
        return []

    idx = _EXEMPLAR_INDEXES[exemplar_type]
    now = time.time()
    if not idx.built_at or now - idx.built_at >= _INDEX_TTL_SECONDS:
        _build_exemplar_index(exemplar_type)
    if not idx.phrases or not idx.vectors:
        return []

    query_vectors = embed_texts([text])
    if not query_vectors:
        return []
    query_vector = query_vectors[0]

    best_by_label: dict[str, float] = {}
    for phrase, label, vector in zip(idx.phrases, idx.labels, idx.vectors):
        score = _cosine(query_vector, vector)
        if score > best_by_label.get(label, -1.0):
            best_by_label[label] = score

    scored = [{"value": label, "score": round(score, 2)} for label, score in best_by_label.items() if score >= floor]
    scored.sort(key=lambda c: -c["score"])
    return scored[:top_k]


def _reset_for_tests() -> None:
    """Test-only hook — see test_entity_linker.py. Clears built indexes
    without losing type registrations (those happen once, at import time,
    in entity_extractor.py)."""
    for name in _INDEXES:
        _INDEXES[name] = _Index()
    for name in _EXEMPLAR_INDEXES:
        _EXEMPLAR_INDEXES[name] = _ExemplarIndex()
