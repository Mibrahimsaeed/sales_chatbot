"""
Semantic retrieval (Part 8) — embedding-based metric matching for
paraphrased queries the fuzzy synonym scan in fallback_reasoning.py can't
reach (e.g. "who's crushing it this month" for achievement_pct: no lexical
overlap with any synonym, so RapidFuzz can't help, but it's semantically
close to "best performer" / "top performer").

Plugs into semantic_parser._rule_based_ir() as a THIRD widening attempt,
after fuzzy_resolve_metric() fails — this one call site is shared by both
NLU_MODE=llm_first's degrade path and rules_first's step 2, so nothing in
semantic_parser.parse()'s branching needs to change.

Fail-soft exactly like every other LLM call site: no key, no quota, a
network error, or settings.semantic_retrieval_enabled=False all just mean
retrieve_metric() returns None and the caller moves on to its next-best
option (a clarifying question). Once embed_texts() fails once in this
process, further attempts are skipped for the rest of the process instead
of retrying a call that's already known to fail — the same "known-dead"
short-circuit entity_extractor's gazetteer cache and llm_client's client
availability check already rely on.

NOT verified against the live API — OpenAI quota was exhausted (429) at
the time this was written. Covered by unit tests with a mocked
embed_texts() instead; the real behavior needs the existing YAML benchmark
harness (tests/benchmark, gated behind `-m benchmark`) run for real once
quota is restored before any claim of improved paraphrase recognition.
"""

from __future__ import annotations

import math

from app.core.config import settings
from app.core.logger import get_logger
from app.llm.llm_client import embed_texts
from app.llm.metric_ontology import METRICS

log = get_logger("llm.semantic_retrieval")

# Below this cosine similarity, a "best match" isn't confident enough to
# trust — same spirit as fuzzy_match.py's STRONG_FLOOR: a false-positive
# metric guess silently answers the wrong question, which is worse than
# asking for clarification.
_SIMILARITY_FLOOR = 0.75

_exemplar_vectors: dict[str, list[float]] | None = None
_unavailable = False


def _exemplar_corpus() -> list[tuple[str, str]]:
    """(phrase, metric_key) pairs generated from the ontology — same
    source metric_catalog_for_prompt() already uses — so there's no
    second, hand-maintained list to drift out of sync."""
    return [
        (phrase, metric.key)
        for metric in METRICS.values()
        for phrase in metric.synonyms + [metric.label.lower()]
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _ensure_exemplar_vectors() -> dict[str, list[float]] | None:
    """Lazily embeds the exemplar corpus once per process (mirrors
    entity_extractor._cache's per-process caching pattern, simpler here
    since the corpus is static — no DB-driven TTL refresh needed)."""
    global _exemplar_vectors, _unavailable
    if _exemplar_vectors is not None:
        return _exemplar_vectors
    if _unavailable:
        return None

    corpus = _exemplar_corpus()
    vectors = embed_texts([phrase for phrase, _key in corpus])
    if vectors is None:
        _unavailable = True
        return None

    _exemplar_vectors = {phrase: vec for (phrase, _key), vec in zip(corpus, vectors)}
    return _exemplar_vectors


def retrieve_metric(text: str) -> str | None:
    if not settings.semantic_retrieval_enabled:
        return None

    exemplar_vectors = _ensure_exemplar_vectors()
    if exemplar_vectors is None:
        return None

    query_vectors = embed_texts([text])
    if not query_vectors:
        return None
    query_vector = query_vectors[0]

    corpus = _exemplar_corpus()
    best_key: str | None = None
    best_score = 0.0
    for phrase, key in corpus:
        vector = exemplar_vectors.get(phrase)
        if vector is None:
            continue
        score = _cosine(query_vector, vector)
        if score > best_score:
            best_key, best_score = key, score

    if best_key is not None and best_score >= _SIMILARITY_FLOOR:
        log.info(f"Semantic retrieval matched '{text}' -> {best_key} (score {best_score:.2f})")
        return best_key
    return None


def _reset_cache_for_tests() -> None:
    """Test-only hook — see test_semantic_retrieval.py."""
    global _exemplar_vectors, _unavailable
    _exemplar_vectors = None
    _unavailable = False
