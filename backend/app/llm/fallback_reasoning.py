"""
Runs only when query_planner comes back "unresolved". For a domain this
narrow, most unresolved cases are an unlisted synonym or slightly odd
phrasing — not something that actually needs a language model. Widening
the match here means the LLM fallback is reached rarely, which matters:
it's slower, costs money, and (as observed) can simply be unavailable due
to rate limits or provider quota — none of which should degrade the whole
chatbot to "I don't understand" for a query this answerable.
"""

from difflib import SequenceMatcher
from app.llm.metric_ontology import METRICS, describe_available_metrics

__all__ = ["fuzzy_resolve_metric", "describe_available_metrics"]


def fuzzy_resolve_metric(text: str, cutoff: float = 0.55) -> str | None:
    q = text.lower()
    best_key, best_score = None, 0.0
    for metric in METRICS.values():
        candidates = metric.synonyms + [metric.label.lower(), metric.key]
        for candidate in candidates:
            if candidate in q:
                return metric.key  # substring hit is a strong signal, short-circuit
            score = SequenceMatcher(None, q, candidate).ratio()
            if score > best_score:
                best_key, best_score = metric.key, score
    return best_key if best_score >= cutoff else None