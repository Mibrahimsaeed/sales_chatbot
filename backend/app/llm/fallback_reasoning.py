"""
Runs only when query_planner comes back "unresolved". For a domain this
narrow, most unresolved cases are an unlisted synonym or slightly odd
phrasing — not something that actually needs a language model. Widening
the match here means the LLM fallback is reached rarely, which matters:
it's slower, costs money, and (as observed) can simply be unavailable due
to rate limits or provider quota — none of which should degrade the whole
chatbot to "I don't understand" for a query this answerable.
"""

from app.llm.fuzzy_match import find_in_text
from app.llm.metric_ontology import METRICS, describe_available_metrics

__all__ = ["fuzzy_resolve_metric", "describe_available_metrics"]


def fuzzy_resolve_metric(text: str, cutoff: float = 0.55) -> str | None:
    q = text.lower()
    synonym_to_key = {}
    for metric in METRICS.values():
        for candidate in metric.synonyms + [metric.label.lower(), metric.key]:
            if candidate in q:
                return metric.key  # substring hit is a strong signal, short-circuit
            synonym_to_key.setdefault(candidate, metric.key)

    # typo'd synonyms ("atendance rate", "revnue") — fuzzy-scan the text
    # windows against every synonym, best hit wins
    hits = find_in_text(q, list(synonym_to_key), kind="metric", floor=cutoff)
    if hits:
        return synonym_to_key[hits[0][0]]
    return None