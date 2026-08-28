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


# Synonyms shorter than this are matched EXACTLY but never fuzzily.
#
# Edit distance is meaningless at three characters: "mars" scores above
# the 0.55 cutoff against "crs", so "advisors in Mars Region" resolved to
# Client Registrations and answered a region query with a metric
# leaderboard. An exact/token match on a short synonym is still strong
# evidence ("top advisors by CR"); an approximate one is noise.
_MIN_FUZZY_SYNONYM_LENGTH = 4

# Fuzzy widening runs against the WHOLE query, so a metric name only has
# to resemble any part of the sentence. Measured against this ontology:
#
#   genuine typos            revnue/revenue 0.92, atendance rate 0.97,
#                            achievment 0.95, conections 0.95, pipline 0.93
#   coincidental resemblance "advisors in mars region" vs "mtd client
#                            registrations" 0.58, "north region" vs
#                            "client registration" 0.59
#
# The old 0.55 cutoff sat BELOW the coincidences — it simply never bit
# until a metric label happened to resemble a level word, at which point
# "advisors in Mars Region" was answered with a metric leaderboard. A
# typo and a coincidence are cleanly separated, so the floor belongs
# between them, not under both.
_FUZZY_CUTOFF = 0.80

# P0: the approximate tier is disabled — see the note at the end of
# fuzzy_resolve_metric. Kept as a named switch rather than deleted so the
# behaviour that was removed is still legible, and so a future change can
# re-enable it deliberately with evidence rather than by accident.
_APPROXIMATE_WIDENING_ENABLED = False


def fuzzy_resolve_metric(text: str, cutoff: float = _FUZZY_CUTOFF) -> str | None:
    from app.llm import metric_aliases, token_match

    # A measure the registry declares as known-but-uncomputable must not
    # be widened onto a neighbouring metric: "cr %" would fuzzy-match the
    # client-registration COUNT, handing back 47 for a percentage
    # question. Refusing to widen is what keeps the declaration
    # meaningful wherever this is called from.
    declared = metric_aliases.resolve(text)
    if declared is not None and not declared.available:
        return None
    # An EXACT registry hit wins over the synonym scan below, for the
    # same reason that scan short-circuits on its own exact hits: an
    # exact match is a strong signal and widening past it can only make
    # the answer worse.
    #
    # This became load-bearing when the working-day rates stopped being
    # refusals. "cr %" now resolves exactly to cr_rate — but it also
    # CONTAINS "cr", the client-registration count's synonym, so the scan
    # would hand back the count for a percentage question: the precise
    # substitution the refusal above was protecting against, arriving by
    # a different route the moment the refusal was retired.
    if declared is not None and declared.metric:
        return declared.metric

    q = text.lower()
    synonym_to_key = {}
    for metric in METRICS.values():
        for candidate in metric.synonyms + [metric.label.lower(), metric.key]:
            # Token-aware, matching resolve_metric_evidence — plain
            # containment let a short synonym fire inside a longer word.
            if token_match.contains(q, candidate):
                return metric.key  # exact hit is a strong signal, short-circuit
            if len(candidate) >= _MIN_FUZZY_SYNONYM_LENGTH:
                synonym_to_key.setdefault(candidate, metric.key)

    # P0 SAFETY: THE APPROXIMATE TIER IS OFF.
    #
    # Everything above this point is an EXACT lookup — the registry, or a
    # token-aware synonym hit — and stays, because it is the same
    # resolution metric_ontology.resolve_metric performs and it guesses
    # nothing.
    #
    # What is disabled is the fuzzy scan that followed: it matched a
    # metric name against ANY window of the sentence, so a measure only
    # had to RESEMBLE part of the question. That turned "I don't know
    # which measure this is" into a confident wrong one — the audit found
    # "Unit Heads with team size > 5" widened onto `one_unit_ratio` from
    # the stray word "Unit", then filtered on it and reported the result
    # as an answer.
    #
    # A wrong measure is worse than no answer, because nothing downstream
    # can tell it apart from a right one: the number is well-formed, the
    # label matches the metric that was guessed, and the reply reads
    # exactly like a correct one. Asking the user which measure they meant
    # costs a turn; guessing costs their trust in every other number.
    #
    # The genuine typos this served ("revnue", "achievment") are a real
    # loss, and the honest place to recover them is the LLM parser, which
    # sees the whole sentence rather than an edit distance.
    if _APPROXIMATE_WIDENING_ENABLED:  # pragma: no cover - off by default
        hits = find_in_text(q, list(synonym_to_key), kind="metric", floor=cutoff)
        if hits:
            return synonym_to_key[hits[0][0]]
    return None