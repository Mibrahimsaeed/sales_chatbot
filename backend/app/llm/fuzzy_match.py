"""
Single home for all fuzzy string matching (P3 of the NLU rework).
Replaces the three scattered difflib.SequenceMatcher call sites
(entity_extractor, ir_validator, fallback_reasoning) with RapidFuzz,
which handles typos ("Grana" -> "Graana"), swapped word order
("Haider Waqar" -> "Waqar Haider"), and partial names far better than
plain longest-common-subsequence ratios.

All scores are normalized to 0-1 so the existing 0.55 threshold
semantics carry over unchanged.

Two shapes of question, two functions:
- best_match(value, candidates, kind): "which candidate IS this value?"
  — for already-isolated strings (an LLM-supplied subject, a name after
  filler-stripping).
- find_in_text(text, candidates, kind): "which candidates APPEAR in this
  free text?" — slides token n-grams sized to each candidate over the
  text, so a candidate embedded in a longer sentence isn't penalized for
  everything around it.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

DEFAULT_FLOOR = 0.55
# Teams/companies matched fuzzily (not substring) need a higher bar before
# being auto-accepted: a false-positive team filter silently narrows the
# result set, which is worse than asking. 0.55-0.80 hits are still
# returned — with their score — so callers can ask "did you mean X?".
STRONG_FLOOR = 0.80


def _scorer(kind: str):
    if kind == "advisor":
        # token_sort handles swapped name order at full strength; token_set
        # lets a bare first/last name hit a full name, discounted to 0.9 so
        # a partial-name match never outranks a full-name match.
        return lambda a, b: max(
            fuzz.token_sort_ratio(a, b), 0.9 * fuzz.token_set_ratio(a, b)
        ) / 100.0
    if kind == "company":
        # short names ("Graana") — token scorers over-match, plain ratio +
        # partial_ratio is the right tool
        return lambda a, b: max(fuzz.ratio(a, b), fuzz.partial_ratio(a, b)) / 100.0
    # teams, metrics, anything else: WRatio blends full/partial/token views
    return lambda a, b: fuzz.WRatio(a, b) / 100.0


def best_match(
    value: str, candidates: list[str], kind: str = "team", floor: float = DEFAULT_FLOOR
) -> tuple[str, float] | None:
    """Best candidate for an already-isolated value, or None below floor.
    Exact (case-insensitive) matches short-circuit to score 1.0."""
    if not value:
        return None
    score_fn = _scorer(kind)
    v = value.lower().strip()
    best_name, best_score = None, 0.0
    for c in candidates:
        if not c:
            continue
        if c.lower() == v:
            return c, 1.0
        score = score_fn(v, c.lower())
        if score > best_score:
            best_name, best_score = c, score
    if best_name and best_score >= floor:
        return best_name, round(best_score, 2)
    return None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def find_in_text(
    text: str, candidates: list[str], kind: str = "team", floor: float = DEFAULT_FLOOR
) -> list[tuple[str, float]]:
    """All candidates fuzzily present in free text, best-scored window per
    candidate, sorted by score descending. A candidate's windows are token
    n-grams of its own token count +/- 1, so 'only grana advisors' scores
    'Graana' against the window 'grana', not the whole sentence."""
    text_tokens = _tokens(text)
    if not text_tokens:
        return []
    # windows are already sized to the candidate, so partial-substring
    # scorers (WRatio/partial_ratio) are wrong here — they'd score "man"
    # against "performance" at ~0.9 via the embedded substring. Strict
    # full-window comparison only; token_sort covers word-order swaps.
    score_fn = lambda a, b: max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b)) / 100.0

    results: list[tuple[str, float]] = []
    for c in candidates:
        if not c:
            continue
        c_lower = c.lower()
        n = max(len(_tokens(c_lower)), 1)
        best = 0.0
        for size in (n - 1, n, n + 1):
            if size < 1 or size > len(text_tokens):
                continue
            for i in range(len(text_tokens) - size + 1):
                window = " ".join(text_tokens[i : i + size])
                score = 1.0 if window == c_lower else score_fn(window, c_lower)
                if score > best:
                    best = score
        if best >= floor:
            results.append((c, round(best, 2)))

    return sorted(results, key=lambda pair: -pair[1])
