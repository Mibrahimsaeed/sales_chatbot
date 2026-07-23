"""
Multi-intent splitter — light version (Part 8). Handles genuinely
compound requests ("top advisors; also list attendance issues") by
splitting into independent sub-queries, each resolved through the normal
single-query pipeline (nlu_pipeline.resolve()) and stitched into labeled
sections. This is NOT a QueryIR schema change (no sub_queries[] field) —
each section is answered as its own separate query.

The critical constraint: tests/benchmark/cases/compound.yaml is full of
"X and Y" phrasings that are correctly ONE multi-filter query today (e.g.
"attendance at least 90 and achievement above 80, sorted by meetings") —
semantic_parser._COMPOUND_HINTS already routes bare and/but to a single
compound-filter parse. This splitter must never fire on bare and/but, so
it only triggers on markers that don't appear in any existing compound
case: also/additionally/separately, ';', a newline, two or more '?', or a
numbered-list marker ('1)'/'1.'). Anything else returns None and the
normal single-query path (including the existing multi-filter handling)
runs completely unchanged.
"""

from __future__ import annotations

import re

_MIN_WORDS = 3

# "1) " anywhere after whitespace (numbering with parens is unambiguous —
# nobody ends a sentence with a bare digit+paren), but "1. " only at the
# very start of the string/line — a bare "N." ending a sentence ("cleared
# more than 60.") must NOT be mistaken for a list marker.
_LIST_MARKER_RE = re.compile(
    r"(?:^|\n)\s*\d{1,2}\.\s+" r"|(?:^|(?<=\s))\d{1,2}\)\s+", re.M
)
_CONNECTOR_RE = re.compile(r"\b(?:also|additionally|separately)\b", re.I)


def _valid_segments(parts: list[str]) -> list[str] | None:
    segments = [p.strip(" ,.;") for p in parts if p and p.strip(" ,.;")]
    segments = [s for s in segments if len(re.findall(r"\S+", s)) >= _MIN_WORDS]
    return segments if len(segments) >= 2 else None


def split_subqueries(text: str) -> list[str] | None:
    """Two or more independent sub-queries, or None if this text should
    go through the normal single-query pipeline instead."""
    stripped = text.strip()

    if _LIST_MARKER_RE.search(stripped):
        result = _valid_segments(_LIST_MARKER_RE.split(stripped))
        if result:
            return result

    if ";" in stripped or "\n" in stripped:
        result = _valid_segments(re.split(r"[;\n]+", stripped))
        if result:
            return result

    if stripped.count("?") >= 2:
        result = _valid_segments(stripped.split("?"))
        if result:
            return result

    if _CONNECTOR_RE.search(stripped):
        result = _valid_segments(_CONNECTOR_RE.split(stripped))
        if result:
            return result

    return None
