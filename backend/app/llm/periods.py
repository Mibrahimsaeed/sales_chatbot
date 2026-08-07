"""
THE period vocabulary. One tuple, no imports, no behaviour.

WHY IT IS ITS OWN MODULE. The set of periods was written out by hand in
five places — llm_client's JSON schema, planner_schema's Literal and its
JSON schema, prompt_builder's IR schema, planner_prompt's rules — and
they only agreed by luck. The moment DAILY was added to temporal_parser
they stopped agreeing, and because llm_client's copy is sent with
`strict: True`, grammar-constrained decoding made DAILY *unemittable*:
an LLM-parsed "revenue today" was forced back to MTD. That is audit
finding F5 surviving on the default path while the rule-based path had
already been fixed.

The obvious home was temporal_parser, but that module imports
entity_linker (for the semantic widening tier), which reaches the
embedding client, which imports llm_client — so llm_client cannot import
from it. The vocabulary itself has no dependencies at all, so it lives
here, in a module nothing can cycle through.

WHAT A PERIOD MEANS HERE. It is a WINDOW THE SYSTEM CAN NAME, not one it
can necessarily answer. DAILY is recognised so that "revenue today" can
be refused honestly instead of silently answered with month-to-date
numbers; whether a given measure has data at a given period is
metric_ontology's question, answered by metric_for_period().
"""

from __future__ import annotations

from typing import Literal

# Ordered narrowest to widest, which is the order a prompt or a schema
# reads best in.
PERIODS: tuple[str, ...] = ("DAILY", "MTD", "YTD", "3M")

Period = Literal[PERIODS]

# Human wording for each, for replies that need to say which window they
# used or could not use. Declared beside the vocabulary so a new period
# cannot arrive without one.
PERIOD_LABELS: dict[str, str] = {
    "DAILY": "daily",
    "MTD": "month-to-date",
    "YTD": "year-to-date",
    "3M": "3-month",
}


def label_for(period: str | None) -> str:
    """The human wording for a period, falling back to the raw value so
    an unknown one is visible rather than hidden."""
    if period is None:
        return "unspecified"
    return PERIOD_LABELS.get(period, period)


# How each period is SPELLED INSIDE A METRIC LABEL, longest first.
#
# Distinct from PERIOD_LABELS above, which is the prose wording for a
# sentence ("month-to-date"); this is the shorthand the ontology's own
# labels are written with ("Total MTD Connects", "CR % (MTD)",
# "3-Month Revenue Cleared"). Declared here rather than in
# metric_ontology because it is a fact about the period vocabulary, and
# because this module is the one place a new period can be added — a
# period whose spelling were declared elsewhere could arrive without one.
PERIOD_TOKENS: dict[str, tuple[str, ...]] = {
    "DAILY": ("Daily",),
    "MTD": ("MTD",),
    "YTD": ("YTD",),
    "3M": ("3-Month", "3M"),
}


def without_period(label: str, period: str | None) -> str:
    """`label` with its own period's wording removed.

    A metric key encodes ONE period, so its label names that period:
    `total_connects` is "Total MTD Connects". That is right when the
    label captions an answer — the reply says which window it computed.
    It is wrong when the sentence is about a DIFFERENT window, which is
    exactly the unavailable-period message: "I don't have daily figures
    for Total MTD Connects" names two periods and asks the reader to
    work out that only one of them is the question.

    Derived from the metric's declared period rather than declared a
    second time per metric, so a label and its neutral form cannot drift
    — and a new metric gets one without a second edit. The tidy-up
    handles the three shapes the ontology actually uses (prefix, infix,
    parenthesised suffix) with plain string operations; the exhaustive
    test over METRICS is what keeps that claim true.

    Returns the label unchanged when removal would leave nothing, so a
    metric named only for its period stays visible rather than blank.
    """
    stripped = label
    for token in PERIOD_TOKENS.get(period or "", ()):
        stripped = stripped.replace(token, "")
    stripped = stripped.replace("()", "")
    while "  " in stripped:
        stripped = stripped.replace("  ", " ")
    stripped = stripped.strip().strip("-–—").strip()
    return stripped or label
