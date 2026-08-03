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
