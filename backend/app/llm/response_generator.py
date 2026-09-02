"""The final answer, written by the LLM from the verified result set.

    verified result set -> final LLM generator -> final answer

WHAT THIS REPLACES. The model already had a hand in the reply, but only
as a copy-editor: `narrative.polish_explanation` handed it one
deterministic sentence and asked it to smooth the phrasing. It never saw
the user's question, the conversation, or a single row — so it could not
answer "why is that low?", could not compare two rows, and could not say
anything the template had not already said.

This module gives it all four inputs Phase 10 names — the query, the
recent turns, the rows, and the result metadata — and lets it compose.

THE RESULT SET IS THE SOURCE OF TRUTH, and that is enforced rather than
requested. A generated answer is accepted only if it survives every check
in `_violations`; otherwise the deterministic explanation is served
unchanged. The model can therefore improve the reply and cannot corrupt
it — the worst case is exactly the behaviour that existed before.

WHY GUARDS AND NOT JUST INSTRUCTIONS. "Do not invent numbers" in a prompt
is a request. A reply that quietly reports 26,811 as 26,800, or mentions
Downtown when only Blue Area was queried, is well-formed, confident and
wrong, and nothing downstream can tell. The checks below are cheap, and
each one corresponds to a line in this phase's "must not" list.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logger import get_logger
from app.llm.llm_client import call_llm_json
from app.llm.narrative import _numbers_in

log = get_logger("llm.response_generator")

# How much of the result set to show the model. The reply describes the
# page the user is looking at, so this matches what was rendered rather
# than the full match set.
_MAX_ROWS = 25
# A final answer is a few sentences. A long one is a sign the model has
# started narrating rather than answering.
_MAX_CHARS = 700

# Words that begin a sentence or a label and are not entities. Without
# this the capitalised-phrase check fires on ordinary prose.
_SENTENCE_WORDS = frozenset({
    "the", "this", "that", "these", "those", "a", "an", "and", "but", "or",
    "in", "on", "at", "for", "with", "by", "from", "to", "of", "as", "it",
    "its", "their", "there", "here", "they", "he", "she", "we", "you", "i",
    "no", "not", "all", "both", "each", "every", "most", "more", "less",
    "least", "top", "bottom", "best", "worst", "highest", "lowest", "so",
    "if", "when", "while", "however", "overall", "across", "between",
    "compared", "meanwhile", "note", "notably", "together", "meaning",
    "your", "our", "his", "her", "based", "given", "only", "just", "still",
    "yes", "none", "nobody", "everyone", "someone",
})

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# Runs of capitalised words: "Blue Area", "Ahmed Raza", "Downtown".
_CAPITALISED_RE = re.compile(r"\b[A-Z][\w'&/.-]*(?:\s+[A-Z][\w'&/.-]*)*")


@dataclass
class GeneratedAnswer:
    """What the model produced and whether it may be served."""
    text: str | None = None
    accepted: bool = False
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"accepted": self.accepted, "violations": list(self.violations)}


def _render_rows(rows: list[dict]) -> str:
    """The result set as the model sees it: compact, complete, and
    already the only place any figure may come from."""
    trimmed = []
    for row in rows[:_MAX_ROWS]:
        kept = {k: v for k, v in row.items()
                if k in ("name", "team", "company", "value", "wid") and v is not None}
        cells = row.get("metrics")
        if isinstance(cells, dict):
            kept["metrics"] = {k: v for k, v in cells.items()}
        trimmed.append(kept)
    return json.dumps(trimmed, default=str)


def _allowed_numbers(rows: list[dict], facts: dict, explanation: str,
                     metadata: dict) -> set[str]:
    """Every figure the answer is allowed to contain.

    Drawn from the ROWS themselves plus the deterministic facts, so the
    allowlist is the verified result set rather than anything the model
    supplied.
    """
    allowed = _numbers_in(rows) | _numbers_in(facts) | _numbers_in(metadata)
    if explanation:
        allowed |= _numbers_in(explanation)
    # Ordinals and small counts a sentence needs to describe a list of
    # this size ("the top three", "both teams") are not claims about data.
    allowed |= {f"{n:g}" for n in range(0, min(len(rows), _MAX_ROWS) + 1)}
    return allowed


def _allowed_text(rows: list[dict], query: str, explanation: str,
                  metadata: dict) -> str:
    """One blob every named thing in the answer must be found in."""
    parts = [query or "", explanation or "", json.dumps(metadata, default=str)]
    for row in rows[:_MAX_ROWS]:
        for key in ("name", "team", "company"):
            value = row.get(key)
            if isinstance(value, str):
                parts.append(value)
        cells = row.get("metrics")
        if isinstance(cells, dict):
            parts.extend(str(k) for k in cells)
    return " \n ".join(parts).lower()


def _violations(answer: str, rows: list[dict], facts: dict, query: str,
                explanation: str, metadata: dict) -> list[str]:
    """Every way the answer fails to be supported by the result set.

    Returns all of them rather than the first: a rejected answer is
    logged, and one reason rarely explains a bad generation.
    """
    problems: list[str] = []

    if not answer or not answer.strip():
        return ["empty"]
    if len(answer) > _MAX_CHARS:
        problems.append(f"too long ({len(answer)} chars)")

    invented = _numbers_in(answer) - _allowed_numbers(rows, facts, explanation, metadata)
    if invented:
        problems.append(f"numbers not in the result set: {sorted(invented)[:5]}")

    haystack = _allowed_text(rows, query, explanation, metadata)
    unsupported = []
    for phrase in _CAPITALISED_RE.findall(answer):
        cleaned = phrase.strip()
        if not cleaned or cleaned.lower() in _SENTENCE_WORDS:
            continue
        # A multi-word run whose first token is just a sentence opener
        # ("Overall Blue Area led") should be judged on the rest.
        tokens = cleaned.split()
        while tokens and tokens[0].lower() in _SENTENCE_WORDS:
            tokens = tokens[1:]
        if not tokens:
            continue
        candidate = " ".join(tokens)
        if candidate.lower() not in haystack:
            unsupported.append(candidate)
    if unsupported:
        problems.append(f"entities not in the result set: {sorted(set(unsupported))[:5]}")

    return problems


def _prompt(query: str, rows: list[dict], facts: dict, explanation: str,
            metadata: dict, recent_turns: list[tuple[str, str]] | None) -> str:
    conversation = ""
    if recent_turns:
        rendered = "\n".join(f"  {role}: {text}" for role, text in recent_turns[-4:])
        conversation = (
            "Recent conversation (oldest first). Use it only to understand what "
            f"the user is asking; it is not data:\n{rendered}\n"
        )

    return (
        "You are a sales-operations assistant writing the final answer for a "
        "colleague.\n\n"
        "THE RESULT SET BELOW IS THE ONLY SOURCE OF TRUTH. Every figure and every "
        "name in your answer must come from it. Do not calculate, re-derive, "
        "round, estimate or compare anything that is not already there — if a "
        "number is not in the result set, you cannot state it. Do not mention any "
        "team, company or person that does not appear in it. If the result set is "
        "empty, say plainly that there is no data, and nothing else.\n\n"
        "You MAY explain what the numbers mean, summarise them, compare rows that "
        "are both present, and write conversationally.\n\n"
        f"User's question: {query}\n"
        f"{conversation}"
        f"Verified result set (JSON): {_render_rows(rows)}\n"
        f"Result metadata: {json.dumps(metadata, default=str)}\n"
        f"Verified summary computed from those rows: {explanation}\n\n"
        "Write 1-3 short sentences. Do not use bullet points or headings — the "
        "table of results is shown separately beneath your answer.\n"
        'Return ONLY JSON: {"answer": "<your answer>"}'
    )


def generate(query: str, rows: list[dict], facts: dict, explanation: str,
             *, metadata: dict | None = None,
             recent_turns: list[tuple[str, str]] | None = None) -> GeneratedAnswer:
    """Compose the final answer from the verified result set.

    Never raises and never returns unchecked text: `accepted` is True only
    for an answer that passed every guard. Callers serve `explanation`
    otherwise, which is what the reply contained before this existed.
    """
    metadata = metadata or {}

    if not settings.nlu_narrative:
        return GeneratedAnswer(violations=["narrative disabled"])
    if not rows:
        # Nothing to ground against. The deterministic "no results"
        # sentence is already correct, and a model asked to be
        # conversational about an empty set is exactly where inventions
        # come from.
        return GeneratedAnswer(violations=["no rows to ground against"])

    try:
        raw = call_llm_json(_prompt(query, rows, facts, explanation, metadata,
                                    recent_turns))
    except Exception:
        # call_llm_json is documented never to raise, but this function
        # promises the same thing to a caller that is midway through
        # building a reply. A response layer that can throw turns a
        # cosmetic feature into an outage.
        log.warning("Final answer generation failed — serving the "
                    "deterministic explanation", exc_info=True)
        return GeneratedAnswer(violations=["provider raised"])

    if not raw or not isinstance(raw.get("answer"), str):
        return GeneratedAnswer(violations=["provider returned nothing usable"])

    answer = raw["answer"].strip()
    problems = _violations(answer, rows, facts, query, explanation, metadata)
    if problems:
        log.warning("Generated answer rejected (%s) — serving the deterministic "
                    "explanation", "; ".join(problems))
        return GeneratedAnswer(text=answer, accepted=False, violations=problems)

    return GeneratedAnswer(text=answer, accepted=True)


def generate_or(explanation: str, query: str, rows: list[dict], facts: dict,
                *, metadata: dict | None = None,
                recent_turns: list[tuple[str, str]] | None = None) -> str:
    """The final answer, or the deterministic explanation unchanged.

    Same fail-soft contract `polish_explanation` had, so the reply is
    never blanked and the API shape never changes: a rejected generation
    costs the conversational phrasing, nothing else.
    """
    result = generate(query, rows, facts, explanation,
                      metadata=metadata, recent_turns=recent_turns)
    return result.text if result.accepted and result.text else explanation
