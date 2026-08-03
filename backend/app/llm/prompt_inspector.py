"""
Read-only decomposition of an assembled prompt (diagnostics).

WHY THIS EXISTS — the question being investigated is "is the information
the model needs actually IN the prompt?", and that can't be answered by
reading prompt_builder.py, because what a builder can emit and what it
DID emit for one specific request are different things: the gazetteer,
the grounded-entity dict and the prior-turn IR are all interpolated at
call time from live data. Only the assembled string settles it.

This module PARSES that assembled string back into labelled sections. It
is a reader, never a writer:

- It does not build, edit, reorder, truncate or normalise prompt text.
  `segment()` takes a finished prompt and returns views into it.
- Partitioning is by LINE, so it is lossless by construction — every
  line lands in exactly one section. `reconstruct()` re-joins them and
  the audit log asserts the result is byte-identical to the input, so a
  segmentation that has drifted out of date with the prompt builders
  announces itself instead of quietly misrepresenting what was sent.
- An anchor that stops matching costs nothing: its content stays with
  the preceding section and the checksum still matches. Text before any
  anchor is reported as UNCLASSIFIED rather than silently dropped.

A NOTE ON "SYSTEM" AND "DEVELOPER" PROMPTS. This app sends exactly one
message per call — `[{"role": "user", "content": prompt}]` (see
llm_client.py). There is no system-role or developer-role message. The
role instruction ("You are a query-understanding parser…") and the
output contract (the JSON schema, the rules) are plain text INSIDE that
single user message. The categories below therefore label the
role-equivalent SECTIONS; `PromptBreakdown.roles_sent` reports what was
actually transmitted, so the two are never confused.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# The five categories the audit reports, in the order they're printed.
SYSTEM = "System Prompt"
DEVELOPER = "Developer Prompt"
CONTEXT = "Retrieved Context"
HISTORY = "Conversation History"
USER = "User Message"
UNCLASSIFIED = "Unclassified"

CATEGORY_ORDER = (SYSTEM, DEVELOPER, CONTEXT, HISTORY, USER, UNCLASSIFIED)

# (line prefix, category, human label). Ordered only for readability —
# matching is by prefix on each line, first hit wins. Anchors are copied
# from the literal text emitted by prompt_builder.build_ir_prompt(),
# planner_prompt.build_planner_prompt() and narrative.polish_explanation();
# nothing here is imported from those modules on purpose, so that a
# refactor of a prompt cannot be silently coupled to its diagnostics.
_ANCHORS: tuple[tuple[str, str, str], ...] = (
    # --- semantic parser (prompt_builder.py) ---
    ("You are a query-understanding parser", SYSTEM, "role instruction"),
    ("Known teams:", CONTEXT, "gazetteer: teams"),
    ("Known companies:", CONTEXT, "gazetteer: companies"),
    ("Known unit heads:", CONTEXT, "gazetteer: unit heads"),
    ("Known zonal heads:", CONTEXT, "gazetteer: zonal heads"),
    ("Known business centers:", CONTEXT, "gazetteer: business centers"),
    ("Metric catalog (the ONLY valid metric keys):", CONTEXT, "metric ontology"),
    ("How to read common business phrases:", DEVELOPER, "business phrase glossary"),
    ("Entities already found by rule-based grounding", CONTEXT, "grounded entities (per-request)"),
    ("Previous turn's resolved query", HISTORY, "prior turn QueryIR"),
    ("Examples (follow these exactly", CONTEXT, "few-shot examples"),
    ("User message:", USER, "user message"),
    ("Return ONLY a JSON object", DEVELOPER, "output contract + IR schema"),

    # --- LLM planner (planner_prompt.py) ---
    ("You convert a sales-operations question", SYSTEM, "role instruction"),
    ("INTENTS — pick exactly one:", DEVELOPER, "intent definitions"),
    ("DISTINCTIONS THAT DECIDE THE INTENT:", DEVELOPER, "intent distinctions"),
    ("RULES:", DEVELOPER, "planner rules"),
    ("ENTITY TYPES:", CONTEXT, "hierarchy levels"),
    ("METRIC CATALOG (the only valid metric keys):", CONTEXT, "metric ontology"),
    ("Known teams (sample):", CONTEXT, "gazetteer: teams"),
    ("PREVIOUS TURN's plan", HISTORY, "prior turn plan"),
    ("EXAMPLES:", CONTEXT, "few-shot examples"),
    ("USER QUESTION:", USER, "user message"),
    ("JSON:", DEVELOPER, "output contract"),

    # --- narrative polish (narrative.py) ---
    ("Lightly copy-edit the following explanation", SYSTEM, "role instruction"),
    ("Explanation:", CONTEXT, "deterministic explanation (SQL-derived)"),
)


@dataclass
class Section:
    category: str
    label: str
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class PromptBreakdown:
    sections: list[Section]
    prompt: str
    roles_sent: list[str] = field(default_factory=list)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:16]

    def reconstruct(self) -> str:
        """Re-joins the sections. Must equal the original prompt exactly —
        the audit prints the outcome of this check so a stale anchor set
        can never pass itself off as a faithful breakdown."""
        return "\n".join(line for s in self.sections for line in s.lines)

    @property
    def is_lossless(self) -> bool:
        return self.reconstruct() == self.prompt

    def by_category(self, category: str) -> list[Section]:
        return [s for s in self.sections if s.category == category]

    def category_chars(self, category: str) -> int:
        return sum(s.chars for s in self.by_category(category))

    def present(self, category: str) -> bool:
        return any(s.lines for s in self.by_category(category))


def segment(prompt: str, roles_sent: list[str] | None = None) -> PromptBreakdown:
    """Partition `prompt` into labelled sections. Pure and total: any
    string in, a lossless breakdown out — an unrecognised prompt simply
    comes back as one UNCLASSIFIED section."""
    sections: list[Section] = []
    current = Section(UNCLASSIFIED, "before first anchor")

    for line in (prompt or "").split("\n"):
        matched = _match(line)
        if matched is None:
            current.lines.append(line)
            continue
        category, label = matched
        if current.lines:
            sections.append(current)
        current = Section(category, label, [line])

    sections.append(current)

    # Drop a leading UNCLASSIFIED placeholder only when it holds NO lines
    # at all. Testing emptiness of the joined text instead would discard a
    # blank line and break losslessness for a prompt with no anchors.
    if sections and sections[0].category == UNCLASSIFIED and not sections[0].lines:
        sections = sections[1:]

    return PromptBreakdown(sections=sections, prompt=prompt or "", roles_sent=roles_sent or [])


def _match(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    for prefix, category, label in _ANCHORS:
        if stripped.startswith(prefix):
            return category, label
    return None
