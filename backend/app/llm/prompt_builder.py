"""
Builds the prompt for the LLM Semantic Parser (Part 5.3). The LLM's job
changed from "classify an intent into a fixed flat schema" to "author a
structured, composable QueryIR" — so the schema below is the full IR
shape (query_ir.py), not the old {intent, advisor_name, team, company,
metric, period, limit} struct. This is the actual fix for Root Cause #4
in the redesign brief: the old schema had nowhere to put a second filter,
a comparator, or a boolean condition — this one does.

The metric ontology is included as grounding context (same idea this file
already used for teams/companies, extended to metrics) so the model can't
invent a metric key that has no compiler binding — ir_validator.py double
checks this regardless, but grounding it in the prompt cuts down on wasted
round trips.
"""

from app.llm.ir_examples import render_examples
from app.llm.metric_ontology import metric_catalog_for_prompt

# Business phrases that don't literally name a metric but have one
# conventional meaning in this domain. Anything NOT on this list and not a
# catalog synonym should produce intent="clarify", not a guess.
BUSINESS_PHRASE_GLOSSARY = """How to read common business phrases:
- "best performer" / "top performer" / "star" -> sort desc by achievement_pct
- "underperforming" / "weak" / "bottom performers" -> sort asc by achievement_pct
- "almost achieved target" / "close to target" -> two filters: achievement_pct >= 80 AND achievement_pct < 100
- "highest closer" / "biggest closer" -> sort desc by mtd_cleared
- "doing well" -> sort desc by achievement_pct
- "punctual" / "shows up on time" -> attendance_rate high (sort desc or filter >)
- "never on time" / "always late" -> late_count high or attendance_rate low"""

IR_SCHEMA = """Return ONLY a JSON object, no other text, no markdown fences, matching this shape:
{
  "intent": "leaderboard" | "comparison" | "lookup" | "trend" | "filtered_list" | "clarify",
  "subject_level": "advisor" | "team" | "company",
  "subjects": [ { "type": "advisor"|"team"|"company", "value": string, "match_confidence": number } ],
  "metric": { "key": string, "confidence": number } | null,
  "filters": [ { "field": string, "operator": "="|"!="|">"|">="|"<"|"<="|"in", "value": string|number, "confidence": number } ],
  "time_range": { "mode": "snapshot"|"compare", "period": "MTD"|"YTD"|"3M", "compare_to": string|null },
  "sort": { "metric": string|null, "direction": "asc"|"desc" },
  "limit": number|null,
  "group_by": "advisor"|"team"|"company"|null,
  "overall_confidence": number
}

Rules:
- "filters" holds EVERY condition mentioned, AND-combined by default — a query can filter on
  one metric while sorting by another (e.g. sort by revenue, filter attendance_status = Late).
- "field" in a filter is either one of the metric keys below, or one of: team, company, advisor,
  attendance_status.
- Use "comparison" intent with 2+ entries in "subjects" for "compare X with Y" style queries.
- Use "filtered_list" when the user wants a list matching conditions but isn't asking for a
  ranking (no explicit sort implied) — otherwise use "leaderboard".
- If the user's business language is ambiguous ("struggling", "consistently performs well") and
  isn't clearly one of the metrics below, set intent to "clarify" and explain your best guess as
  a filter with a lower confidence rather than inventing a new field.
- Only use metric keys from the catalog below — never invent one."""


def build_ir_prompt(
    text: str,
    known_teams: list[str],
    known_companies: list[str],
    grounded_entities: dict,
    prior_ir_json: str | None = None,
) -> str:
    # 200 is a generosity cap, not a truncation strategy — the real
    # gazetteer is far smaller; the cap only guards against a pathological
    # data load blowing up the prompt.
    teams_sample = ", ".join(known_teams[:200])
    companies = ", ".join(known_companies)
    metric_catalog = metric_catalog_for_prompt()

    context_lines = [f"You are a query-understanding parser for a real-estate sales operations chatbot."]
    context_lines.append(f"Known teams: {teams_sample}")
    context_lines.append(f"Known companies: {companies}")
    context_lines.append("Metric catalog (the ONLY valid metric keys):")
    context_lines.append(metric_catalog)
    context_lines.append(BUSINESS_PHRASE_GLOSSARY)

    if grounded_entities:
        context_lines.append(f"Entities already found by rule-based grounding (use these, don't re-derive): {grounded_entities}")

    if prior_ir_json:
        context_lines.append(
            "Previous turn's resolved query (for follow-ups like 'what about last month' or "
            f"'same for Downtown' — treat the new message as a patch on this): {prior_ir_json}"
        )

    context_lines.append(render_examples())
    context_lines.append(f'User message: "{text}"')
    context_lines.append(IR_SCHEMA)

    return "\n".join(context_lines)
