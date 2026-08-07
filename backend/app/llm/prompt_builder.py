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

from app.llm import hierarchy, periods
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

def _period_union() -> str:
    """The period enum, from app/llm/periods.py — the same list
    llm_client's grammar-constrained schema enforces. Hardcoded here as
    "MTD"|"YTD"|"3M" until Phase 5.1, which is how the prompt came to
    offer a vocabulary the schema had already outgrown."""
    return " | ".join(f'"{period}"' for period in periods.PERIODS)


def _period_glossary() -> str:
    return "\n".join(
        f"    {period}: {periods.label_for(period)}" for period in periods.PERIODS
    )


def _level_union() -> str:
    """The addressable levels as a JSON-schema-style union.

    Generated from hierarchy.HIERARCHY_LEVELS so the prompt and the
    grammar-constrained enum in llm_client.QUERY_IR_JSON_SCHEMA are
    literally the same list. They were hand-written separately, and had
    drifted badly: the prompt offered `unit` and `business_center`, which
    the schema REJECTS, and never mentioned `bcm`, which is a real chain
    level. With strict decoding the model could not comply with the
    instructions it was given.
    """
    return " | ".join(f'"{level}"' for level in hierarchy.HIERARCHY_LEVELS)


def _chain_description() -> str:
    """The verified nesting chain, top to bottom, with business labels.

    Phase 1 disproved the chain this prompt used to state and Phase 3
    replaced it; the prompt was never updated, so the model reasoned
    about an org chart that does not exist — team at the BOTTOM, a
    non-existent `unit` level, and no BCM.
    """
    return " -> ".join(
        f"{level} ({hierarchy.label_for(level)})" for level in hierarchy.CHAIN
    )


def _attribute_description() -> str:
    """The groupable attributes, which do NOT nest — see
    hierarchy.ATTRIBUTE_LEVELS. Kept separate from the chain in the
    prompt because conflating the two is what produced "company ->
    region -> unit_head" as though those contained one another."""
    return ", ".join(
        f"{level} ({hierarchy.label_for(level)})" for level in hierarchy.ATTRIBUTE_LEVELS
    )


def _role_vocabulary() -> str:
    """Which words name which chain level, from the SAME relation
    registry the rule-based planner reads (relations.role_aliases via
    intent_catalog). Previously a hand-written gloss listing synonyms for
    levels that no longer exist."""
    lines = []
    for level in hierarchy.CHAIN:
        if level == "advisor":
            continue
        keywords = hierarchy.LEVEL_KEYWORDS.get(level, [])
        if keywords:
            lines.append(f'  {level}: say "{'", "'.join(keywords[:6])}"')
    return "\n".join(lines)


def _ir_schema() -> str:
    """Built at call time so a hierarchy change reaches the prompt without
    anyone remembering to edit prose."""
    levels = _level_union()
    return f"""Return ONLY a JSON object, no other text, no markdown fences, matching this shape:
{{
  "intent": "leaderboard" | "comparison" | "lookup" | "trend" | "filtered_list" | "breakdown" | "clarify",
  "subject_level": {levels},
  "subjects": [ {{ "type": {levels}, "value": string, "match_confidence": number }} ],
  "metric": {{ "key": string, "confidence": number }} | null,
  "filters": [ {{ "field": string, "operator": "="|"!="|">"|">="|"<"|"<="|"in", "value": string|number, "confidence": number }} ],
  "time_range": {{ "mode": "snapshot"|"compare", "period": {_period_union()}, "compare_to": string|null, "confidence": number }},
  "sort": {{ "metric": string|null, "direction": "asc"|"desc" }},
  "limit": number|null,
  "group_by": {levels}|null,
  "flat": boolean,
  "overall_confidence": number,
  "intent_confidence": number
}}

Org hierarchy, top to bottom — each level CONTAINS the next:
{_chain_description()}

A Team is the widest grouping; an Advisor is one person. Unit Head, Zonal Head and BCM are
PEOPLE who oversee the level below them — they are not synonyms for "team". Use one of them as
subject_level (for rankings like "top 5 unit heads by connects") or as a filter field (for
"advisors under zonal head X") only when the user's words actually name that role:
{_role_vocabulary()}
Never infer one of these from a bare team or company mention.

These are ATTRIBUTES, not levels of the chain — an advisor has one, but they do not contain or
nest inside each other: {_attribute_description()}.
"advisors in North Region" filters region; "who is X's zonal head" asks for a person. Use an
attribute only when the user says that word. A ranking over one ("top companies by revenue",
"top business centers by connects") is valid and should use it as subject_level.

Rules:
- "filters" holds EVERY condition mentioned, AND-combined by default — a query can filter on
  one metric while sorting by another (e.g. sort by revenue, filter attendance_status = Late).
- "field" in a filter is either one of the metric keys below, or one of: {", ".join(hierarchy.HIERARCHY_LEVELS)}, attendance_status.
- Use "comparison" intent with 2+ entries in "subjects" for "compare X with Y" style queries —
  this applies at any subject_level, including "compare unit head A with unit head B".
- Use "breakdown" intent for a question about ONE specific named entity at any level above advisor
  ("tell me about unit head X", "give me a breakdown of zonal head Y", "show me business center
  Z") — put that one entity in "subjects" (exactly one entry) at the matching subject_level. This
  is NOT a ranking (no "metric"/"sort" needed) — it returns that one entity's advisors nested by
  the level below it. Set "flat": true only if the user explicitly asks for a flat/ungrouped list
  (e.g. "list all advisors under X, not grouped") — "flat" defaults to false and is ignored for
  every intent other than "breakdown".
- Use "filtered_list" when the user wants a list matching conditions but isn't asking for a
  ranking (no explicit sort implied) — otherwise use "leaderboard".
- If the user's business language is ambiguous ("struggling", "consistently performs well") and
  isn't clearly one of the metrics below, set intent to "clarify" and explain your best guess as
  a filter with a lower confidence rather than inventing a new field.
- Only use metric keys from the catalog below — never invent one.
- Every confidence number (including the two below) is 0-1 and scores ONLY its own dimension —
  don't let one shaky field drag another field's score down.
- "intent_confidence" scores whether you picked the right QUERY SHAPE (leaderboard vs comparison
  vs filtered_list vs clarify), independent of whether any one metric/filter/subject value is
  itself uncertain. A query can have a very confident shape (0.9+) even if the metric guess is
  shaky, or vice versa.
- "period" values, and what each means:
{_period_glossary()}
  DAILY is for "today" / "right now" / "this morning". Use it when the user names
  TODAY — do not substitute MTD. Most measures have no daily data and the query will be
  refused, which is the correct outcome: answering a question about today with
  month-to-date figures is a wrong answer, not a partial one.
- Time windows this system cannot express at all (last month, yesterday, this week, a
  custom date range) are NOT period values. Set intent to "clarify" rather than
  choosing the nearest period.
- "time_range.confidence" scores how sure you are about the period. The user not mentioning a
  time period at all and you defaulting to MTD is a LOW-confidence guess (~0.5-0.6) even though
  it's a common default — reserve high confidence (0.9+) for when the user's words actually imply
  or state the period ("this month", "ytd", "year to date", "last 3 months")."""


def build_ir_prompt(
    text: str,
    known_teams: list[str],
    known_companies: list[str],
    grounded_entities: dict,
    prior_ir_json: str | None = None,
    recent_turns: list | None = None,
    known_unit_heads: list[str] | None = None,
    known_zonal_heads: list[str] | None = None,
    known_bcms: list[str] | None = None,
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
    if known_unit_heads:
        context_lines.append(f"Known unit heads: {', '.join(known_unit_heads[:200])}")
    if known_zonal_heads:
        context_lines.append(f"Known zonal heads: {', '.join(known_zonal_heads[:200])}")
    if known_bcms:
        context_lines.append(f"Known BCMs: {', '.join(known_bcms[:200])}")
    context_lines.append("Metric catalog (the ONLY valid metric keys):")
    context_lines.append(metric_catalog)
    context_lines.append(BUSINESS_PHRASE_GLOSSARY)

    # Underscore-prefixed keys are META about the extraction (provenance
    # — see entity_extractor._finalize_provenance), not grounded values
    # the model should read. Filtered rather than never-stored so the
    # metadata stays available to the planner and the audit log while the
    # prompt text remains exactly what it was before provenance existed.
    prompt_entities = {k: v for k, v in grounded_entities.items() if not k.startswith("_")}
    if prompt_entities:
        context_lines.append(f"Entities already found by rule-based grounding (use these, don't re-derive): {prompt_entities}")

    if prior_ir_json:
        context_lines.append(
            "Previous turn's resolved query (for follow-ups like 'what about last month' or "
            f"'same for Downtown' — treat the new message as a patch on this): {prior_ir_json}"
        )

    # The conversation as MESSAGES, alongside the structured prior IR
    # above rather than instead of it. The IR is the authoritative record
    # of what the last query RESOLVED to and the deterministic layer
    # patches it directly; this is the wording, which carries the
    # references that layer cannot see — a name that appeared only in a
    # reply, or a subject that never grounded to an entity.
    #
    # Bounded upstream by conversation_memory.recent_turns(), so this
    # renders whatever fits the configured turn and character budget.
    if recent_turns:
        rendered = "\n".join(f"  {role}: {text}" for role, text in recent_turns)
        context_lines.append(
            "Recent conversation (oldest first) — resolve pronouns and "
            f"ellipsis against it, but prefer the resolved query above when "
            f"the two disagree:\n{rendered}"
        )

    context_lines.append(render_examples())
    context_lines.append(f'User message: "{text}"')
    context_lines.append(_ir_schema())

    return "\n".join(context_lines)
