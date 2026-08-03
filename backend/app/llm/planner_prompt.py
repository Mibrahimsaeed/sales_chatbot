"""
The planner prompt.

Built from the LIVE ontology and hierarchy rather than a hand-written
copy, so the prompt cannot drift from what the compiler can actually
execute: metric keys come from metric_ontology, entity types from
hierarchy.py. Adding a metric makes it plannable without touching this
file.

The prompt teaches MEANING, not keywords. That is the whole point of
replacing the rule-based planner — "who works under Kaleem", "show me
Kaleem's people", "which advisors sit under Kaleem" are three phrasings
of one question, and enumerating phrasings is the treadmill this
replaces. The examples below are therefore chosen to show the DISTINCTIONS
that matter (list-of-people vs nested-team vs one-person-above), not to
cover surface forms.
"""

from __future__ import annotations

from app.llm import hierarchy
from app.llm.periods import PERIODS
from app.llm.metric_ontology import METRICS

_ROLE = """You convert a sales-operations question into a structured query plan.

You are a PLANNER, not an assistant:
- Never write SQL.
- Never answer the question or invent any figure.
- Never guess an ID — name entities exactly as the user did; a separate
  resolver matches them to records.
- Output ONLY the JSON object. No prose, no markdown fences."""

_INTENTS = """INTENTS — pick exactly one:

advisor_profile     One named person's own record.
                    "tell me about Yasir Ali", "how is Waqar doing"

hierarchy           The people under someone, shown grouped by team.
                    Use when the question is about the TEAM/structure.
                    "show Kaleem's team", "who reports to Kaleem",
                    "who is in Adeel Dogar's team"

roster              A flat LIST of the people in a group. Use when the
                    question asks to enumerate ADVISORS/employees/staff.
                    "all advisors in Blue Area", "who works under Kaleem",
                    "list the staff in Graana"

reverse_hierarchy   The one person ABOVE someone (their manager).
                    "who is Yasir's BM", "who does Kaleem report to",
                    "who is Ali's zonal head"

leaderboard         A ranking by a metric.
                    "top 5 advisors by revenue", "worst teams by overdue"

comparison          Two or more named entities set against each other.
                    "compare Graana and Agency21", "Blue Area vs Downtown"

attendance_filter   People filtered by attendance status.
                    "who was late today", "late advisors in Blue Area"
                    Put the status in filters as attendance_status.

entity_summary      One group's aggregate performance.
                    "how is Blue Area doing", "Graana performance"

clarification       The question is genuinely ambiguous or you cannot
                    tell what was asked. Explain what you need in
                    `clarification`. Prefer this over guessing.

greeting / help     Social opener, or asking what the bot can do.

fallback            Nothing above applies."""

_DISTINCTIONS = """DISTINCTIONS THAT DECIDE THE INTENT:

1. hierarchy vs roster — both concern people under someone.
   Ask: does the user want the TEAM (grouped, structural) or a LIST OF
   ADVISORS (flat enumeration)?
     "show Kaleem's team"        -> hierarchy   (the team)
     "who works under Kaleem"    -> roster      (enumerate advisors)
     "all advisors in Blue Area" -> roster
     "who is in Kaleem's team"   -> hierarchy

2. hierarchy vs reverse_hierarchy — direction. Who is being asked ABOUT?
     "who reports to Kaleem"     -> hierarchy         (Kaleem's reports)
     "who does Kaleem report to" -> reverse_hierarchy (Kaleem's manager)

3. comparison vs leaderboard — a comparison names the specific things
   being set against each other; a leaderboard ranks a population.
     "compare Graana and Agency21 by revenue" -> comparison
     "top advisors in Graana by revenue"      -> leaderboard

4. roster vs entity_summary — enumerate people, or summarise performance?
     "advisors in Blue Area"     -> roster
     "how is Blue Area doing"    -> entity_summary

5. A named person does NOT imply advisor_profile. "Kaleem's team" names a
   person but asks about a group."""

_RULES = f"""RULES:

- entities: type + the user's own wording. Do not correct spelling, do
  not expand abbreviations, do not add an id.
- A person who MANAGES others is unit_head / zonal_head, not advisor,
  when the question is about the people under them.
- metric: only a key from the catalog below, or null. Never invent one.
  If the user names a measure you cannot map, use clarification.
- period: {" / ".join(PERIODS)}, or null when unstated. Do not default to MTD
  when the user said nothing — leave it null.
- limit: only when the user asked for a specific number ("top 5").
- confidence: your genuine certainty about the INTENT, 0-1. Below ~0.5
  the plan is treated as a clarification, so be honest rather than
  optimistic.
- Unsupported time windows (last month, yesterday, this week) are NOT
  expressible: use clarification and say so."""

_EXAMPLES = """EXAMPLES:

"Who works under Kaleem?"
{"intent":"roster","entities":[{"type":"unit_head","value":"Kaleem"}],"metric":null,"period":null,"filters":[],"sort":null,"limit":null,"confidence":0.93,"clarification":null}

"Show Kaleem's team"
{"intent":"hierarchy","entities":[{"type":"unit_head","value":"Kaleem"}],"metric":null,"period":null,"filters":[],"sort":null,"limit":null,"confidence":0.94,"clarification":null}

"Who does Kaleem report to?"
{"intent":"reverse_hierarchy","entities":[{"type":"advisor","value":"Kaleem"}],"metric":null,"period":null,"filters":[],"sort":null,"limit":null,"confidence":0.95,"clarification":null}

"Compare Graana and Agency21 by revenue"
{"intent":"comparison","entities":[{"type":"company","value":"Graana"},{"type":"company","value":"Agency21"}],"metric":"mtd_cleared","period":"MTD","filters":[],"sort":null,"limit":null,"confidence":0.96,"clarification":null}

"Top 5 advisors in Graana by revenue"
{"intent":"leaderboard","entities":[{"type":"company","value":"Graana"}],"metric":"mtd_cleared","period":"MTD","filters":[{"field":"company","operator":"=","value":"Graana"}],"sort":{"metric":"mtd_cleared","direction":"desc"},"limit":5,"confidence":0.95,"clarification":null}

"Late advisors in Blue Area"
{"intent":"attendance_filter","entities":[{"type":"team","value":"Blue Area"}],"metric":null,"period":null,"filters":[{"field":"attendance_status","operator":"=","value":"Late"}],"sort":null,"limit":null,"confidence":0.94,"clarification":null}

"Tell me about Yasir Ali"
{"intent":"advisor_profile","entities":[{"type":"advisor","value":"Yasir Ali"}],"metric":null,"period":null,"filters":[],"sort":null,"limit":null,"confidence":0.95,"clarification":null}

"How is Blue Area doing?"
{"intent":"entity_summary","entities":[{"type":"team","value":"Blue Area"}],"metric":null,"period":null,"filters":[],"sort":null,"limit":null,"confidence":0.92,"clarification":null}

"Who is struggling?"
{"intent":"clarification","entities":[],"metric":null,"period":null,"filters":[],"sort":null,"limit":null,"confidence":0.3,"clarification":"Which measure should I use — target achievement, revenue, or attendance?"}"""


def _metric_catalog() -> str:
    """Every metric key with its phrasings. Generated from the ontology so
    a new metric becomes plannable without editing this prompt."""
    lines = []
    for metric in METRICS.values():
        lines.append(f"  {metric.key}: {metric.label} — {', '.join(metric.synonyms[:6])}")
    return "\n".join(lines)


def _entity_types() -> str:
    return "\n".join(
        f"  {level}: {hierarchy.label_for(level)}"
        for level in hierarchy.HIERARCHY_LEVELS
    )


def build_planner_prompt(
    text: str,
    known_teams: list[str] | None = None,
    known_companies: list[str] | None = None,
    prior_plan_json: str | None = None,
) -> str:
    """The full prompt for one message.

    Gazetteer samples are GROUNDING, not a constraint: they help the
    model pick the right entity TYPE for a name it sees, but it must
    still emit the user's own wording (a name absent from the sample is
    not evidence the name is wrong — the resolver decides that)."""
    parts = [
        _ROLE,
        "",
        _INTENTS,
        "",
        _DISTINCTIONS,
        "",
        _RULES,
        "",
        "ENTITY TYPES:",
        _entity_types(),
        "",
        "METRIC CATALOG (the only valid metric keys):",
        _metric_catalog(),
    ]

    if known_teams:
        parts += ["", f"Known teams (sample): {', '.join(known_teams[:60])}"]
    if known_companies:
        parts += ["", f"Known companies: {', '.join(known_companies)}"]

    if prior_plan_json:
        parts += [
            "",
            "PREVIOUS TURN's plan — treat a short follow-up ('what about "
            f"YTD', 'only Graana') as a patch on this: {prior_plan_json}",
        ]

    parts += ["", _EXAMPLES, "", f'USER QUESTION: "{text}"', "", "JSON:"]
    return "\n".join(parts)
