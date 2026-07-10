# """
# Single entrypoint for turning free text into a resolved (intent, entities)
# pair. Everything downstream (chat_service.py) calls resolve_intent() and
# doesn't need to know whether the answer came from regex or an LLM call.

# Routing decision: rule-based confidence >= CONFIDENCE_THRESHOLD -> trust it.
# Below that -> ask the LLM, merge its entities on top of what regex already
# found (regex entities are cheap and exact where they hit; LLM fills gaps).
# """

# from sqlalchemy.orm import Session
# from app.llm.preprocessing import normalize
# from app.llm.entity_extractor import extract_entities, get_known_teams, get_known_companies
# from app.llm.intent_detector import classify_intent, find_missing_slots, IntentResult
# from app.llm.prompt_builder import build_intent_prompt
# from app.llm.llm_client import classify_with_llm
# from app.core.logger import get_logger

# log = get_logger("llm.nlu_pipeline")

# CONFIDENCE_THRESHOLD = 0.65


# def resolve_intent(text: str, db: Session) -> IntentResult:
#     cleaned = normalize(text)
#     entities = extract_entities(cleaned, db)
#     result = classify_intent(cleaned, entities)

#     if result.confidence < CONFIDENCE_THRESHOLD:
#         log.info(f"Low confidence ({result.confidence:.2f}) for '{text}' — trying LLM fallback")
#         prompt = build_intent_prompt(cleaned, get_known_teams(db), get_known_companies(db))
#         llm_output = classify_with_llm(prompt)

#         if llm_output and llm_output.get("intent"):
#             merged_entities = {
#                 **entities,
#                 **{k: v for k, v in (llm_output.get("entities") or {}).items() if v},
#             }
#             result = IntentResult(
#                 intent=llm_output["intent"],
#                 confidence=0.6,  # LLM-resolved: trusted enough to act on, tagged for logging
#                 entities=merged_entities,
#                 used_llm_fallback=True,
#             )
#         # if the LLM also failed/unavailable, keep the low-confidence rule-based
#         # result as-is rather than silently becoming "unknown" — it may still
#         # be right, just not confident enough to have skipped the fallback

#     result.missing_slots = find_missing_slots(result)
#     return result

"""
Resolution order, cheapest and most deterministic first:

1. Shortcuts — greeting / help / attendance_check. Fixed patterns, zero cost,
   handled by intent_detector.py.
2. Entity extraction — advisor/team/company gazetteer match + period/limit.
3. Query planner — ontology-driven. Resolves a QueryPlan directly from
   entities + ranking/level keywords. No LLM involved for the large
   majority of real queries, including ones like "give me target
   achievement" that a fixed keyword list would have missed.
4. Fallback reasoning — if the planner comes back unresolved, widen the
   metric match with fuzzy synonym comparison before giving up.
5. LLM assist — genuinely last resort, and optional. If the provider is
   unavailable, rate-limited, or misconfigured, this degrades to a
   schema-grounded clarifying question — never a dead end and never a
   blocking retry loop.
"""

from dataclasses import dataclass
from sqlalchemy.orm import Session
from app.llm.preprocessing import normalize
from app.llm.entity_extractor import extract_entities, get_known_teams, get_known_companies
from app.llm.intent_detector import classify_intent as classify_shortcut
from app.llm.query_planner import build_query_plan, QueryPlan
from app.llm.fallback_reasoning import fuzzy_resolve_metric, describe_available_metrics
from app.llm.prompt_builder import build_intent_prompt
from app.llm.llm_client import classify_with_llm
from app.core.logger import get_logger

log = get_logger("llm.nlu_pipeline")

SHORTCUT_INTENTS = ("greeting", "help", "attendance_check")


@dataclass
class Resolution:
    kind: str                          # "shortcut" | "plan" | "clarify"
    shortcut_intent: str | None = None
    plan: QueryPlan | None = None
    entities: dict | None = None
    used_llm_fallback: bool = False
    clarify_message: str | None = None


def resolve(text: str, db: Session) -> Resolution:
    cleaned = normalize(text)

    shortcut = classify_shortcut(cleaned, {})
    if shortcut.intent in SHORTCUT_INTENTS:
        return Resolution(kind="shortcut", shortcut_intent=shortcut.intent, entities={})

    entities = extract_entities(cleaned, db)
    plan = build_query_plan(cleaned, entities)

    if plan.action != "unresolved":
        return Resolution(kind="plan", plan=plan, entities=entities)

    widened_metric = fuzzy_resolve_metric(cleaned)
    if widened_metric:
        entities["metric"] = widened_metric
        plan = build_query_plan(cleaned, entities)
        if plan.action != "unresolved":
            log.info(f"Fallback reasoning resolved '{text}' via widened metric match: {widened_metric}")
            return Resolution(kind="plan", plan=plan, entities=entities)

    prompt = build_intent_prompt(cleaned, get_known_teams(db), get_known_companies(db))
    llm_output = classify_with_llm(prompt)
    if llm_output and llm_output.get("intent") not in (None, "unknown"):
        merged_entities = {**entities, **{k: v for k, v in (llm_output.get("entities") or {}).items() if v}}
        plan = build_query_plan(cleaned, merged_entities)
        if plan.action != "unresolved":
            return Resolution(kind="plan", plan=plan, entities=merged_entities, used_llm_fallback=True)

    return Resolution(
        kind="clarify",
        entities=entities,
        clarify_message=(
            f"I couldn't match that to something I track. I can answer about: "
            f"{describe_available_metrics()} — or look up an advisor, team, or company by name."
        ),
    )