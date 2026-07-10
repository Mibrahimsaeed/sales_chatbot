"""
Single entrypoint for turning free text into a resolved (intent, entities)
pair. Everything downstream (chat_service.py) calls resolve_intent() and
doesn't need to know whether the answer came from regex or an LLM call.

Routing decision: rule-based confidence >= CONFIDENCE_THRESHOLD -> trust it.
Below that -> ask the LLM, merge its entities on top of what regex already
found (regex entities are cheap and exact where they hit; LLM fills gaps).
"""

from sqlalchemy.orm import Session
from app.llm.preprocessing import normalize
from app.llm.entity_extractor import extract_entities, get_known_teams, get_known_companies
from app.llm.intent_detector import classify_intent, find_missing_slots, IntentResult
from app.llm.prompt_builder import build_intent_prompt
from app.llm.llm_client import classify_with_llm
from app.core.logger import get_logger

log = get_logger("llm.nlu_pipeline")

CONFIDENCE_THRESHOLD = 0.65


def resolve_intent(text: str, db: Session) -> IntentResult:
    cleaned = normalize(text)
    entities = extract_entities(cleaned, db)
    result = classify_intent(cleaned, entities)

    if result.confidence < CONFIDENCE_THRESHOLD:
        log.info(f"Low confidence ({result.confidence:.2f}) for '{text}' — trying LLM fallback")
        prompt = build_intent_prompt(cleaned, get_known_teams(db), get_known_companies(db))
        llm_output = classify_with_llm(prompt)

        if llm_output and llm_output.get("intent"):
            merged_entities = {
                **entities,
                **{k: v for k, v in (llm_output.get("entities") or {}).items() if v},
            }
            result = IntentResult(
                intent=llm_output["intent"],
                confidence=0.6,  # LLM-resolved: trusted enough to act on, tagged for logging
                entities=merged_entities,
                used_llm_fallback=True,
            )
        # if the LLM also failed/unavailable, keep the low-confidence rule-based
        # result as-is rather than silently becoming "unknown" — it may still
        # be right, just not confident enough to have skipped the fallback

    result.missing_slots = find_missing_slots(result)
    return result