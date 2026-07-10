import json
from openai import OpenAI
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("llm.client")

_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None


def classify_with_llm(prompt: str) -> dict | None:
    """Fails soft — returns None on any error so the pipeline can fall back
    to 'unknown' rather than crash the chat endpoint."""
    if not _client:
        log.warning("openai_api_key not set — LLM fallback unavailable, staying rule-based only")
        return None

    try:
        response = _client.responses.create(
            model="gpt-5.5",
            input=prompt,
            max_output_tokens=300,
        )

        text = response.output_text.strip()
        return json.loads(text)

    except json.JSONDecodeError:
        log.warning("LLM returned unparseable JSON for intent classification")
        return None

    except Exception:
        log.exception("LLM intent classification call failed")
        return None
    


    