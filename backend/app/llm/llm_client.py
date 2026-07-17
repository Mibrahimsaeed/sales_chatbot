import json
from openai import OpenAI
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("llm.client")

_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None


def call_llm_json(prompt: str) -> dict | None:
    """Fails soft — returns None on any error (missing API key, malformed
    JSON, provider error/timeout/rate-limit) so the pipeline can degrade to
    the rule-based fallback (query_ir.plan_to_ir) rather than crash the
    chat endpoint or block on a retry loop. This is now schema-agnostic —
    semantic_parser.py is responsible for validating the parsed dict into
    a QueryIR; this function's only job is "get JSON back from the model,
    or fail soft."

    NOTE: the previous version's `classify_with_llm()` name is retained as
    a thin alias below for anything still importing it directly, but new
    code should call this instead.
    """
    if not _client:
        log.warning("openai_api_key not set — LLM parsing unavailable, staying rule-based only")
        return None

    try:
        response = _client.responses.create(
            model="gpt-5.5",
            input=prompt,
            max_output_tokens=600,
        )

        text = response.output_text.strip()
        return json.loads(text)

    except json.JSONDecodeError:
        log.warning("LLM returned unparseable JSON")
        return None

    except Exception:
        log.exception("LLM call failed")
        return None


def classify_with_llm(prompt: str) -> dict | None:
    """Backward-compat alias — see call_llm_json()."""
    return call_llm_json(prompt)