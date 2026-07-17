import json
from openai import OpenAI
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("llm.client")

_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

# Hand-written rather than derived from QueryIR.model_json_schema(): OpenAI's
# strict structured-output mode requires every property "required" (optional
# fields are expressed as a type union with "null", not by omission) and
# additionalProperties: false at every object level. Pydantic v2's generated
# schema doesn't emit that shape out of the box, and reshaping it generically
# on every call is more moving parts than hand-maintaining one schema that
# mirrors query_ir.QueryIR field-for-field (see prompt_builder.IR_SCHEMA,
# which documents the same shape for the model to read).
QUERY_IR_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "subject_level", "subjects", "metric", "filters", "time_range", "sort", "limit", "group_by", "overall_confidence"],
    "properties": {
        "intent": {"type": "string", "enum": ["leaderboard", "comparison", "lookup", "trend", "filtered_list", "clarify"]},
        "subject_level": {"type": "string", "enum": ["advisor", "team", "company"]},
        "subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "value", "match_confidence"],
                "properties": {
                    "type": {"type": "string", "enum": ["advisor", "team", "company"]},
                    "value": {"type": "string"},
                    "match_confidence": {"type": "number"},
                },
            },
        },
        "metric": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["key", "confidence"],
                    "properties": {"key": {"type": "string"}, "confidence": {"type": "number"}},
                },
            ]
        },
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "operator", "value", "confidence"],
                "properties": {
                    "field": {"type": "string"},
                    "operator": {"type": "string", "enum": ["=", "!=", ">", ">=", "<", "<=", "in"]},
                    "value": {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}]}}, {"type": "null"}]},
                    "confidence": {"type": "number"},
                },
            },
        },
        "time_range": {
            "type": "object",
            "additionalProperties": False,
            "required": ["mode", "period", "compare_to"],
            "properties": {
                "mode": {"type": "string", "enum": ["snapshot", "compare"]},
                "period": {"type": "string", "enum": ["MTD", "YTD", "3M"]},
                "compare_to": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        },
        "sort": {
            "type": "object",
            "additionalProperties": False,
            "required": ["metric", "direction"],
            "properties": {
                "metric": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "direction": {"type": "string", "enum": ["asc", "desc"]},
            },
        },
        "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "group_by": {"anyOf": [{"type": "string", "enum": ["advisor", "team", "company"]}, {"type": "null"}]},
        "overall_confidence": {"type": "number"},
    },
}


def call_llm_structured(prompt: str, schema: dict, schema_name: str) -> dict | None:
    """API-enforced structured output (Part 5.3): the model is constrained
    to the given JSON schema server-side, so a malformed shape becomes a
    provider-level error this function catches — not a `json.JSONDecodeError`
    or a QueryIR that silently fails pydantic validation downstream. This
    is what actually removes the "same narrow schema as the regex layer"
    bottleneck the architecture review flagged: the model is grounded in a
    schema that CAN express compound queries, and can't drift from it.

    Fails soft exactly like call_llm_json() — any error returns None so
    semantic_parser.py degrades to the rule-based plan_to_ir() path.
    """
    if not _client:
        log.warning("openai_api_key not set — LLM parsing unavailable, staying rule-based only")
        return None

    try:
        response = _client.responses.create(
            model="gpt-5.5",
            input=prompt,
            max_output_tokens=800,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        return json.loads(response.output_text.strip())

    except json.JSONDecodeError:
        log.warning("LLM returned unparseable JSON despite schema constraint")
        return None

    except Exception:
        log.exception("Structured LLM call failed")
        return None


def call_llm_json(prompt: str) -> dict | None:
    """Schema-agnostic fallback for any caller without a strict JSON
    schema to enforce (e.g. the legacy INTENT_SCHEMA-style prompt some
    older code paths may still build). New code parsing a QueryIR should
    call call_llm_structured() with QUERY_IR_JSON_SCHEMA instead — see
    semantic_parser.py.

    Fails soft — returns None on any error (missing API key, malformed
    JSON, provider error/timeout/rate-limit) so the pipeline can degrade to
    the rule-based fallback rather than crash the chat endpoint or block on
    a retry loop.
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
