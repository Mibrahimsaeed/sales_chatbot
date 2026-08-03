import json

from openai import OpenAI

from app.core import audit
from app.core.config import settings
from app.core.logger import get_logger
from app.llm.hierarchy import HIERARCHY_LEVELS
from app.llm.periods import PERIODS

log = get_logger("llm.client")

# Back on OpenAI's hosted API (was Ollama — see git history) since a local
# model was too slow for interactive chat latency. Client construction is
# cheap and doesn't connect eagerly; "unavailable" (bad key, no quota,
# network error, timeout) surfaces as an exception at call time, caught by
# each function below exactly like every other failure mode already was.
#
# max_retries=0: the SDK's default (2 retries, exponential backoff) is
# actively counterproductive here — a 429 insufficient_quota or bad-key
# error can never succeed on retry, so the default just triples the
# latency of every failed call before this module's own fail-soft path
# (return None -> rule-based degrade) ever gets a chance to run. One
# attempt, fail fast, degrade immediately.
_client = OpenAI(api_key=settings.openai_api_key, timeout=20.0, max_retries=0)

# Hand-written rather than derived from QueryIR.model_json_schema(): keeping
# one schema that mirrors query_ir.QueryIR field-for-field (see
# prompt_builder.IR_SCHEMA, which documents the same shape for the model to
# read) is simpler than reshaping pydantic's generated schema on every call.
# Passed as OpenAI's Structured Outputs `response_format` as-is — every
# object (including nested ones) already has additionalProperties: False
# and every property already listed in `required`, which is what OpenAI's
# strict mode needs.
QUERY_IR_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "subject_level", "subjects", "metric", "filters", "time_range", "sort", "limit", "group_by", "flat", "overall_confidence", "intent_confidence"],
    "properties": {
        "intent": {"type": "string", "enum": ["leaderboard", "comparison", "lookup", "trend", "filtered_list", "breakdown", "clarify"]},
        "subject_level": {"type": "string", "enum": HIERARCHY_LEVELS},
        "subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "value", "match_confidence"],
                "properties": {
                    "type": {"type": "string", "enum": HIERARCHY_LEVELS},
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
            "required": ["mode", "period", "compare_to", "confidence"],
            "properties": {
                "mode": {"type": "string", "enum": ["snapshot", "compare"]},
                # DERIVED. Hardcoding the triple here is why the LLM could
                # not emit DAILY: with strict:True the grammar forbade a
                # value temporal_parser had recognised for weeks, so an
                # LLM-parsed "revenue today" was forced back to MTD —
                # finding F5, alive on the default path while the
                # rule-based path was already fixed.
                "period": {"type": "string", "enum": list(PERIODS)},
                "compare_to": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "confidence": {"type": "number"},
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
        "group_by": {"anyOf": [{"type": "string", "enum": HIERARCHY_LEVELS}, {"type": "null"}]},
        "flat": {"type": "boolean"},
        "overall_confidence": {"type": "number"},
        "intent_confidence": {"type": "number"},
    },
}


def call_llm_structured(prompt: str, schema: dict, schema_name: str) -> dict | None:
    """Schema-constrained structured output (Part 5.3): OpenAI's Structured
    Outputs (`response_format={"type": "json_schema", ...}`) grammar-
    constrains decoding to the given JSON schema, so a malformed shape
    becomes a provider-level error this function catches — not a
    `json.JSONDecodeError` or a QueryIR that silently fails pydantic
    validation downstream. `schema_name` (previously unused under Ollama,
    which has no equivalent naming requirement) is the schema's name in
    OpenAI's request — required by the API, has no effect on the output
    shape itself.

    Fails soft exactly like call_llm_json(): any error (bad key, no quota,
    network error, timeout, malformed output, a safety refusal) returns
    None so semantic_parser.py degrades to the rule-based plan_to_ir()
    path.
    """
    # Audit capture happens BEFORE the call (and outside the try) so a
    # prompt is recorded even when inference times out or is refused —
    # those runs degrade to the rule-based path, which is precisely the
    # kind of "why was this answer different?" the audit is here to
    # explain. No-op unless CHAT_AUDIT_DEBUG is on.
    # Hoisted only so the audit can record the LITERAL payload — same list,
    # same single user message, unchanged content.
    messages = [{"role": "user", "content": prompt}]
    audit.record_prompt(
        prompt, purpose=f"structured:{schema_name}", model=settings.openai_model, messages=messages
    )
    try:
        response = _client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
            max_completion_tokens=800,
        )
        raw = response.choices[0].message.content.strip()
        audit.record_llm_response(raw)
        return json.loads(raw)

    except json.JSONDecodeError:
        log.warning("LLM returned unparseable JSON despite schema constraint")
        return None

    except Exception:
        log.exception("Structured LLM call failed")
        return None


def call_llm_json(prompt: str) -> dict | None:
    """Schema-agnostic fallback for any caller without a strict JSON schema
    to enforce (e.g. narrative.py's phrasing-only polish). New code parsing
    a QueryIR should call call_llm_structured() with QUERY_IR_JSON_SCHEMA
    instead — see semantic_parser.py.

    Fails soft — returns None on any error (bad key, no quota, network
    error, timeout, malformed JSON) so the pipeline can degrade to the
    rule-based fallback rather than crash the chat endpoint or block on a
    retry loop.
    """
    messages = [{"role": "user", "content": prompt}]
    audit.record_prompt(prompt, purpose="json", model=settings.openai_model, messages=messages)
    try:
        response = _client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        audit.record_llm_response(raw)
        return json.loads(raw)

    except json.JSONDecodeError:
        log.warning("LLM returned unparseable JSON")
        return None

    except Exception:
        log.exception("LLM call failed")
        return None


def classify_with_llm(prompt: str) -> dict | None:
    """Backward-compat alias — see call_llm_json()."""
    return call_llm_json(prompt)


def create_embeddings(texts: list[str]) -> list[list[float]]:
    """RAW embeddings call — RAISES on any provider failure.

    Unlike every other function in this module, this one does not fail
    soft. Availability policy (classify the error, disable the subsystem,
    stop calling) lives in app/llm/embeddings.py, and it needs the
    exception to decide WHY it failed: swallowing it here would collapse
    "no quota" and "network blip" into an indistinguishable None, and
    both would then be retried forever.

    Callers should use embeddings.embed_texts(), not this."""
    if not texts:
        return []
    response = _client.embeddings.create(model=settings.openai_embedding_model, input=texts)
    return [d.embedding for d in response.data]
