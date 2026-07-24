import json

import ollama

from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("llm.client")

# Local Ollama server — no API key, so there's no "key missing" precondition
# to gate on like the old OpenAI client had. Client construction is cheap
# and doesn't connect eagerly; "unavailable" (daemon down, model not
# pulled, timeout) surfaces as an exception at call time, caught by each
# function below exactly like every other failure mode already was.
_client = ollama.Client(host=settings.ollama_host, timeout=15.0)

# Ollama unloads a model from memory after its keep_alive window of
# inactivity (default 5m) — the next request then pays a ~15-30s reload
# cost. Chat traffic is bursty, not constant, so a generous window here
# (instead of Ollama's default) keeps qwen3 resident through normal gaps
# between messages; app startup (main.py) also warms it once at boot so
# the FIRST real request isn't the one that pays the cold-start cost.
_KEEP_ALIVE = "30m"

# Hand-written rather than derived from QueryIR.model_json_schema(): keeping
# one schema that mirrors query_ir.QueryIR field-for-field (see
# prompt_builder.IR_SCHEMA, which documents the same shape for the model to
# read) is simpler than reshaping pydantic's generated schema on every call.
# Passed to Ollama's `format` param as-is — Ollama's grammar-constrained
# decoding reads standard JSON Schema (type/properties/required/enum/anyOf),
# no OpenAI-specific wrapper needed.
QUERY_IR_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "subject_level", "subjects", "metric", "filters", "time_range", "sort", "limit", "group_by", "overall_confidence", "intent_confidence"],
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
            "required": ["mode", "period", "compare_to", "confidence"],
            "properties": {
                "mode": {"type": "string", "enum": ["snapshot", "compare"]},
                "period": {"type": "string", "enum": ["MTD", "YTD", "3M"]},
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
        "group_by": {"anyOf": [{"type": "string", "enum": ["advisor", "team", "company"]}, {"type": "null"}]},
        "overall_confidence": {"type": "number"},
        "intent_confidence": {"type": "number"},
    },
}


def call_llm_structured(prompt: str, schema: dict, schema_name: str) -> dict | None:
    """Schema-constrained structured output (Part 5.3): Ollama's `format`
    param grammar-constrains decoding to the given JSON schema, so a
    malformed shape becomes a provider-level error this function catches —
    not a `json.JSONDecodeError` or a QueryIR that silently fails pydantic
    validation downstream. `think=False` — this is a single-turn extraction
    task, not one that benefits from qwen3's visible reasoning, and skipping
    it cuts latency noticeably.

    Fails soft exactly like call_llm_json(): any error (daemon down, model
    not pulled, timeout, malformed output) returns None so semantic_parser.py
    degrades to the rule-based plan_to_ir() path. `schema_name` is unused —
    kept for call-site compatibility (it named the schema for OpenAI's
    strict-mode wrapper, which Ollama's format param doesn't need).
    """
    try:
        response = _client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            format=schema,
            think=False,
            options={"num_predict": 800},
            keep_alive=_KEEP_ALIVE,
        )
        return json.loads(response.message.content.strip())

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

    Fails soft — returns None on any error (daemon down, model not pulled,
    malformed JSON, timeout) so the pipeline can degrade to the rule-based
    fallback rather than crash the chat endpoint or block on a retry loop.
    """
    try:
        response = _client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            think=False,
            options={"num_predict": 600},
            keep_alive=_KEEP_ALIVE,
        )
        return json.loads(response.message.content.strip())

    except json.JSONDecodeError:
        log.warning("LLM returned unparseable JSON")
        return None

    except Exception:
        log.exception("LLM call failed")
        return None


def classify_with_llm(prompt: str) -> dict | None:
    """Backward-compat alias — see call_llm_json()."""
    return call_llm_json(prompt)


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Part 8: fail-soft embeddings primitive for semantic_retrieval.py,
    same contract as call_llm_structured() — any failure (daemon down,
    embedding model not pulled, timeout) returns None so the caller
    degrades to its next-best option instead of raising."""
    if not texts:
        return []

    try:
        response = _client.embed(model=settings.ollama_embedding_model, input=texts, keep_alive=_KEEP_ALIVE)
        return list(response.embeddings)

    except Exception:
        log.exception("Embedding call failed")
        return None
