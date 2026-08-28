# import json

# from openai import OpenAI

# from app.core import audit
# from app.core.config import settings
# from app.core.logger import get_logger
# from app.llm.hierarchy import HIERARCHY_LEVELS
# from app.llm.periods import PERIODS

# log = get_logger("llm.client")

# # Back on OpenAI's hosted API (was Ollama — see git history) since a local
# # model was too slow for interactive chat latency. Client construction is
# # cheap and doesn't connect eagerly; "unavailable" (bad key, no quota,
# # network error, timeout) surfaces as an exception at call time, caught by
# # each function below exactly like every other failure mode already was.
# #
# # max_retries=0: the SDK's default (2 retries, exponential backoff) is
# # actively counterproductive here — a 429 insufficient_quota or bad-key
# # error can never succeed on retry, so the default just triples the
# # latency of every failed call before this module's own fail-soft path
# # (return None -> rule-based degrade) ever gets a chance to run. One
# # attempt, fail fast, degrade immediately.
# _client = OpenAI(api_key=settings.openai_api_key, timeout=20.0, max_retries=0)

# # Hand-written rather than derived from QueryIR.model_json_schema(): keeping
# # one schema that mirrors query_ir.QueryIR field-for-field (see
# # prompt_builder.IR_SCHEMA, which documents the same shape for the model to
# # read) is simpler than reshaping pydantic's generated schema on every call.
# # Passed as OpenAI's Structured Outputs `response_format` as-is — every
# # object (including nested ones) already has additionalProperties: False
# # and every property already listed in `required`, which is what OpenAI's
# # strict mode needs.

# # A single filter leaf, reused by both the flat list and the tree below.
# _FILTER_LEAF_SCHEMA = {
#     "type": "object",
#     "additionalProperties": False,
#     "required": ["field", "operator", "value", "confidence"],
#     "properties": {
#         "field": {"type": "string"},
#         "operator": {"type": "string", "enum": ["=", "!=", ">", ">=", "<", "<=", "in"]},
#         "value": {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}]}}, {"type": "null"}]},
#         "confidence": {"type": "number"},
#     },
# }


# def _filter_group_schema(depth: int) -> dict:
#     """The AND/OR/NOT tree, expanded to a FIXED depth.

#     query_ir.FilterGroup is genuinely recursive, but OpenAI's strict mode
#     constrains decoding with a grammar built from this schema, and an
#     unbounded $ref cycle has no grammar. Inlining a bounded expansion
#     keeps the contract explicit: three levels is enough for every shape
#     the audit found unrepresentable ("A or B", "A and not (B or C)"), and
#     a deeper tree is refused by the schema rather than silently truncated
#     at parse time.
#     """
#     child = {"anyOf": [_FILTER_LEAF_SCHEMA]}
#     if depth > 1:
#         child = {"anyOf": [_FILTER_LEAF_SCHEMA, _filter_group_schema(depth - 1)]}
#     return {
#         "type": "object",
#         "additionalProperties": False,
#         "required": ["op", "children"],
#         "properties": {
#             "op": {"type": "string", "enum": ["and", "or", "not"]},
#             "children": {"type": "array", "items": child},
#         },
#     }


# _FILTER_TREE_DEPTH = 3

# # Operations the IR can actually express. DERIVED from the registry, so
# # an operation added there reaches the model without a second edit here —
# # and one the IR cannot hold is never offered to it.
# def _ir_operations() -> list[str]:
#     from app.llm.operations import IR_EXPRESSIBLE
#     return sorted(IR_EXPRESSIBLE)


# _IR_OPERATIONS = _ir_operations()

# QUERY_IR_JSON_SCHEMA = {
#     "type": "object",
#     "additionalProperties": False,
#     "required": ["operation", "intent", "subject_level", "subjects", "metric", "metrics", "filters", "filter_tree", "time_range", "sort", "limit", "group_by", "flat", "overall_confidence", "intent_confidence"],
#     "properties": {
#         # THE operation, from the one registry that declares them
#         # (app/llm/operations.py). `intent` below is the older, narrower
#         # name for the same idea and is kept so an existing consumer
#         # still reads what it always did.
#         "operation": {"anyOf": [{"type": "null"},
#                                 {"type": "string", "enum": list(_IR_OPERATIONS)}]},
#         "intent": {"type": "string", "enum": ["leaderboard", "comparison", "lookup", "trend", "filtered_list", "breakdown", "clarify"]},
#         "subject_level": {"type": "string", "enum": HIERARCHY_LEVELS},
#         "subjects": {
#             "type": "array",
#             "items": {
#                 "type": "object",
#                 "additionalProperties": False,
#                 "required": ["type", "value", "match_confidence", "metric"],
#                 "properties": {
#                     "type": {"type": "string", "enum": HIERARCHY_LEVELS},
#                     "value": {"type": "string"},
#                     "match_confidence": {"type": "number"},
#                     "metric": {
#                         "anyOf": [
#                             {"type": "null"},
#                             {
#                                 "type": "object",
#                                 "additionalProperties": False,
#                                 "required": ["key", "confidence"],
#                                 "properties": {"key": {"type": "string"}, "confidence": {"type": "number"}},
#                             },
#                         ]
#                     },
#                 },
#             },
#         },
#         "metric": {
#             "anyOf": [
#                 {"type": "null"},
#                 {
#                     "type": "object",
#                     "additionalProperties": False,
#                     "required": ["key", "confidence"],
#                     "properties": {"key": {"type": "string"}, "confidence": {"type": "number"}},
#                 },
#             ]
#         },
#         # EVERY measure the turn named, primary first. `metric` above stays
#         # the one the answer is ranked by; this is what stops the second
#         # measure of a two-measure question being dropped.
#         "metrics": {
#             "type": "array",
#             "items": {
#                 "type": "object",
#                 "additionalProperties": False,
#                 "required": ["key", "confidence"],
#                 "properties": {"key": {"type": "string"}, "confidence": {"type": "number"}},
#             },
#         },
#         "filters": {"type": "array", "items": _FILTER_LEAF_SCHEMA},
#         # The AND/OR/NOT structure, AND-combined with `filters` above. Null
#         # for every query that needs only conjunction, which is most of
#         # them — emit it only for a disjunction or an exclusion.
#         "filter_tree": {"anyOf": [{"type": "null"}, _filter_group_schema(_FILTER_TREE_DEPTH)]},
#         "time_range": {
#             "type": "object",
#             "additionalProperties": False,
#             "required": ["mode", "period", "compare_to", "confidence"],
#             "properties": {
#                 "mode": {"type": "string", "enum": ["snapshot", "compare"]},
#                 # DERIVED. Hardcoding the triple here is why the LLM could
#                 # not emit DAILY: with strict:True the grammar forbade a
#                 # value temporal_parser had recognised for weeks, so an
#                 # LLM-parsed "revenue today" was forced back to MTD —
#                 # finding F5, alive on the default path while the
#                 # rule-based path was already fixed.
#                 "period": {"type": "string", "enum": list(PERIODS)},
#                 "compare_to": {"anyOf": [{"type": "string"}, {"type": "null"}]},
#                 "confidence": {"type": "number"},
#             },
#         },
#         "sort": {
#             "type": "object",
#             "additionalProperties": False,
#             "required": ["metric", "direction"],
#             "properties": {
#                 "metric": {"anyOf": [{"type": "string"}, {"type": "null"}]},
#                 "direction": {"type": "string", "enum": ["asc", "desc"]},
#             },
#         },
#         "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
#         "group_by": {"anyOf": [{"type": "string", "enum": HIERARCHY_LEVELS}, {"type": "null"}]},
#         "flat": {"type": "boolean"},
#         "overall_confidence": {"type": "number"},
#         "intent_confidence": {"type": "number"},
#     },
# }


# def call_llm_structured(prompt: str, schema: dict, schema_name: str) -> dict | None:
#     """Schema-constrained structured output (Part 5.3): OpenAI's Structured
#     Outputs (`response_format={"type": "json_schema", ...}`) grammar-
#     constrains decoding to the given JSON schema, so a malformed shape
#     becomes a provider-level error this function catches — not a
#     `json.JSONDecodeError` or a QueryIR that silently fails pydantic
#     validation downstream. `schema_name` (previously unused under Ollama,
#     which has no equivalent naming requirement) is the schema's name in
#     OpenAI's request — required by the API, has no effect on the output
#     shape itself.

#     Fails soft exactly like call_llm_json(): any error (bad key, no quota,
#     network error, timeout, malformed output, a safety refusal) returns
#     None so semantic_parser.py degrades to the rule-based plan_to_ir()
#     path.
#     """
#     # Audit capture happens BEFORE the call (and outside the try) so a
#     # prompt is recorded even when inference times out or is refused —
#     # those runs degrade to the rule-based path, which is precisely the
#     # kind of "why was this answer different?" the audit is here to
#     # explain. No-op unless CHAT_AUDIT_DEBUG is on.
#     # Hoisted only so the audit can record the LITERAL payload — same list,
#     # same single user message, unchanged content.
#     messages = [{"role": "user", "content": prompt}]
#     audit.record_prompt(
#         prompt, purpose=f"structured:{schema_name}", model=settings.openai_model, messages=messages
#     )
#     try:
#         response = _client.chat.completions.create(
#             model=settings.openai_model,
#             messages=messages,
#             response_format={
#                 "type": "json_schema",
#                 "json_schema": {"name": schema_name, "strict": True, "schema": schema},
#             },
#             max_completion_tokens=800,
#         )
#         raw = response.choices[0].message.content.strip()
#         audit.record_llm_response(raw)
#         return json.loads(raw)

#     except json.JSONDecodeError:
#         log.warning("LLM returned unparseable JSON despite schema constraint")
#         return None

#     except Exception:
#         log.exception("Structured LLM call failed")
#         return None


# def call_llm_json(prompt: str) -> dict | None:
#     """Schema-agnostic fallback for any caller without a strict JSON schema
#     to enforce (e.g. narrative.py's phrasing-only polish). New code parsing
#     a QueryIR should call call_llm_structured() with QUERY_IR_JSON_SCHEMA
#     instead — see semantic_parser.py.

#     Fails soft — returns None on any error (bad key, no quota, network
#     error, timeout, malformed JSON) so the pipeline can degrade to the
#     rule-based fallback rather than crash the chat endpoint or block on a
#     retry loop.
#     """
#     messages = [{"role": "user", "content": prompt}]
#     audit.record_prompt(prompt, purpose="json", model=settings.openai_model, messages=messages)
#     try:
#         response = _client.chat.completions.create(
#             model=settings.openai_model,
#             messages=messages,
#             response_format={"type": "json_object"},
#             max_completion_tokens=600,
#         )
#         raw = response.choices[0].message.content.strip()
#         audit.record_llm_response(raw)
#         return json.loads(raw)

#     except json.JSONDecodeError:
#         log.warning("LLM returned unparseable JSON")
#         return None

#     except Exception:
#         log.exception("LLM call failed")
#         return None


# def classify_with_llm(prompt: str) -> dict | None:
#     """Backward-compat alias — see call_llm_json()."""
#     return call_llm_json(prompt)


# def create_embeddings(texts: list[str]) -> list[list[float]]:
#     """RAW embeddings call — RAISES on any provider failure.

#     Unlike every other function in this module, this one does not fail
#     soft. Availability policy (classify the error, disable the subsystem,
#     stop calling) lives in app/llm/embeddings.py, and it needs the
#     exception to decide WHY it failed: swallowing it here would collapse
#     "no quota" and "network blip" into an indistinguishable None, and
#     both would then be retried forever.

#     Callers should use embeddings.embed_texts(), not this."""
#     if not texts:
#         return []
#     response = _client.embeddings.create(model=settings.openai_embedding_model, input=texts)
#     return [d.embedding for d in response.data]



import json
import time

from ollama import Client

from app.core import audit
from app.core.config import settings
from app.core.logger import get_logger
from app.llm.hierarchy import HIERARCHY_LEVELS
from app.llm.periods import PERIODS

log = get_logger("llm.client")


# ============================================================================
# OLLAMA CLIENT
# ============================================================================
#
# Local reasoning model:
#
#   Ollama
#   └── qwen3:14b
#
# Default endpoint:
#
#   http://localhost:11434
#
# Unlike the previous OpenAI implementation, Ollama does not require:
#
#   OPENAI_API_KEY
#
# The model is identified only by its local model name.
#
# Example:
#
#   ollama pull qwen3:14b
#   ollama serve
#
# Configuration:
#
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_MODEL=qwen3:14b
#
# ============================================================================

_ollama = Client(
    host=settings.ollama_base_url,
    timeout=60.0,
)


def _ns_to_ms(value):
    """Nanoseconds -> milliseconds, or None.

    Every duration Ollama reports is an integer of NANOSECONDS
    (ollama._types.BaseGenerateResponse documents each one). None stays
    None: a field the provider did not send must be recorded as absent
    rather than as zero, which would read as "instant".
    """
    return None if value is None else round(value / 1_000_000, 1)


def _tokens_per_second(tokens, duration_ns):
    """Generation throughput, or None when it cannot be computed.

    Derived only from two figures the provider actually sent. Guarded
    against a zero duration, which Ollama reports for a response served
    without generating (an empty or fully-cached completion).
    """
    if not tokens or not duration_ns:
        return None
    return round(tokens / (duration_ns / 1_000_000_000), 1)


def _log_llm_call(*, purpose, duration_ms, response=None, success=True, error=None):
    """One structured line per provider call, success or failure.

    WHY AT THE BOUNDARY. The audit could not say whether latency came
    from prefill, generation, model loading or the network, because
    nothing timed the call — `tracing` records whole-request and per-SQL
    durations only. Timing here covers both call sites without either
    knowing about it.

    TOKEN COUNTS ARE THE PROVIDER'S, NEVER ESTIMATED. `prompt_eval_count`
    and `eval_count` are what the model actually processed; a
    character-length guess would be wrong by exactly the amount that
    matters when deciding whether the prompt is too big.

    `duration_ms` is measured here and always present. Everything else is
    Ollama's own metadata and is None whenever the provider omitted it —
    which is the whole state on a failed call.
    """
    fields = {
        "provider": settings.llm_provider,
        "model": settings.ollama_model,
        "purpose": purpose,
        "duration_ms": duration_ms,
        "prompt_tokens": getattr(response, "prompt_eval_count", None),
        "output_tokens": getattr(response, "eval_count", None),
        "load_duration_ms": _ns_to_ms(getattr(response, "load_duration", None)),
        "prompt_eval_duration_ms": _ns_to_ms(getattr(response, "prompt_eval_duration", None)),
        "eval_duration_ms": _ns_to_ms(getattr(response, "eval_duration", None)),
        "total_duration_ms": _ns_to_ms(getattr(response, "total_duration", None)),
        "prompt_tokens_per_second": _tokens_per_second(
            getattr(response, "prompt_eval_count", None),
            getattr(response, "prompt_eval_duration", None),
        ),
        "eval_tokens_per_second": _tokens_per_second(
            getattr(response, "eval_count", None),
            getattr(response, "eval_duration", None),
        ),
        "success": success,
    }
    if error is not None:
        fields["error"] = type(error).__name__

    log.info("llm_call " + " ".join(f"{k}={v}" for k, v in fields.items()))
    return fields


def _chat(*, messages: list[dict], fmt, purpose: str = "unknown"):
    """THE provider boundary — every inference call goes through here.

    ONE function talks to the model, so a provider swap touches this and
    nothing else. It also gives tests a stable seam: they patch `_chat`
    rather than reaching into a vendor SDK's own call shape.

    That reach is what the OpenAI->Ollama migration broke. Eight
    end-to-end suites disabled the LLM with

        monkeypatch.setattr(llm_client._client.chat.completions, "create", ...)

    which names `_client`, `.chat`, `.completions` and `.create` — four
    OpenAI SDK details, none of them this project's. Renaming the client
    to `_ollama` invalidated all four at once and produced 146 errors, in
    tests that were not about the provider at all.

    `fmt` is Ollama's structured-output argument: a JSON schema for
    grammar-constrained decoding, or the string "json" for free-form
    object output.

    `purpose` labels the call in the latency log. Optional with a default
    so the contract every existing caller and test fake relies on is
    unchanged — the two real call sites pass it.

    TIMED ON BOTH PATHS. A failure is a latency event too: a call that
    takes 60 seconds to time out is the most expensive thing this
    function does, and leaving it unlogged would hide it from exactly the
    measurement this instrumentation exists for. The exception is
    re-raised unchanged, so the fail-soft handling in each caller behaves
    as before.
    """
    started = time.perf_counter()

    # EXPLICIT, NOT INHERITED. Every one of these was previously unset, so
    # each call silently took an Ollama default — which is why the audit
    # could not say whether latency was the model or the configuration.
    # See config.py for how each value was derived.
    #
    # `think` and `keep_alive` are top-level chat() parameters in the
    # installed client (ollama 0.6.2); `num_ctx` and `num_predict` are
    # Options fields. Verified against the installed package rather than
    # assumed — an older client has no `think` at all.
    kwargs = {
        "model": settings.ollama_model,
        "messages": messages,
        "format": fmt,
        "keep_alive": settings.ollama_keep_alive,
        "options": {
            # Semantic parsing, not creative generation — the same value
            # both call sites used before they shared this function, now
            # configurable so a benchmark can vary it without a code edit.
            "temperature": settings.ollama_temperature,
            "num_ctx": settings.ollama_num_ctx,
            "num_predict": settings.ollama_num_predict,
        },
    }
    # None means "don't send it": a model that rejects the parameter can
    # be accommodated by config alone.
    if settings.ollama_think is not None:
        kwargs["think"] = settings.ollama_think

    try:
        response = _ollama.chat(**kwargs)
    except Exception as exc:
        _log_llm_call(
            purpose=purpose,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            success=False,
            error=exc,
        )
        raise

    _log_llm_call(
        purpose=purpose,
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
        response=response,
        success=True,
    )
    return response


# ============================================================================
# QUERY IR SCHEMA
# ============================================================================
#
# This schema mirrors QueryIR and is used as the structured-output contract
# sent to Ollama.
#
# IMPORTANT:
#
# Ollama supports structured output by accepting a JSON schema through
# `format=`.
#
# Unlike OpenAI Structured Outputs, Ollama does not provide the same strict
# API-level guarantee, so the returned JSON is additionally parsed and
# validated before it is allowed to enter the rest of the pipeline.
#
# Pipeline:
#
#   Qwen3
#      ↓
#   JSON
#      ↓
#   json.loads()
#      ↓
#   QueryIR / validator
#      ↓
#   compiler
#
# ============================================================================


_FILTER_LEAF_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "field",
        "operator",
        "value",
        "confidence",
    ],
    "properties": {
        "field": {
            "type": "string",
        },
        "operator": {
            "type": "string",
            "enum": [
                "=",
                "!=",
                ">",
                ">=",
                "<",
                "<=",
                "in",
            ],
        },
        "value": {
            "anyOf": [
                {
                    "type": "string",
                },
                {
                    "type": "number",
                },
                {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {
                                "type": "string",
                            },
                            {
                                "type": "number",
                            },
                        ]
                    },
                },
                {
                    "type": "null",
                },
            ]
        },
        "confidence": {
            "type": "number",
        },
    },
}


def _filter_group_schema(depth: int) -> dict:
    """
    Build a bounded recursive AND / OR / NOT filter tree.

    QueryIR.FilterGroup is recursive, but an unbounded recursive JSON schema
    is unnecessary for the current application.

    Three levels are sufficient for the complex boolean cases currently
    covered by the evaluation suite.

    Example:

        AND(
            OR(A, B),
            NOT(C)
        )

    can be represented without allowing an unbounded tree.
    """

    child = {
        "anyOf": [
            _FILTER_LEAF_SCHEMA,
        ]
    }

    if depth > 1:
        child = {
            "anyOf": [
                _FILTER_LEAF_SCHEMA,
                _filter_group_schema(depth - 1),
            ]
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "op",
            "children",
        ],
        "properties": {
            "op": {
                "type": "string",
                "enum": [
                    "and",
                    "or",
                    "not",
                ],
            },
            "children": {
                "type": "array",
                "items": child,
            },
        },
    }


_FILTER_TREE_DEPTH = 3


# ============================================================================
# IR OPERATIONS
# ============================================================================
#
# Derived from the central operation registry.
#
# This prevents the LLM schema from drifting away from operations.py.
#
# ============================================================================


def _ir_operations() -> list[str]:
    from app.llm.operations import IR_EXPRESSIBLE

    return sorted(IR_EXPRESSIBLE)


_IR_OPERATIONS = _ir_operations()


# ============================================================================
# QUERY IR JSON SCHEMA
# ============================================================================

QUERY_IR_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "operation",
        "intent",
        "subject_level",
        "subjects",
        "metric",
        "metrics",
        "filters",
        "filter_tree",
        "time_range",
        "sort",
        "limit",
        "group_by",
        "flat",
        "overall_confidence",
        "intent_confidence",
    ],
    "properties": {
        # ------------------------------------------------------------------
        # Operation
        # ------------------------------------------------------------------

        "operation": {
            "anyOf": [
                {
                    "type": "null",
                },
                {
                    "type": "string",
                    "enum": list(_IR_OPERATIONS),
                },
            ]
        },

        # ------------------------------------------------------------------
        # Legacy intent field
        # ------------------------------------------------------------------

        "intent": {
            "type": "string",
            "enum": [
                "leaderboard",
                "comparison",
                "lookup",
                "trend",
                "filtered_list",
                "breakdown",
                "clarify",
            ],
        },

        # ------------------------------------------------------------------
        # Subject level
        # ------------------------------------------------------------------

        "subject_level": {
            "type": "string",
            "enum": HIERARCHY_LEVELS,
        },

        # ------------------------------------------------------------------
        # Subjects
        # ------------------------------------------------------------------

        "subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "type",
                    "value",
                    "match_confidence",
                    "metric",
                ],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": HIERARCHY_LEVELS,
                    },
                    "value": {
                        "type": "string",
                    },
                    "match_confidence": {
                        "type": "number",
                    },
                    "metric": {
                        "anyOf": [
                            {
                                "type": "null",
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "key",
                                    "confidence",
                                ],
                                "properties": {
                                    "key": {
                                        "type": "string",
                                    },
                                    "confidence": {
                                        "type": "number",
                                    },
                                },
                            },
                        ]
                    },
                },
            },
        },

        # ------------------------------------------------------------------
        # Primary metric
        # ------------------------------------------------------------------

        "metric": {
            "anyOf": [
                {
                    "type": "null",
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "key",
                        "confidence",
                    ],
                    "properties": {
                        "key": {
                            "type": "string",
                        },
                        "confidence": {
                            "type": "number",
                        },
                    },
                },
            ]
        },

        # ------------------------------------------------------------------
        # All metrics
        # ------------------------------------------------------------------

        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "key",
                    "confidence",
                ],
                "properties": {
                    "key": {
                        "type": "string",
                    },
                    "confidence": {
                        "type": "number",
                    },
                },
            },
        },

        # ------------------------------------------------------------------
        # Flat filters
        # ------------------------------------------------------------------

        "filters": {
            "type": "array",
            "items": _FILTER_LEAF_SCHEMA,
        },

        # ------------------------------------------------------------------
        # Boolean filter tree
        # ------------------------------------------------------------------

        "filter_tree": {
            "anyOf": [
                {
                    "type": "null",
                },
                _filter_group_schema(_FILTER_TREE_DEPTH),
            ]
        },

        # ------------------------------------------------------------------
        # Time range
        # ------------------------------------------------------------------

        "time_range": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "mode",
                "period",
                "compare_to",
                "confidence",
            ],
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": [
                        "snapshot",
                        "compare",
                    ],
                },
                "period": {
                    "type": "string",
                    "enum": list(PERIODS),
                },
                "compare_to": {
                    "anyOf": [
                        {
                            "type": "string",
                        },
                        {
                            "type": "null",
                        },
                    ]
                },
                "confidence": {
                    "type": "number",
                },
            },
        },

        # ------------------------------------------------------------------
        # Sort
        # ------------------------------------------------------------------

        "sort": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "metric",
                "direction",
            ],
            "properties": {
                "metric": {
                    "anyOf": [
                        {
                            "type": "string",
                        },
                        {
                            "type": "null",
                        },
                    ]
                },
                "direction": {
                    "type": "string",
                    "enum": [
                        "asc",
                        "desc",
                    ],
                },
            },
        },

        # ------------------------------------------------------------------
        # Limit
        # ------------------------------------------------------------------

        "limit": {
            "anyOf": [
                {
                    "type": "integer",
                },
                {
                    "type": "null",
                },
            ]
        },

        # ------------------------------------------------------------------
        # Grouping
        # ------------------------------------------------------------------

        "group_by": {
            "anyOf": [
                {
                    "type": "string",
                    "enum": HIERARCHY_LEVELS,
                },
                {
                    "type": "null",
                },
            ]
        },

        # ------------------------------------------------------------------
        # Flat query
        # ------------------------------------------------------------------

        "flat": {
            "type": "boolean",
        },

        # ------------------------------------------------------------------
        # Confidence
        # ------------------------------------------------------------------

        "overall_confidence": {
            "type": "number",
        },

        "intent_confidence": {
            "type": "number",
        },
    },
}


# ============================================================================
# OLLAMA RESPONSE EXTRACTION
# ============================================================================


def _extract_message_content(response) -> str:
    """
    Extract the textual content from an Ollama response.

    Ollama's Python client normally returns:

        response.message.content

    but this helper keeps the rest of the client isolated from the response
    object's exact representation.
    """

    try:
        content = response.message.content
    except AttributeError:
        content = None

    if content is None:
        return ""

    return str(content).strip()


# ============================================================================
# STRUCTURED LLM CALL
# ============================================================================


def call_llm_structured(
    prompt: str,
    schema: dict,
    schema_name: str,
) -> dict | None:
    """
    Call the local Ollama model and request JSON constrained by `schema`.

    This replaces the previous OpenAI Structured Outputs implementation.

    OpenAI previously used:

        response_format={
            "type": "json_schema",
            ...
        }

    Ollama instead receives:

        format=<JSON schema>

    The result is then parsed locally.

    IMPORTANT SAFETY PROPERTY:

        provider failure
            ↓
        None
            ↓
        semantic_parser fallback
            ↓
        rule-based degradation / refusal

    Therefore this function never allows provider errors to crash the
    analytical chat pipeline.

    `schema_name` is retained because existing callers and audit logging
    depend on it. Ollama itself does not require the OpenAI-style schema name.
    """

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    # ----------------------------------------------------------------------
    # Audit BEFORE inference
    # ----------------------------------------------------------------------
    #
    # This is intentionally outside the try block.
    #
    # If Ollama times out, crashes, or returns malformed JSON, we still want
    # the prompt captured so the audit can explain why the request degraded.
    #

    audit.record_prompt(
        prompt,
        purpose=f"structured:{schema_name}",
        model=settings.ollama_model,
        messages=messages,
    )

    try:
        log.debug(
            "Calling Ollama structured model=%s schema=%s",
            settings.ollama_model,
            schema_name,
        )

        response = _chat(messages=messages, fmt=schema,
                         purpose=f"structured:{schema_name}")

        raw = _extract_message_content(response)

        if not raw:
            log.warning(
                "Ollama returned an empty response for structured schema=%s",
                schema_name,
            )
            return None

        audit.record_llm_response(raw)

        # ------------------------------------------------------------------
        # Parse JSON
        # ------------------------------------------------------------------

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(
                "Ollama returned unparseable JSON for schema=%s: %r",
                schema_name,
                raw[:500],
            )
            return None

        # ------------------------------------------------------------------
        # Basic structural sanity check
        # ------------------------------------------------------------------

        if not isinstance(parsed, dict):
            log.warning(
                "Ollama structured response was not a JSON object: %s",
                type(parsed).__name__,
            )
            return None

        return parsed

    except Exception:
        log.exception(
            "Ollama structured LLM call failed "
            "(model=%s, schema=%s)",
            settings.ollama_model,
            schema_name,
        )
        return None


# ============================================================================
# GENERIC JSON CALL
# ============================================================================


def call_llm_json(prompt: str) -> dict | None:
    """
    Generic JSON-producing Ollama call.

    Used by callers such as narrative.py that do not provide a strict
    QueryIR schema.

    Unlike call_llm_structured(), this uses Ollama's generic JSON mode:

        format="json"

    rather than the QueryIR JSON schema.

    The result is still parsed locally and failures return None.
    """

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    audit.record_prompt(
        prompt,
        purpose="json",
        model=settings.ollama_model,
        messages=messages,
    )

    try:
        response = _chat(messages=messages, fmt="json", purpose="narrative")

        raw = _extract_message_content(response)

        if not raw:
            log.warning("Ollama returned an empty JSON response")
            return None

        audit.record_llm_response(raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(
                "Ollama returned unparseable JSON: %r",
                raw[:500],
            )
            return None

        if not isinstance(parsed, dict):
            log.warning(
                "Ollama JSON response was not an object: %s",
                type(parsed).__name__,
            )
            return None

        return parsed

    except Exception:
        log.exception("Ollama JSON call failed")
        return None


# ============================================================================
# BACKWARD-COMPATIBILITY ALIAS
# ============================================================================


def classify_with_llm(prompt: str) -> dict | None:
    """
    Backward-compatible alias.

    Existing callers can continue using classify_with_llm() without knowing
    that the provider has moved from OpenAI to Ollama.
    """

    return call_llm_json(prompt)


# ============================================================================
# EMBEDDINGS
# ============================================================================
#
# EMBEDDINGS — the configured provider, and OFF until one is named.
#
# This was the migration's second casualty. It kept an OpenAI
# implementation reading `settings.openai_api_key`, a field config.py no
# longer declares, so every call raised AttributeError. embeddings.py
# caught it and disabled the tier (fail-soft worked), but the project was
# paying a guaranteed failure per process for a subsystem that could not
# work.
#
# It now uses the SAME provider as inference. There is no local embedding
# model configured by default, so `ollama_embedding_model` is empty and
# this raises a clear, catchable error — which embeddings.py already
# handles by disabling the tier once and moving on. Set
# OLLAMA_EMBEDDING_MODEL (and pull the model) to turn it on.
#
# ============================================================================


class EmbeddingsNotConfigured(RuntimeError):
    """No embedding model is configured for the active provider.

    A distinct type so embeddings.classify_error() can tell "not set up"
    apart from "the provider broke", and so the log line says which one.
    """


def create_embeddings(texts: list[str]) -> list[list[float]]:
    """RAW embeddings call, on the same provider as inference.

    Availability and error handling continue to live in embeddings.py —
    this raises, that decides what to do about it. Raising rather than
    returning None keeps the existing contract: embeddings.embed() treats
    an exception as the signal to disable the tier for the process.
    """
    if not texts:
        return []

    model = (settings.ollama_embedding_model or "").strip()
    if not model:
        raise EmbeddingsNotConfigured(
            "OLLAMA_EMBEDDING_MODEL is not set — embedding-based entity "
            "linking and semantic retrieval stay disabled until it names a "
            "model pulled into Ollama."
        )

    response = _ollama.embed(model=model, input=texts)

    # The client returns an EmbedResponse; `embeddings` is the list of
    # vectors, one per input, in order.
    vectors = getattr(response, "embeddings", None)
    if vectors is None and isinstance(response, dict):
        vectors = response.get("embeddings")
    return [list(v) for v in (vectors or [])]
