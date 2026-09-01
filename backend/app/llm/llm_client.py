import json
import re
import time

from app.core import audit
from app.core.config import settings
from app.core.logger import get_logger
from app.llm.hierarchy import HIERARCHY_LEVELS
from app.llm.periods import PERIODS

log = get_logger("llm.client")


# ============================================================================
# PROVIDER — OpenAI
# ============================================================================
#
# ONE provider, ONE boundary. `_chat()` below is the only function that
# talks to a model; everything above it — the prompt builder, the QueryIR
# schema, the validator, the compiler, the formatter — is unchanged and
# knows nothing about which model answers.
#
#   OPENAI_API_KEY=...        (set in .env; never read, logged or exposed here)
#   OPENAI_MODEL=...
#   OPENAI_BASE_URL=          (optional: Azure, a gateway, a proxy)
#
# THE FALLBACK IS NOT A SECOND PROVIDER. When this provider is
# unreachable, misconfigured, or returns something unusable, every call
# here returns None and semantic_parser degrades to the deterministic
# rule-based planner (see nlu_pipeline._semantic_gaps). That is what
# keeps the system answering during an outage, and it is unchanged by
# there being one provider instead of two.
#
# ============================================================================


class ProviderNotConfigured(RuntimeError):
    """The selected provider cannot be reached as configured.

    A distinct type, for the reason EmbeddingsNotConfigured is one: the
    callers already degrade on any exception, but "you have not set a key"
    and "the provider is broken" want different log lines and different
    fixes. Raised rather than returned so the existing fail-soft handling
    in call_llm_structured / call_llm_json catches it unchanged and the
    pipeline degrades to the rule-based plan exactly as it does for a
    network error.
    """


# Built lazily and cached. NOT at import time, for two reasons: importing
# `openai` costs a noticeable amount of time at startup, and
# constructing a client with an empty key
# would raise during module import — turning a misconfiguration into an
# application that cannot boot, instead of one that degrades.
_openai_client = None


# The provider name, for the telemetry line and the embeddings health
# report. A CONSTANT rather than a setting: there is one provider, so a
# configurable value here could only ever disagree with what actually
# ran — the drift that had audit logs naming a model that had not served
# the call.
PROVIDER = "openai"


def _active_model() -> str:
    """The model this call will use.

    Read by the telemetry line and by the audit record, so both name the
    model that actually served the call rather than whichever setting the
    call site happened to reference.
    """
    return settings.openai_model


def _openai():
    """The OpenAI client, constructed on first use.

    Never logs, prints or returns the key. The only thing this asserts
    about it is whether it is empty.
    """
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    key = (settings.openai_api_key or "").strip()
    if not key:
        raise ProviderNotConfigured(
            "LLM_PROVIDER=openai but OPENAI_API_KEY is not set — the semantic "
            "parser will degrade to the rule-based plan until it is."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - packaging issue, not logic
        raise ProviderNotConfigured(
            "LLM_PROVIDER=openai but the `openai` package is not installed."
        ) from exc

    kwargs = {"api_key": key, "timeout": settings.openai_timeout_seconds}
    base_url = (settings.openai_base_url or "").strip()
    if base_url:
        kwargs["base_url"] = base_url

    _openai_client = OpenAI(**kwargs)
    return _openai_client


def _reset_provider_clients_for_tests() -> None:
    """Drop the cached OpenAI client so a test can change the key or
    base URL and have the next call rebuild against it."""
    global _openai_client
    _openai_client = None


# OpenAI requires the json_schema's `name` to match ^[a-zA-Z0-9_-]+$.
_SCHEMA_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_-]")


_EMPTY_METRICS = {
    "prompt_tokens": None,
    "output_tokens": None,
    "total_tokens": None,
}


def _response_metrics(response) -> dict:
    """The provider's OWN token counts.

    NOTHING IS ESTIMATED. A count the provider did not send stays None
    rather than becoming zero — "nothing generated" and "not reported"
    are different facts, and only one of them is ever true. A
    character-length guess would be wrong by exactly the amount that
    matters when deciding whether the prompt is too big.

    WHAT WENT AWAY WITH OLLAMA. The six nanosecond timing fields
    (load / prompt_eval / eval / total duration, and the two derived
    tokens-per-second figures) were Ollama's own metadata and have no
    equivalent here — OpenAI reports usage, not a breakdown of where the
    time went. They are removed rather than kept as permanently-None
    columns, which would read as "the provider stopped reporting" instead
    of "this provider never did".

    `duration_ms` on the log line is unaffected: it is measured at the
    boundary by `_chat`, not read from the response, so end-to-end call
    latency is still recorded on every call, success or failure.
    """
    usage = getattr(response, "usage", None) if response is not None else None
    if usage is None:
        return dict(_EMPTY_METRICS)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _log_llm_call(*, purpose, duration_ms, response=None, success=True, error=None):
    """One structured line per provider call, success or failure.

    WHY AT THE BOUNDARY. The audit could not say whether latency came
    from prefill, generation, model loading or the network, because
    nothing timed the call — `tracing` records whole-request and per-SQL
    durations only. Timing here covers both call sites without either
    knowing about it.

    TOKEN COUNTS ARE THE PROVIDER'S, NEVER ESTIMATED. They are what the
    model actually processed; a character-length guess would be wrong by
    exactly the amount that matters when deciding whether the prompt is
    too big.

    `duration_ms` is measured here and always present. The token counts
    come from the response and are None whenever there was no response —
    which is the whole state on a failed call.
    """
    fields = {
        "provider": PROVIDER,
        "model": _active_model(),
        "purpose": purpose,
        "duration_ms": duration_ms,
        **_response_metrics(response),
        "success": success,
    }
    if error is not None:
        fields["error"] = type(error).__name__

    log.info("llm_call " + " ".join(f"{k}={v}" for k, v in fields.items()))
    return fields


def _openai_response_format(fmt, purpose: str) -> dict:
    """Translate the boundary's `fmt` into OpenAI's `response_format`.

    `fmt` is the seam's existing structured-output argument and keeps its
    existing two shapes, so no caller changes:

        a JSON schema dict -> Structured Outputs, strict
        the string "json"  -> the older JSON-object mode

    STRICT IS THE POINT. With `strict: True` OpenAI constrains decoding to
    the schema, which is why QueryIR parsing is safe: an invalid level or
    period is not a validation failure downstream, it is unemittable.
    QUERY_IR_JSON_SCHEMA already satisfies every strict-mode rule — each
    object carries `additionalProperties: false` and lists all of its
    properties in `required`.

    The schema NAME is derived from `purpose` rather than added as a
    parameter. `_chat`'s signature is a contract that eight end-to-end
    suites and several test fakes bind to by keyword; widening it to carry
    a field only one provider needs would break fakes that are not about
    providers at all — the exact failure the boundary exists to prevent.
    `call_llm_structured` already encodes the schema name in `purpose` as
    "structured:<name>", so the information is present.
    """
    if isinstance(fmt, dict):
        name = purpose.split(":", 1)[1] if ":" in purpose else "response"
        name = _SCHEMA_NAME_SAFE.sub("_", name) or "response"
        return {
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": fmt},
        }
    return {"type": "json_object"}


def _chat_openai(*, messages: list[dict], fmt, purpose: str):
    """The OpenAI transport, via Chat Completions.

    `max_completion_tokens` is the current spelling of the output ceiling
    (`max_tokens` is deprecated for newer models) and exists as a runaway
    guard, not a target — see config.openai_max_output_tokens for how the
    value was derived.
    """
    return _openai().chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        response_format=_openai_response_format(fmt, purpose),
        temperature=settings.openai_temperature,
        max_completion_tokens=settings.openai_max_output_tokens,
    )


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

    `fmt` is the structured-output argument: a JSON schema for
    grammar-constrained decoding, or the string "json" for free-form
    object output. `_openai_response_format` translates it.

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

    try:
        response = _chat_openai(messages=messages, fmt=fmt, purpose=purpose)
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
# This schema mirrors QueryIR and is the structured-output contract sent
# to the provider.
#
# IMPORTANT:
#
# It is applied through OpenAI Structured Outputs with `strict: True`,
# which constrains decoding — an invalid level, period or operation is
# not caught downstream, it is unemittable.
#
# The returned JSON is STILL parsed and validated before it enters the
# rest of the pipeline. Grammar constraint guarantees the SHAPE; only
# ir_validator can say whether the values name real metrics, real people
# and answerable levels.
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


# Dispatch modes that lead NOWHERE — an operation routed to one of these
# cannot serve the user no matter how well the query was parsed. `trend`
# dispatches to "unsupported" because there is no historical snapshot to
# diff against, so offering it would let the model pick a value that is
# guaranteed a refusal.
#
# "clarification" IS NOT ONE OF THESE, and the distinction is the point.
# Asking the user a question is a legitimate outcome — for a genuinely
# ambiguous message it is the CORRECT outcome — and it is the only way
# the model can say "I don't know". While `clarification` sat here the
# grammar offered five operations, every one of which asserts an answer,
# so a model that was unsure had no representable way to say so and
# produced its best guess instead. A confident wrong answer replacing an
# honest question is the worst trade this pipeline can make.
#
# Named as a CATEGORY rather than as a list of the operations that
# currently fall into it, so an operation that later becomes answerable —
# or a new one that does not — is classified by what it does.
_NON_EXECUTABLE_DISPATCH = frozenset({"unsupported", "no_data"})


def _ir_operations() -> list[str]:
    """The operations the model may select.

    Two conditions, both read from the registry: the IR must be able to
    EXPRESS it, and the operation must lead somewhere — an answer, or a
    question put back to the user. `trend` satisfies the first and fails
    the second, so it would sit in the grammar as a value the model could
    pick and the system could never serve.

    THIS IS THE ONE DEFINITION. `prompt_builder._operation_union()` renders
    it for the model to read, so the set the prompt advertises and the set
    grammar-constrained decoding permits cannot disagree — they did, and
    the prompt named two operations (`trend`, `clarify_metric`) that were
    unemittable, which is an instruction the model can only fail to follow.
    """
    from app.llm.operations import OPERATIONS

    return sorted(
        name for name, op in OPERATIONS.items()
        if op.expressible_in_ir and op.dispatch_mode not in _NON_EXECUTABLE_DISPATCH
    )


def _ir_intents() -> list[str]:
    """The compatibility intents reachable from an offered operation.

    `intent` is the legacy second name for what `operation` says, and the
    registry already owns the correspondence (Operation.ir_intent). Deriving
    the enum from it means the model cannot name an intent that belongs to
    no operation it is allowed to choose — which is how `lookup` (rejected
    outright), `trend` (unexecutable), `clarify` and `breakdown` (both
    routed away from the compiler) came to be selectable at all.

    Not a second mapping: this reads the one the registry declares.
    """
    from app.llm.operations import OPERATIONS

    return sorted({
        OPERATIONS[name].ir_intent
        for name in _ir_operations()
        if OPERATIONS[name].ir_intent
    })


_IR_OPERATIONS = _ir_operations()
_IR_INTENTS = _ir_intents()


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
        "target_level",
        "subject_of",
        "relation",
        "overall_confidence",
        "intent_confidence",
    ],
    "properties": {
        # ------------------------------------------------------------------
        # Operation
        # ------------------------------------------------------------------

        # AUTHORITATIVE, AND THEREFORE REQUIRED.
        #
        # This used to permit null, and the prompt told the model to use
        # that: "operation may be null and the intent should communicate
        # the uncertainty". So on exactly the queries it was least sure
        # about, the model discarded the field every downstream consumer
        # reads (resolved_operation, the compiler, the response planner,
        # the dispatcher) and signalled through the legacy one instead —
        # which then decided the route. Every observed failure of "what is
        # X's <measure>?" arrived with operation=null.
        #
        # Uncertainty belongs in overall_confidence / intent_confidence,
        # which exist for it and which the validator already gates on.
        "operation": {
            "type": "string",
            "enum": list(_IR_OPERATIONS),
        },

        # ------------------------------------------------------------------
        # Legacy intent field
        # ------------------------------------------------------------------
        #
        # THE GRAMMAR MUST NOT OFFER A VALUE THE VALIDATOR ALWAYS REFUSES.
        # `lookup` and `trend` used to sit in this enum while
        # ir_validator._UNSUPPORTED_INTENTS rejected both outright, so the
        # model could satisfy the schema and still be guaranteed a
        # clarifying question. That is not a hypothetical: `lookup` is the
        # value a model naturally picks for "what is X's Y?" — the most
        # ordinary way there is to ask for a measure — and it was emitted,
        # and refused, on every single attempt at that phrasing.
        #
        # Removing them here makes the mistake unrepresentable rather than
        # merely detected. _UNSUPPORTED_INTENTS stays exactly as it is: it
        # still guards IRs built anywhere other than this schema (plan_to_ir,
        # conversation patches, tests). test_intent_contract.py asserts the
        # two can never contradict each other again.
        "intent": {
            "type": "string",
            "enum": list(_IR_INTENTS),
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
        # Hierarchy reads
        # ------------------------------------------------------------------
        #
        # Three fields carrying "enumerate THIS level beneath THAT
        # subject, directly or throughout". Without them the shape had no
        # representation, the hierarchy operations were plan-only, and the
        # model was never asked — routing fell to whether the sentence
        # happened to contain the word "directly".

        "target_level": {
            "anyOf": [
                {"type": "null"},
                {"type": "string", "enum": HIERARCHY_LEVELS},
            ]
        },
        "subject_of": {
            "anyOf": [
                {"type": "null"},
                {"type": "string", "enum": HIERARCHY_LEVELS},
            ]
        },
        "relation": {
            "type": "string",
            "enum": ["subtree", "direct"],
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
    """The textual content of a provider response:

        response.choices[0].message.content

    Keeps the rest of the client isolated from the SDK's response
    representation, so `call_llm_structured` and `call_llm_json` parse one
    string and never reach into a vendor object themselves.

    A REFUSAL IS NOT CONTENT. OpenAI can decline a request and return
    `message.refusal` with `content` null; returning "" for it routes into
    the callers' existing empty-response branch, which degrades to the
    rule-based plan. It is logged because a refusal and an outage need
    different fixes and would otherwise look identical.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)

    if content is None:
        if getattr(message, "refusal", None):
            log.warning("Provider refused the request: %s",
                        str(message.refusal)[:200])
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
    Call the model and request JSON constrained by `schema`.

    The schema is sent as OpenAI Structured Outputs:

        response_format={"type": "json_schema",
                         "json_schema": {"name": ..., "strict": True,
                                         "schema": <schema>}}

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

    `schema_name` names the schema in the request and labels the call in
    the audit log and the latency line.
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
    # If the provider times out, errors, or returns malformed JSON, we still want
    # the prompt captured so the audit can explain why the request degraded.
    #

    audit.record_prompt(
        prompt,
        purpose=f"structured:{schema_name}",
        model=_active_model(),
        messages=messages,
    )

    try:
        log.debug(
            "Calling %s structured model=%s schema=%s",
            PROVIDER,
            _active_model(),
            schema_name,
        )

        response = _chat(messages=messages, fmt=schema,
                         purpose=f"structured:{schema_name}")

        raw = _extract_message_content(response)

        if not raw:
            log.warning(
                "Provider returned an empty response for structured schema=%s",
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
                "Provider returned unparseable JSON for schema=%s: %r",
                schema_name,
                raw[:500],
            )
            return None

        # ------------------------------------------------------------------
        # Basic structural sanity check
        # ------------------------------------------------------------------

        if not isinstance(parsed, dict):
            log.warning(
                "Structured response was not a JSON object: %s",
                type(parsed).__name__,
            )
            return None

        return parsed

    except Exception:
        log.exception(
            "Structured LLM call failed "
            "(provider=%s, model=%s, schema=%s)",
            PROVIDER,
            _active_model(),
            schema_name,
        )
        return None


# ============================================================================
# GENERIC JSON CALL
# ============================================================================


def call_llm_json(prompt: str) -> dict | None:
    """
    Generic JSON-producing call.

    Used by callers such as narrative.py that do not provide a strict
    QueryIR schema.

    Unlike call_llm_structured(), this asks for the looser JSON-object
    mode:

        response_format={"type": "json_object"}

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
        model=_active_model(),
        messages=messages,
    )

    try:
        response = _chat(messages=messages, fmt="json", purpose="narrative")

        raw = _extract_message_content(response)

        if not raw:
            log.warning("Provider returned an empty JSON response")
            return None

        audit.record_llm_response(raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(
                "Provider returned unparseable JSON: %r",
                raw[:500],
            )
            return None

        if not isinstance(parsed, dict):
            log.warning(
                "JSON response was not an object: %s",
                type(parsed).__name__,
            )
            return None

        return parsed

    except Exception:
        log.exception("JSON LLM call failed")
        return None


# ============================================================================
# BACKWARD-COMPATIBILITY ALIAS
# ============================================================================


def classify_with_llm(prompt: str) -> dict | None:
    """
    Backward-compatible alias.

    Existing callers can continue using classify_with_llm() without
    knowing which provider answers.
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
# model configured by default, so `openai_embedding_model` is empty and
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

    model = (settings.openai_embedding_model or "").strip()
    if not model:
        raise EmbeddingsNotConfigured(
            "OPENAI_EMBEDDING_MODEL is not set — embedding-based entity "
            "linking and semantic retrieval stay disabled until it names a "
            "model."
        )

    # _openai() raises ProviderNotConfigured on a missing key, which
    # embeddings.py treats the same way it treats this module's own
    # not-configured error: disable the tier once, stop calling.
    response = _openai().embeddings.create(model=model, input=texts)
    # `data` is one item per input, in order.
    return [list(item.embedding) for item in response.data]

