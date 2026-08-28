"""
Embedding availability layer — graceful degradation when the embedding
provider is unusable.

THE PROBLEM: embeddings are an OPTIONAL widening step. Semantic entity
linking (entity_linker.py) and semantic metric retrieval
(semantic_retrieval.py) both sit BEHIND exact and fuzzy matching — they
only run when the deterministic tiers found nothing. So an unusable
provider should cost the chatbot a small amount of recall on paraphrased
queries and nothing else.

In practice it cost far more: every failed call logged a full traceback,
each index build retried on a timer, and a single chat message could
trigger a dozen doomed API round trips — turning "slightly worse recall"
into multi-second latency and unreadable logs.

THE POLICY, deliberately simple:

  probe once -> on ANY failure, mark unavailable and STOP CALLING.

Re-enabling requires an application restart or an explicit rebuild()
call. There is no timer, no backoff, no half-open state. That is a real
tradeoff — a transient blip disables embeddings until someone acts —
and it is the right one here: the degraded mode is nearly as good (exact
+ fuzzy + WID resolution still answer everything the deterministic path
covers), while the retrying mode is actively harmful. A cheap, always-
available fallback is exactly the case where failing fast beats retrying.

`reason` is a stable machine-readable slug ("insufficient_quota",
"invalid_api_key", "connection_error", ...) so the health endpoint can be
alerted on without parsing prose.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from app.core.config import settings
from app.core.logger import get_logger
from app.llm import llm_client

log = get_logger("llm.embeddings")

# The provider these slugs describe. Reads the configured value rather
# than naming a vendor: `llm_provider` was declared in config.py during
# the Ollama migration and then read by nothing, so the health endpoint
# went on reporting "openai" after the switch.
PROVIDER = settings.llm_provider

# Stable slugs. Kept as constants so the health endpoint, tests, and any
# alerting rule all reference the same strings.
REASON_QUOTA = "insufficient_quota"
REASON_RATE_LIMIT = "rate_limited"
REASON_AUTH = "invalid_api_key"
REASON_PERMISSION = "permission_denied"
REASON_CONNECTION = "connection_error"
REASON_TIMEOUT = "timeout"
REASON_API_ERROR = "api_error"
REASON_UNEXPECTED = "unexpected_error"
REASON_DISABLED = "disabled_by_config"
# No embedding model named for the active provider — a setup state, not a
# provider fault, and worth its own slug so the log says which.
REASON_NOT_CONFIGURED = "not_configured"
REASON_EMPTY = "empty_response"


@dataclass
class EmbeddingStatus:
    enabled: bool          # config flag (ENTITY_LINKING_ENABLED / SEMANTIC_RETRIEVAL_ENABLED)
    ready: bool            # provider actually usable
    provider: str
    reason: str | None
    checked_at: float | None

    def to_dict(self) -> dict:
        return asdict(self)


# ready is tri-state: None = never probed, True = working, False = given up.
_state: dict = {"ready": None, "reason": None, "checked_at": None}


def _config_enabled() -> bool:
    """Embeddings serve two independent features; the subsystem is worth
    calling if EITHER wants it."""
    return bool(settings.entity_linking_enabled or settings.semantic_retrieval_enabled)


def status() -> EmbeddingStatus:
    enabled = _config_enabled()
    if not enabled:
        return EmbeddingStatus(
            enabled=False, ready=False, provider=PROVIDER,
            reason=REASON_DISABLED, checked_at=_state["checked_at"],
        )
    return EmbeddingStatus(
        enabled=True,
        ready=bool(_state["ready"]),
        provider=PROVIDER,
        reason=_state["reason"],
        checked_at=_state["checked_at"],
    )


def is_available() -> bool:
    """True only when embeddings are configured AND have not been given
    up on. `ready is None` (never probed) counts as available so the
    first real call gets a chance to succeed."""
    return _config_enabled() and _state["ready"] is not False


def _disable(reason: str) -> None:
    """Mark unavailable and say so ONCE. Repeating the warning on every
    subsequent call is the log spam this module exists to remove — after
    this, callers short-circuit before reaching the provider at all."""
    already_disabled = _state["ready"] is False
    _state["ready"] = False
    _state["reason"] = reason
    _state["checked_at"] = time.time()
    if not already_disabled:
        log.warning(
            "Embeddings unavailable (%s). Semantic entity search disabled. "
            "Falling back to exact/fuzzy resolver. Restart or call embeddings.rebuild() to retry.",
            reason,
        )


def _mark_ready() -> None:
    if _state["ready"] is not True:
        log.info("Embeddings available (%s)", PROVIDER)
    _state["ready"] = True
    _state["reason"] = None
    _state["checked_at"] = time.time()


def classify_error(exc: BaseException) -> str:
    """Map a provider exception to a stable slug.

    Imported lazily and defensively: the openai package's exception
    hierarchy is not something to hard-depend on at module import, and a
    classification failure must never be what takes the subsystem down."""
    # Checked before any provider's exception hierarchy: this one is
    # raised by llm_client itself and means "nobody configured a model",
    # which no provider-specific branch below would recognise.
    if isinstance(exc, llm_client.EmbeddingsNotConfigured):
        return REASON_NOT_CONFIGURED

    # Provider-agnostic transport failures. The Ollama client raises
    # httpx errors directly rather than wrapping them, so a local daemon
    # that is simply not running must classify as a connection error and
    # not fall through to "unexpected".
    name = type(exc).__name__
    if "Timeout" in name:
        return REASON_TIMEOUT
    if "Connect" in name:
        return REASON_CONNECTION

    # The OpenAI hierarchy is still mapped: these slugs remain correct if
    # embeddings are pointed back at it, and the mapping is what
    # test_embeddings_degradation.py pins. Imported lazily and
    # defensively — a classification failure must never be what takes the
    # subsystem down.
    try:
        import openai
    except Exception:  # pragma: no cover — openai is an optional dependency now
        return REASON_UNEXPECTED

    # Quota exhaustion arrives as a RateLimitError whose body carries
    # code="insufficient_quota". The two are worth distinguishing: a rate
    # limit is transient and worth a restart-retry, an exhausted quota
    # needs a human to add credit.
    if isinstance(exc, openai.RateLimitError):
        return REASON_QUOTA if _is_quota_error(exc) else REASON_RATE_LIMIT
    if isinstance(exc, openai.AuthenticationError):
        return REASON_AUTH
    if isinstance(exc, openai.PermissionDeniedError):
        return REASON_PERMISSION
    if isinstance(exc, openai.APITimeoutError):
        return REASON_TIMEOUT
    if isinstance(exc, openai.APIConnectionError):
        return REASON_CONNECTION
    if isinstance(exc, openai.APIError):
        return REASON_API_ERROR
    return REASON_UNEXPECTED


def _is_quota_error(exc: BaseException) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("code") == REASON_QUOTA:
            return True
    return REASON_QUOTA in str(exc)


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Embed `texts`, or return None if embeddings are unusable.

    None is the "no semantic tier available" signal every caller already
    handles by falling back to exact/fuzzy matching — the contract is
    unchanged from before this module existed. What changed is that after
    the first failure this returns None WITHOUT calling the provider, so
    a broken key costs one failed request per process instead of one per
    query."""
    if not texts:
        return []
    if not is_available():
        return None

    try:
        vectors = llm_client.create_embeddings(texts)
    except Exception as exc:
        reason = classify_error(exc)
        # Full detail only at DEBUG — the WARNING in _disable() carries
        # the actionable part, and a 400-line traceback per query was the
        # reported symptom.
        log.debug("Embedding call failed (%s)", reason, exc_info=True)
        _disable(reason)
        return None

    if not vectors:
        _disable(REASON_EMPTY)
        return None

    _mark_ready()
    return vectors


def probe(force: bool = False) -> EmbeddingStatus:
    """One-shot availability check, for application startup.

    Cheap by design (a single short input). `force` re-probes even after
    a previous give-up — that is what rebuild() uses."""
    if not _config_enabled():
        return status()
    if force:
        _state["ready"] = None
        _state["reason"] = None
    if _state["ready"] is False:
        return status()

    embed_texts(["healthcheck"])
    return status()


def rebuild() -> EmbeddingStatus:
    """Explicit operator-triggered retry — the only way back to enabled
    short of a restart (see the module docstring on why there is no
    timer). Re-probes the provider and, if it comes back, rebuilds the
    entity/exemplar indexes lazily on next use."""
    log.info("Embedding rebuild requested — re-probing provider")
    return probe(force=True)


def _reset_for_tests() -> None:
    _state["ready"] = None
    _state["reason"] = None
    _state["checked_at"] = None
