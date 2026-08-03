from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = Field(alias="DATABASE_URL")
    google_service_account_path: str = Field(alias="GOOGLE_SERVICE_ACCOUNT")

    # OpenAI (back from a local Ollama model — see git history — which was
    # too slow for interactive chat latency)
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")

    ccmc_sheet_id: str = Field(alias="CCMC_SHEET_ID")
    biometric_sheet_id: str = Field(alias="BIOMETRIC_SHEET_ID")

    jwt_secret: str = Field(alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")

    # ---- ETL scheduling ----
    # "interval": fire every `sync_interval_minutes`. "daily": fire once at
    # sync_hour:sync_minute. Previously the scheduler ALWAYS used the daily
    # cron trigger and `sync_interval` was declared here but read by nothing
    # — so a deployment that set SYNC_INTERVAL=10 expecting 10-minute syncs
    # silently got one daily run instead. Now explicit either way.
    sync_mode: str = Field(default="interval", alias="SYNC_MODE")
    sync_interval_minutes: int = Field(default=10, alias="SYNC_INTERVAL")
    sync_hour: int = Field(default=2, alias="SYNC_HOUR")
    sync_minute: int = Field(default=0, alias="SYNC_MINUTE")

    # ---- ETL reliability ----
    # A sync run retries this many times on a transient failure (Google API
    # 5xx/timeout, brief DB blip) before recording a failed SyncLog row.
    # Sheets rate limits and network errors were previously fatal for the
    # whole run, leaving data stale until the next scheduled fire.
    sync_max_attempts: int = Field(default=3, alias="SYNC_MAX_ATTEMPTS")
    sync_retry_backoff_seconds: float = Field(default=5.0, alias="SYNC_RETRY_BACKOFF_SECONDS")

    # ---- Data freshness ----
    # Hours since the last SUCCESSFUL sync before data is considered stale
    # (warning) / critically stale. The chatbot surfaces staleness on
    # analytical answers rather than presenting old numbers as current —
    # see app/services/data_health_service.py.
    data_stale_after_hours: float = Field(default=24.0, alias="DATA_STALE_AFTER_HOURS")
    data_critical_after_hours: float = Field(default=72.0, alias="DATA_CRITICAL_AFTER_HOURS")

    # "llm_first": the LLM parses every analytical query into a QueryIR
    # (rule-based layer kept for greetings/help/simple shortcuts and as the
    # fail-soft degrade path). "rules_first": the pre-inversion behavior —
    # rules primary, LLM only for compound-looking queries. Env-switchable
    # so the inversion can be rolled back without a deploy.
    nlu_mode: str = Field(default="llm_first", alias="NLU_MODE")

    # LLM-powered PLANNER (app/llm/llm_planner.py). Off by default so the
    # rule-based planner stays authoritative until this has been A/B'd.
    # Flipping it on is safe in both directions: any planner failure —
    # provider down, invalid JSON, hallucinated metric — falls back to the
    # rule-based planner, so the worst case is today's behaviour.
    use_llm_planner: bool = Field(default=False, alias="USE_LLM_PLANNER")
    # LLM phrasing polish on analytical replies (narrative layer); numbers
    # always come from SQL results, never the LLM. Off = templated replies.
    nlu_narrative: bool = Field(default=True, alias="NLU_NARRATIVE")

    # Embedding-based metric retrieval (Part 8) — a widening attempt for
    # paraphrased queries, tried after fuzzy synonym matching and before
    # giving up. Fail-soft like every other LLM call site, so this costs
    # nothing when the key/quota is unavailable; a separate flag from
    # nlu_mode so it can be killed independently via env var if it ever
    # misbehaves.
    semantic_retrieval_enabled: bool = Field(
        default=True,
        alias="SEMANTIC_RETRIEVAL_ENABLED",
    )

    # Embedding-based entity linking (Part 9) — the same widening idea as
    # semantic_retrieval_enabled, applied to advisor/team/company/office/
    # portfolio_lead/management_lead/unit_head/zonal_head/business_center
    # names instead of metrics. A separate flag so either can be killed
    # independently via env var.
    entity_linking_enabled: bool = Field(
        default=True,
        alias="ENTITY_LINKING_ENABLED",
    )

    # Confidence-aware QueryIR execution gate (Part 10) — see
    # ir_validator.classify_confidence(). An IR with nothing unresolved
    # (`missing` empty) is "high" only if overall_confidence also clears
    # confidence_high_threshold; short of that it's "medium" (ask about the
    # specific uncertain slot) down to confidence_low_threshold, below which
    # it's "low" (never executed — the user is asked to rephrase instead of
    # being asked about one slot when the whole parse is shaky).
    confidence_high_threshold: float = Field(
        default=0.8,
        alias="CONFIDENCE_HIGH_THRESHOLD",
    )

    confidence_low_threshold: float = Field(
        default=0.4,
        alias="CONFIDENCE_LOW_THRESHOLD",
    )

    # ---- Relationship inference (app/llm/relation_resolver.py) ----
    # Resolves an entity referred to THROUGH another entity ("Waqar
    # Haider's team") instead of only entities named literally. OFF by
    # default: it is the first change in this pipeline that can turn a
    # working single-person lookup into a group query, so it ships dark
    # and is enabled deliberately.
    relation_inference_enabled: bool = Field(default=False, alias="RELATION_INFERENCE_ENABLED")
    # Comma-separated target levels the inference is allowed to act on.
    # Per-relation granularity (not one master switch) so a rollback can
    # be surgical — disabling "company" need not disable "team". M1 ships
    # the two relations already carried on AdvisorIdentity, which cost no
    # extra database read; the rest arrive with M3.
    relation_inference_levels: str = Field(
        default="team,company", alias="RELATION_INFERENCE_LEVELS"
    )

    @property
    def relation_inference_level_set(self) -> frozenset[str]:
        """Parsed form of relation_inference_levels. A string rather than
        a list field because env vars are strings and pydantic's list
        coercion expects JSON — "team,company" is what an operator will
        actually type."""
        return frozenset(
            part.strip() for part in (self.relation_inference_levels or "").split(",") if part.strip()
        )

    # ---- Chat audit log (app/core/audit.py) ----
    # Diagnostics only, OFF by default: writes one readable block per user
    # query — the query, the timestamp, every complete LLM prompt sent
    # while answering it, and the final response — to
    # backend/logs/chat_audit.log. Separate from core/tracing.py, which
    # emits one machine-readable JSON line and never stores prompt text.
    # Nothing about answering a query changes when this is on.
    chat_audit_debug: bool = Field(default=False, alias="CHAT_AUDIT_DEBUG")
    # Relative paths resolve against backend/, not the working directory.
    chat_audit_dir: str = Field(default="logs", alias="CHAT_AUDIT_DIR")
    # Echo each block to stdout as well as the file, for watching live.
    chat_audit_console: bool = Field(default=True, alias="CHAT_AUDIT_CONSOLE")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
