from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = Field(alias="DATABASE_URL")
    google_service_account_path: str = Field(alias="GOOGLE_SERVICE_ACCOUNT")

    # OpenAI (back from a local Ollama model — see git history — which was
    # too slow for interactive chat latency)
    # openai_api_key: str = Field(alias="OPENAI_API_KEY")
    # openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    # openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    # Which provider serves inference AND embeddings. Read by
    # embeddings.PROVIDER so the health endpoint reports the truth — it
    # was declared during the migration and then read by nothing, so
    # status kept saying "openai" after the switch.
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")

    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="qwen3:8b", alias="OLLAMA_MODEL")

    # EMBEDDINGS ARE OPT-IN. Empty means no embedding model is available,
    # which is the honest default: the migration left create_embeddings
    # calling OpenAI with a deleted API-key setting, so the tier could
    # only ever fail. Name a model pulled into Ollama (e.g.
    # "nomic-embed-text") and turn the two flags below back on to enable
    # semantic entity linking and metric retrieval.
    ollama_embedding_model: str = Field(default="", alias="OLLAMA_EMBEDDING_MODEL")

    # ---- Ollama inference options ----
    #
    # All four were previously unset, so every call silently inherited
    # Ollama's defaults. Making them explicit is what lets a change be
    # measured rather than guessed at; the values below are reasoned from
    # the measured prompt and output sizes, not picked round.

    # REASONING OFF. qwen3 is a hybrid-reasoning model and thinks by
    # default, emitting a reasoning block before the answer. Both calls
    # this project makes are constrained transformations, not open
    # problems: the structured call's output shape is already enforced by
    # a JSON grammar, and the narrative call is copy-editing whose result
    # is rejected outright if it introduces a number. Thinking tokens are
    # therefore pure added latency on the interactive path.
    #
    # None means "do not send the parameter at all" — an escape hatch for
    # a model that rejects it, without a code change.
    # The installed client (ollama 0.6.2) types this as
    # `bool | Literal["low","medium","high"] | None` and takes it as a
    # TOP-LEVEL chat() argument, not an Options field — verified against
    # inspect.signature(ollama.chat) rather than assumed. The effort
    # levels are accepted here so a model that supports graded reasoning
    # can be tried from config; qwen3 takes the boolean form.
    ollama_think: bool | Literal["low", "medium", "high"] | None = Field(
        default=False, alias="OLLAMA_THINK"
    )

    # HOLD THE MODEL BETWEEN QUESTIONS. Ollama unloads after ~5 minutes
    # idle, so the first question after any pause pays a full model load —
    # which reads to a user as "the model is slow" rather than "the model
    # was evicted". A chat session has gaps of exactly that size. Accepts
    # Ollama's duration syntax; "-1" pins indefinitely, "0" unloads
    # immediately for a memory-constrained host.
    ollama_keep_alive: str = Field(default="30m", alias="OLLAMA_KEEP_ALIVE")

    # CONTEXT WINDOW, sized from the measured prompt.
    #
    # Re-derived after the Task 3 prompt reduction, by BUILDING the worst
    # case rather than estimating it: an unresolved metric phrase (so the
    # full synonym catalog is sent), a name from all three person
    # gazetteers (so none is retired), a prior IR and a full six-turn
    # conversation window. That prompt measures 28,548 chars ~= 7,137
    # tokens at chars/4. chars/4 UNDER-counts dense JSON and proper names,
    # so applying a conservative 1.35x gives ~9,634, and ~10,402 with the
    # num_predict output budget below.
    #
    # 16384 holds that with 1.57x headroom, and leaves room for the
    # gazetteers to grow as the org does. It must NOT be 4096: the user's
    # question sits near the END of the prompt (build_ir_prompt appends it
    # after the examples), so a window that truncates loses the question
    # itself and the model confidently answers something else.
    ollama_num_ctx: int = Field(default=16384, alias="OLLAMA_NUM_CTX")

    # OUTPUT CEILING — a runaway guard, not a target. Measured, not
    # guessed: a maximal IR that still passes QueryIR validation
    # (comparison, two subjects with their own metric bindings, three
    # metrics, three flat filters AND a nested or/and filter tree,
    # compare-mode time range, group_by) serialises to 1,390 chars ~= 347
    # tokens. The narrative reply is 60-120. 768 is 2.2x the largest real
    # structured output, so it bounds a runaway without ever truncating a
    # legitimate one.
    #
    # CAUTION: on Ollama, reasoning tokens count against this budget too.
    # This ceiling is sized for ollama_think=False. Enabling thinking
    # without raising it can truncate a valid IR mid-object.
    ollama_num_predict: int = Field(default=768, alias="OLLAMA_NUM_PREDICT")

    # DECODING TEMPERATURE. 0.0 is not a tuning choice here, it is the
    # correctness requirement: both call sites are deterministic
    # transforms — text -> QueryIR under a JSON grammar, and a copy-edit
    # whose output is rejected if it introduces a number. Sampling adds
    # variance to a task with one right answer, and makes a wrong parse
    # unreproducible. Exposed so a benchmark can vary it deliberately;
    # the default preserves exactly the value both call sites already used.
    ollama_temperature: float = Field(default=0.0, alias="OLLAMA_TEMPERATURE")
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
    # Default OFF: this tier calls embeddings, and no embedding model is
    # configured by default (see ollama_embedding_model). Leaving it on
    # bought one guaranteed failed call per process for a subsystem that
    # could not work. Set OLLAMA_EMBEDDING_MODEL, then re-enable.
    semantic_retrieval_enabled: bool = Field(
        default=False,
        alias="SEMANTIC_RETRIEVAL_ENABLED",
    )

    # Embedding-based entity linking (Part 9) — the same widening idea as
    # semantic_retrieval_enabled, applied to advisor/team/company/office/
    # portfolio_lead/management_lead/unit_head/zonal_head/business_center
    # names instead of metrics. A separate flag so either can be killed
    # independently via env var.
    # Default OFF for the same reason as semantic_retrieval_enabled.
    entity_linking_enabled: bool = Field(
        default=False,
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
    # Conversation window: how many recent TURNS (one turn = the user
    # message plus the assistant reply) are shown to the LLM alongside the
    # structured prior IR, and the character ceiling on that block.
    #
    # Two limits rather than one because they guard different failures. The
    # turn count keeps stale topics out — six messages back is usually a
    # different question. The character cap keeps a single huge reply (a
    # 15-row leaderboard) from crowding out the schema and the gazetteer
    # that the prompt actually needs to parse against.
    conversation_window_turns: int = Field(default=3, alias="CONVERSATION_WINDOW_TURNS")
    conversation_window_chars: int = Field(default=1200, alias="CONVERSATION_WINDOW_CHARS")
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
