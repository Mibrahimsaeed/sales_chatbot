from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = Field(alias="DATABASE_URL")
    google_service_account_path: str = Field(alias="GOOGLE_SERVICE_ACCOUNT")

    # Local Ollama server (llm_client.py) — replaces the OpenAI provider.
    # No API key: Ollama runs locally, so "unavailable" is a connection/
    # model-not-found error at call time, not a missing credential.
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_model: str = Field(default="qwen3:8b", alias="OLLAMA_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")

    ccmc_sheet_id: str = Field(alias="CCMC_SHEET_ID")
    biometric_sheet_id: str = Field(alias="BIOMETRIC_SHEET_ID")

    jwt_secret: str = Field(alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")

    sync_interval: int = Field(default=10, alias="SYNC_INTERVAL")
    sync_hour: int = Field(default=2, alias="SYNC_HOUR")
    sync_minute: int = Field(default=0, alias="SYNC_MINUTE")

    # "llm_first": the LLM parses every analytical query into a QueryIR
    # (rule-based layer kept for greetings/help/simple shortcuts and as the
    # fail-soft degrade path). "rules_first": the pre-inversion behavior —
    # rules primary, LLM only for compound-looking queries. Env-switchable
    # so the inversion can be rolled back without a deploy.
    nlu_mode: str = Field(default="llm_first", alias="NLU_MODE")
    # LLM phrasing polish on analytical replies (narrative layer); numbers
    # always come from SQL results, never the LLM. Off = templated replies.
    nlu_narrative: bool = Field(default=True, alias="NLU_NARRATIVE")
    # Embedding-based metric retrieval (Part 8) — a widening attempt for
    # paraphrased queries, tried after fuzzy synonym matching and before
    # giving up. Fail-soft like every other LLM call site, so this costs
    # nothing when the key/quota is unavailable; a separate flag from
    # nlu_mode so it can be killed independently via env var if it ever
    # misbehaves once quota is restored (unverified against the live API
    # as of this change).
    semantic_retrieval_enabled: bool = Field(default=True, alias="SEMANTIC_RETRIEVAL_ENABLED")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()


