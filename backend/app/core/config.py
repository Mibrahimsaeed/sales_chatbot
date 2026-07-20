from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = Field(alias="DATABASE_URL")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    google_service_account_path: str = Field(alias="GOOGLE_SERVICE_ACCOUNT")

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

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()


