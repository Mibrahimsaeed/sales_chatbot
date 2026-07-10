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

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
# from pydantic_settings import BaseSettings


# class Settings(BaseSettings):
#     database_url: str
#     google_service_account_path: str
#     ccmc_spreadsheet_id: str
#     biometric_spreadsheet_id: str
#     jwt_secret: str
#     jwt_algorithm: str = "HS256"
#     anthropic_api_key: str | None = None
#     sync_interval_minutes: int = 10
#     sync_hour: int = 2      # daily sync fires at this hour (server time)
#     sync_minute: int = 0

#     class Config:
#         env_file = ".env"


# settings = Settings()