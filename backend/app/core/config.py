from pydantic_settings import BaseSettings
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    DATABASE_URL: str
    OPENAI_API_KEY: str
    GOOGLE_SERVICE_ACCOUNT: str
    CCMC_SHEET_ID: str
    BIOMETRIC_SHEET_ID: str
    SYNC_INTERVAL: int = 10

    class Config:
        env_file = BASE_DIR / ".env"


settings = Settings()


