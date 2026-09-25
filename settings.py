from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: SecretStr
    OPENAI_MODEL: str
    CAMERA_INDEX: int = 0
    ALERT_WEBHOOK_URL: AnyHttpUrl | None = None
    ALERT_COUNTDOWN_SECONDS: int = 15
    EVENT_DATA_DIR: Path = Path("data")


@lru_cache
def get_settings() -> Settings:
    return Settings()
