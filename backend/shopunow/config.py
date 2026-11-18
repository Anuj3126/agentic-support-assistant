from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
DATA_DIR = PACKAGE_DIR / "data"
CHROMA_DIR = PACKAGE_DIR / "chroma_db"
BUSINESS_DB_PATH = PACKAGE_DIR / "shopunow.sqlite"


class Settings(BaseSettings):
    # An absolute path, not a bare ".env" pydantic-settings otherwise resolves it relative
    model_config = SettingsConfigDict(env_file=str(BACKEND_DIR / ".env"), extra="ignore")

    google_api_key: str = ""
    tavily_api_key: str = ""

    chat_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-001"

    rag_top_k: int = 4
    max_reflections: int = 2


settings = Settings()
