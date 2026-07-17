"""Central configuration.

Every knob that affects pipeline behavior lives here and nowhere else, so the
pipeline we evaluate offline is byte-for-byte the pipeline we serve. Values come
from environment variables / .env and fall back to the defaults below.
"""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor paths to the repo root so behavior does not depend on the caller's cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets — SecretStr keeps them out of repr()/logs; unwrap only at the client boundary.
    deepseek_api_key: SecretStr
    openai_api_key: SecretStr

    # Generation
    chat_model: str = "deepseek-chat"
    chat_base_url: str = "https://api.deepseek.com"
    temperature: float = 0.2
    max_tokens: int = 1000

    # Embeddings
    embedding_model: str = "text-embedding-3-small"

    # Ingestion / retrieval — changing any of these requires an index rebuild
    # (enforced automatically via the index fingerprint, see retrieval/vectorstore.py).
    docs_dir: Path = PROJECT_ROOT / "docs"
    index_dir: Path = PROJECT_ROOT / "faiss_index"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4


def get_settings() -> Settings:
    return Settings()
