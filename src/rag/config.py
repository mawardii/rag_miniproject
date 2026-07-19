"""Central configuration.

Every knob that affects pipeline behavior lives here and nowhere else, so the
pipeline we evaluate offline is byte-for-byte the pipeline we serve. Values come
from environment variables / .env and fall back to the defaults below.
"""

from pathlib import Path
from typing import Literal

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
    # deepseek-chat alias is deprecated 2026-07-24 per official docs
    chat_model: str = "deepseek-v4-flash"
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

    # Retrieval strategy — every mode is env-switchable (RETRIEVAL_MODE=dense rag-eval)
    # so configurations can be compared on the eval harness without code changes.
    # Defaults are corpus-dependent and chosen by rag-eval, never by vibes.
    # SEC-filings corpus (2026-07-18, 69 FinanceBench questions): dense 0.93 hit@4 /
    # 0.81 MRR beat hybrid (0.86/0.77), hybrid+rerank (0.90/0.77), dense+rerank
    # (0.91/0.83) — BM25's keyword votes drag in other companies' filings on a
    # vocabulary-homogeneous corpus. (Old road-damage paper corpus: hybrid+rerank won.)
    retrieval_mode: Literal["dense", "hybrid"] = "dense"
    rerank_enabled: bool = False
    fetch_k: int = 20  # candidates fetched before rerank/truncation to top_k


def get_settings() -> Settings:
    return Settings()
