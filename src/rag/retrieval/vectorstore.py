"""FAISS index lifecycle: build, fingerprint, load-or-rebuild.

The fingerprint is the safety mechanism: an index is only valid for the exact
(corpus, embedding model, chunking) that produced it. Query vectors from a
different embedding model, or chunks cut differently, degrade retrieval
*silently* — answers get worse with no error anywhere. So we persist the
fingerprint next to the index and refuse to trust a stale one.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from rag.config import Settings
from rag.ingestion.chunking import chunk_documents
from rag.ingestion.loaders import load_documents

logger = logging.getLogger(__name__)

FINGERPRINT_FILE = "fingerprint.json"

# Bump whenever ingestion/preprocessing CODE changes behavior (loaders, normalization,
# splitter). Settings and corpus changes are hashed automatically; code changes are not
# — this constant closes that gap so stale indexes still rebuild.
PIPELINE_VERSION = 2


def _corpus_hash(docs_dir: Path) -> str:
    """Content hash of the corpus — deterministic across machines (unlike mtimes)."""
    hasher = hashlib.sha256()
    for path in sorted(p for p in docs_dir.rglob("*") if p.is_file()):
        hasher.update(path.name.encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def compute_fingerprint(settings: Settings) -> dict:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "corpus_hash": _corpus_hash(settings.docs_dir),
    }


def _embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )


def build_index(settings: Settings) -> FAISS:
    """Build the index from scratch and persist it with its fingerprint."""
    docs = load_documents(settings.docs_dir)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    logger.info("Embedding %d chunks from %d pages...", len(chunks), len(docs))

    index = FAISS.from_documents(chunks, _embeddings(settings))

    if settings.index_dir.exists():
        shutil.rmtree(settings.index_dir)
    index.save_local(str(settings.index_dir))
    fingerprint = compute_fingerprint(settings)
    (settings.index_dir / FINGERPRINT_FILE).write_text(json.dumps(fingerprint, indent=2))
    logger.info("Index built: %d chunks -> %s", len(chunks), settings.index_dir)
    return index


def load_or_build_index(settings: Settings, force_rebuild: bool = False) -> FAISS:
    """Load the persisted index, rebuilding automatically if it is missing or stale."""
    fingerprint_path = settings.index_dir / FINGERPRINT_FILE

    if not force_rebuild and fingerprint_path.exists():
        stored = json.loads(fingerprint_path.read_text())
        current = compute_fingerprint(settings)
        if stored == current:
            # Safe only because we exclusively load indexes this code built.
            return FAISS.load_local(
                str(settings.index_dir),
                _embeddings(settings),
                allow_dangerous_deserialization=True,
            )
        changed = [k for k in current if stored.get(k) != current[k]]
        logger.warning("Index stale (changed: %s) — rebuilding.", ", ".join(changed))

    return build_index(settings)
