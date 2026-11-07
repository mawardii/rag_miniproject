"""FAISS index lifecycle: build, fingerprint, load-or-rebuild.

The fingerprint is the safety mechanism: an index is only valid for the exact
corpus, embedding model, and chunking settings that produced it. Query vectors
from a different embedding model, or chunks cut differently, degrade retrieval
silently. We persist the fingerprint next to the index and refuse to trust a
stale one.
"""

import hashlib
import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from rag.config import Settings
from rag.ingestion.chunking import chunk_documents
from rag.ingestion.loaders import load_documents
from rag.ingestion.tables import tag_chunk_type

logger = logging.getLogger(__name__)

FINGERPRINT_FILE = "fingerprint.json"

# Bump whenever ingestion/preprocessing CODE changes behavior (loaders, normalization,
# splitter). Settings and corpus changes are hashed automatically; code changes are not.
PIPELINE_VERSION = 4


def _corpus_hash(docs_dir: Path) -> str:
    """Content hash of the corpus, deterministic across machines."""
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


def _new_temp_index_dir(index_dir: Path) -> Path:
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{index_dir.name}-", dir=index_dir.parent))


def _swap_index_dir(new_dir: Path, index_dir: Path) -> None:
    """Replace the live index with rollback if the final move fails."""
    backup_dir: Path | None = None
    if index_dir.exists():
        backup_dir = index_dir.with_name(f".{index_dir.name}.backup-{uuid.uuid4().hex}")
        shutil.move(str(index_dir), str(backup_dir))

    try:
        shutil.move(str(new_dir), str(index_dir))
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not index_dir.exists():
            shutil.move(str(backup_dir), str(index_dir))
        raise
    else:
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
    finally:
        if new_dir.exists():
            shutil.rmtree(new_dir, ignore_errors=True)


def build_index(settings: Settings) -> FAISS:
    """Build the index from scratch and persist it with its fingerprint."""
    docs = load_documents(settings.docs_dir)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    tag_chunk_type(chunks)
    logger.info("Embedding %d chunks from %d pages...", len(chunks), len(docs))

    index = FAISS.from_documents(chunks, _embeddings(settings))

    temp_index_dir = _new_temp_index_dir(settings.index_dir)
    index.save_local(str(temp_index_dir))
    fingerprint = compute_fingerprint(settings)
    (temp_index_dir / FINGERPRINT_FILE).write_text(json.dumps(fingerprint, indent=2))
    _swap_index_dir(temp_index_dir, settings.index_dir)
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
        logger.warning("Index stale (changed: %s) - rebuilding.", ", ".join(changed))

    return build_index(settings)
