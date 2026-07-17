"""Load raw documents from the corpus directory."""

import logging
import unicodedata
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def load_documents(docs_dir: Path) -> list[Document]:
    """Load all supported files in docs_dir; one Document per PDF page / text file.

    Unsupported files are skipped loudly (logged), never silently — a corpus that
    half-loads is worse than one that fails, because nobody notices.
    """
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {docs_dir}")

    docs: list[Document] = []
    for path in sorted(docs_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        elif suffix == ".txt":
            loaded = TextLoader(str(path), autodetect_encoding=True).load()
        elif path.is_file():
            logger.warning("Skipping unsupported file: %s", path.name)
            continue
        else:
            continue
        for doc in loaded:
            # Normalize source to the bare filename so citations are stable across machines.
            doc.metadata["source"] = path.name
            # NFKC folds PDF ligatures (ﬁ -> fi) and odd Unicode forms; without this,
            # "classiﬁcation" and "classification" embed differently and retrieval misses.
            doc.page_content = unicodedata.normalize("NFKC", doc.page_content)
        docs.extend(loaded)
        logger.info("Loaded %s (%d pages/sections)", path.name, len(loaded))

    if not docs:
        raise RuntimeError(f"No loadable documents in {docs_dir} (expected .pdf/.txt)")
    return docs
