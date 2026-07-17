"""CLI for index management: `rag-ingest [--force] [--query "..."]`.

Ingestion is an offline batch job, so it gets a CLI — in production it runs
from a terminal, a cron job, or a CI step, never from a button in the UI.
"""

import argparse
import logging
import sys

from rag.config import get_settings
from rag.retrieval.vectorstore import build_index, load_or_build_index


def main() -> None:
    # Windows consoles default to cp1252, which can't print PDF-derived Unicode;
    # never let a display issue crash a data job.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build/refresh the FAISS index.")
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if the fingerprint matches."
    )
    parser.add_argument(
        "--query", help="After loading, run a retrieval spot-check with this query."
    )
    args = parser.parse_args()

    settings = get_settings()
    index = build_index(settings) if args.force else load_or_build_index(settings)
    print(f"Index ready at {settings.index_dir} ({index.index.ntotal} vectors)")

    if args.query:
        print(f"\nTop-{settings.top_k} retrieval for: {args.query!r}\n")
        for i, doc in enumerate(index.similarity_search(args.query, k=settings.top_k), 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            location = f"{source}, p.{page + 1}" if page is not None else source
            preview = " ".join(doc.page_content.split())[:150]
            print(f"[{i}] {location}\n    {preview}\n")


if __name__ == "__main__":
    main()
