---
name: ingest
description: Build or refresh the FAISS vector index via rag-ingest, and when a rebuild is required
---

# Ingesting documents / rebuilding the index

## How
```bash
rag-ingest                      # loads if fingerprint matches, rebuilds if stale
rag-ingest --force              # rebuild unconditionally
rag-ingest --query "question"   # retrieval spot-check (embeds query only; no LLM)
```
Requires `OPENAI_API_KEY` in `.env`. Full rebuild costs ~$0.02 — when in doubt, rebuild.

## Staleness is mostly automatic
The index stores a fingerprint (embedding model, chunk size/overlap, corpus content
hash, `PIPELINE_VERSION`). Settings or corpus changes trigger auto-rebuild on next load.

**The one manual step:** if you change ingestion *code* behavior (loaders, text
normalization, splitter), bump `PIPELINE_VERSION` in `src/rag/retrieval/vectorstore.py`
— content hashing cannot see code changes.

## Verify
Build logs `Index built: N chunks`. Sanity bounds for the current 13-PDF corpus:
~800 chunks at the default 800/120 settings. N < 100 means loaders silently failed;
N > 5,000 means chunking is misconfigured.

Then spot-check retrieval:
```bash
rag-ingest --query "YOLOv8 road damage detection"
```
All hits should come from relevant papers and read as clean text (no ligature junk
like "classiﬁcation" — NFKC normalization handles this at load time).

## Rules
- `faiss_index/` is a build artifact — never commit it.
- Never hand-edit files inside `faiss_index/`; the fingerprint is written only by
  `build_index()`.
