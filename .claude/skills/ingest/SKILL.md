---
name: ingest
description: Build or rebuild the FAISS vector index from the PDF corpus in docs/, and when a rebuild is required
---

# Ingesting documents / rebuilding the index

## When a rebuild is REQUIRED (not optional)
- Files added to / removed from / edited in `docs/`
- Chunk size, chunk overlap, splitter, or loader code changed
- Embedding model changed (this silently breaks retrieval if skipped — old vectors
  and new query vectors live in different spaces)

## How
```bash
# Full rebuild: delete the old index first so a stale one can't be loaded
rm -rf faiss_index
python embeddings.py
```
Requires `OPENAI_API_KEY` in `.env`. Cost: ~$0.02 per full rebuild of the current
corpus (text-embedding-3-small) — cheap, so when in doubt, rebuild.

## Verify
The script prints `Indexed N chunks`. Sanity bounds for the current 13-PDF corpus:
N should be in the hundreds-to-low-thousands. N < 50 means loaders silently failed
on the PDFs; N > 10,000 means chunking is misconfigured.

Then run a retrieval spot-check (no LLM needed):
```python
from embeddings import load_or_build_vectorstore
db = load_or_build_vectorstore()
for d in db.similarity_search("YOLOv8 road damage detection", k=3):
    print(d.metadata.get("source"), "|", d.page_content[:80])
```
All 3 hits should come from relevant papers and contain readable text (not PDF junk).

## Rules
- `faiss_index/` is a build artifact — never commit it.
- Never ship an index built with different settings than the code defaults; the
  "Rebuild index" button in the UI exists for users, not as a substitute for this process.
