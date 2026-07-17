# Road-Damage Research RAG

A Retrieval-Augmented Generation (RAG) QA system that answers questions grounded in a
corpus of road-damage-detection research papers (YOLO-based detection, pavement crack
classification, Malaysian road datasets — see `docs/`).

> **Status:** under active refactor from prototype to production service.
> Roadmap and progress live in [CLAUDE.md](CLAUDE.md).

## How it works

```
                 ┌─────────────── ingestion (offline) ───────────────┐
 PDFs in docs/ ──► load ──► chunk ──► embed (OpenAI text-embedding-3-small) ──► FAISS index
                                                                                    │
 user question ──► dense search (FAISS) ──┐
               └─► keyword search (BM25) ──┤ RRF fusion ──► LLM rerank ──► top-k chunks
                                                                              │
                                                                    chunks + question
                                                                              ▼
                                                DeepSeek chat LLM ──► grounded answer + sources
```

Retrieval is **hybrid** (dense + BM25, Reciprocal Rank Fusion) with **listwise LLM
reranking** — each step config-switchable and chosen by measurement on the eval set:
dense-only 0.92 hit@4 / 0.83 MRR → hybrid 1.00 / 0.83 → hybrid+rerank 1.00 / 0.96.

Two API providers are involved on purpose: **OpenAI** serves embeddings (DeepSeek does
not offer an embedding API), **DeepSeek** generates answers (cheaper for chat).

## Quickstart

```bash
# 1. Create and activate a virtualenv (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install (include dev tools: lint + tests)
pip install -e ".[dev]"

# 3. Configure secrets
copy example.env .env          # then fill in your real keys
# Required: DEEPSEEK_API_KEY (chat), OPENAI_API_KEY (embeddings)

# 4. Sanity-check connectivity (optional but recommended)
python scripts/smoke_test.py

# 5. Build the index (~1 min, ~$0.02 in embedding calls), then run the app
rag-ingest
streamlit run ui/app.py
```

Ask something grounded in the corpus, e.g. *"Which YOLO versions are used for road
damage detection and how do they compare?"* — answers cite their source PDFs.

## Project layout

```
docs/                     source corpus (PDF research papers)
src/rag/
├── config.py             all settings (pydantic-settings; env/.env-driven)
├── ingestion/            loaders (PDF/txt + NFKC normalization), chunking
├── retrieval/            FAISS index lifecycle + staleness fingerprint
├── generation/           prompts (explicit, versioned)
├── pipeline.py           RAGPipeline facade — the only entry point consumers use
└── cli.py                `rag-ingest` command
ui/app.py                 Streamlit chat UI (thin client over RAGPipeline)
scripts/smoke_test.py     API connectivity check
faiss_index/              generated vector index (git-ignored build artifact)
.claude/                  AI-collaborator skills and agents
```

A FastAPI backend arrives in Phase 3 of the roadmap (see [CLAUDE.md](CLAUDE.md)).

## Development

```bash
ruff check .        # lint
ruff format .       # format
rag-eval            # retrieval metrics against evals/golden.jsonl
rag-eval --answers  # + LLM-judged answer faithfulness/relevance
pytest              # tests (arrive in Phase 6)
```

Retrieval/generation changes must ship with a `rag-eval` before/after delta.

The index carries a fingerprint (embedding model, chunk params, corpus hash, pipeline
version); `rag-ingest` and the app rebuild automatically when it goes stale. After
changing ingestion **code**, bump `PIPELINE_VERSION` in `src/rag/retrieval/vectorstore.py`.
