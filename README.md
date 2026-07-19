# Financial Filings RAG

A Retrieval-Augmented Generation (RAG) QA system that answers analyst-style questions
grounded in corporate SEC filings — 10-Ks, 10-Qs, 8-Ks, and earnings releases from 14
public companies (3M, AMD, Boeing, Pfizer, PepsiCo, ...), curated from the
[FinanceBench](https://github.com/patronus-ai/financebench) benchmark. Document QA over
filings is the flagship enterprise RAG use case (research copilots at investment banks,
diligence tooling), which makes this corpus an honest, business-shaped testbed.

> **Status:** under active refactor from prototype to production service.
> Roadmap and progress live in [CLAUDE.md](CLAUDE.md).

## How it works

```
                 ┌─────────────── ingestion (offline) ───────────────┐
 PDFs in docs/ ──► load ──► chunk ──► embed (OpenAI text-embedding-3-small) ──► FAISS index
                                                                                    │
 user question ──► dense search (FAISS) ──┬──► top-k chunks ──► chunks + question
               └─► keyword search (BM25) ─┘ [+ RRF fusion,               │
                     (hybrid mode only)       optional LLM rerank]       ▼
                                                DeepSeek chat LLM ──► grounded answer + sources
```

Retrieval strategy is **config-switchable** — dense-only, hybrid (dense + BM25 with
Reciprocal Rank Fusion), and optional listwise LLM reranking — and the default is
chosen by measurement on the eval set, per corpus. On the current SEC-filings corpus
(69 FinanceBench questions) dense-only wins (0.93 hit@4 / 0.81 MRR); on a
research-paper corpus with distinctive vocabulary, hybrid+rerank won. Corpus
properties decide; `rag-eval` is the referee.

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

# 4. Fetch the corpus (public SEC filings; PDFs are git-ignored)
python scripts/download_corpus.py

# 5. Sanity-check connectivity (optional but recommended)
python scripts/smoke_test.py

# 6. Build the index (~14k chunks, ~$0.10 in embedding calls), then run the app
rag-ingest
streamlit run ui/app.py
```

Ask something an analyst would, e.g. *"What drove AMD's revenue change in FY2022?"* —
answers cite the source filing and page.

## Project layout

```
docs/                     corpus: SEC filings (git-ignored; scripts/download_corpus.py)
evals/                    golden QA set generated from FinanceBench (scripts/build_golden.py)
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
