# RAG Mini-Project → Production RAG Service

A RAG (Retrieval-Augmented Generation) QA system over a corpus of road-damage-detection
research papers (PDFs in `docs/`). Currently mid-refactor from a Streamlit prototype to a
production-grade service. **Check the Roadmap section below for the current phase before
making changes.**

## Stack

- **LLM:** DeepSeek (`deepseek-chat`) via OpenAI-compatible API (`base_url=https://api.deepseek.com`)
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector store:** FAISS (local, persisted to `faiss_index/`)
- **Orchestration:** LangChain
- **UI:** Streamlit (`main.py`)
- **Python:** 3.10

## Commands

```bash
# Install (deps + dev tools; pyproject.toml is the single source of truth)
pip install -e ".[dev]"

# Lint / format
ruff check .
ruff format .

# Run the app
streamlit run ui/app.py

# Build / refresh the vector index (auto-detects staleness via fingerprint)
rag-ingest
rag-ingest --force                      # rebuild unconditionally
rag-ingest --query "some question"      # retrieval spot-check (no LLM call)

# Smoke-test API keys and connectivity
python scripts/smoke_test.py

# Evaluate the pipeline (retrieval metrics; --answers adds LLM-judged quality)
rag-eval
rag-eval --answers
RETRIEVAL_MODE=dense RERANK_ENABLED=false rag-eval   # compare configs via env overrides
```

Secrets live in `.env` (never committed — `example.env` is the template).
Required keys: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`.

## Repo rules (non-negotiable)

- **Never commit:** `.env`, `rag-cloud/` (a virtualenv — being purged from git),
  `faiss_index/` (build artifact), `__pycache__/`.
- **Never log or print API keys**, even partially.
- The FAISS index is derived data. Settings/corpus changes are caught automatically by
  the index fingerprint; **code** changes to ingestion behavior (loaders, normalization,
  splitter) must bump `PIPELINE_VERSION` in `src/rag/retrieval/vectorstore.py`.
- `FAISS.load_local(..., allow_dangerous_deserialization=True)` is acceptable only because
  we exclusively load indexes we built ourselves. Never load an index from an untrusted source.
- Pin dependencies in `pyproject.toml` (there is no requirements.txt); don't add unpinned packages.

## Conventions we are adopting during the refactor

- Package code lives in `src/rag/`; scripts in `scripts/`; tests in `tests/`; eval sets in `evals/`.
- Configuration via `pydantic-settings` (single `Settings` class in `src/rag/config.py`) —
  no scattered `os.getenv()` calls.
- Ingestion, retrieval, and generation are separate modules with typed interfaces —
  the UI and API are thin layers on top.
- Every retrieval/generation change must be run through `rag-eval` and the metric
  delta reported in the commit message. No upgrade lands on vibes.
- Type hints everywhere; `ruff` for lint/format; `pytest` for tests.

## Roadmap (production refactor)

- [x] **Phase 1 — Repo hygiene:** purge `rag-cloud/` venv + binaries from git, proper
  `.gitignore`, `pyproject.toml`, pinned deps, README.
- [x] **Phase 2 — Architecture:** `src/rag/` package: `config.py`, `ingestion/`,
  `retrieval/`, `generation/`; `rag-ingest` CLI; Streamlit is a thin client (`ui/app.py`).
- [x] **Phase 3 — Eval harness:** golden QA dataset in `evals/`, retrieval metrics
  (hit@k, MRR), LLM-judged answer faithfulness, `rag-eval` CLI. Lands BEFORE retrieval
  upgrades so every upgrade ships with a measured delta.
- [x] **Phase 4 — Industry-standard retrieval:** hybrid search (BM25 + dense, RRF fusion),
  LLM listwise reranking, config-switchable modes. Measured: dense 0.92 hit@4 / 0.83 MRR
  → hybrid 1.00 / 0.83 → hybrid+rerank 1.00 / 0.96.
- [ ] **Phase 5 — API layer:** FastAPI service (`/query`, `/healthz`), request/response
  models, error handling, timeouts, streaming.
- [ ] **Phase 6 — Observability + tests:** structured logging, request IDs, token/cost
  tracking, pytest unit tests, CI via GitHub Actions.
- [ ] **Phase 7 — Deployment:** Dockerfile (multi-stage), docker-compose (API + UI),
  health checks, deploy docs.

Update the checkboxes as phases complete.

## Gotchas

- All pipeline consumers (UI, CLI, future API/eval) must go through `rag.pipeline.RAGPipeline`
  and `rag.config.Settings` — never instantiate LLM/retriever clients ad hoc.
- Windows consoles are cp1252: PDF-derived text can contain characters they can't print.
  CLI entry points call `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`.
- PDF text is NFKC-normalized at load time (ligatures like ﬁ → fi); skipping this makes
  embeddings mismatch plain-ASCII queries.
- Sanity bound: the current 13-PDF corpus yields ~800 chunks at 800/120. An order-of-
  magnitude deviation after an ingestion change means loaders or chunking broke.
- Git history was rewritten on 2026-07-17 (git filter-repo) to purge the 190 MB venv;
  any pre-rewrite clone must be re-cloned, not pulled.
