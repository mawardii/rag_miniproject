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
# Run the app (prototype entry point — will move to ui/ + FastAPI later)
streamlit run main.py

# Build / rebuild the vector index
python embeddings.py

# Smoke-test API keys and connectivity
python query_test.py
```

Secrets live in `.env` (never committed — `example.env` is the template).
Required keys: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`.

## Repo rules (non-negotiable)

- **Never commit:** `.env`, `rag-cloud/` (a virtualenv — being purged from git),
  `faiss_index/` (build artifact), `__pycache__/`.
- **Never log or print API keys**, even partially.
- The FAISS index is derived data: any change to chunking, embedding model, or loaders
  requires a rebuild — say so in the PR/commit message.
- `FAISS.load_local(..., allow_dangerous_deserialization=True)` is acceptable only because
  we exclusively load indexes we built ourselves. Never load an index from an untrusted source.
- Pin dependencies in `requirements.txt` / `pyproject.toml`; don't add unpinned packages.

## Conventions we are adopting during the refactor

- Package code lives in `src/rag/`; scripts in `scripts/`; tests in `tests/`; eval sets in `evals/`.
- Configuration via `pydantic-settings` (single `Settings` class in `src/rag/config.py`) —
  no scattered `os.getenv()` calls.
- Ingestion, retrieval, and generation are separate modules with typed interfaces —
  the UI and API are thin layers on top.
- Every retrieval/generation change must keep the eval harness passing (once Phase 4 lands).
- Type hints everywhere; `ruff` for lint/format; `pytest` for tests.

## Roadmap (production refactor)

- [ ] **Phase 1 — Repo hygiene:** purge `rag-cloud/` venv + binaries from git, proper
  `.gitignore`, `pyproject.toml`, pinned deps, README.
- [ ] **Phase 2 — Architecture:** `src/rag/` package: `config.py`, `ingestion/`,
  `retrieval/`, `generation/`; CLI script for indexing; Streamlit becomes a thin client.
- [ ] **Phase 3 — API layer:** FastAPI service (`/query`, `/healthz`), request/response
  models, error handling, timeouts, streaming.
- [ ] **Phase 4 — Quality:** golden QA dataset in `evals/`, retrieval + answer metrics
  (hit-rate, faithfulness), pytest unit tests, CI via GitHub Actions.
- [ ] **Phase 5 — Observability:** structured logging, request IDs, token/cost tracking,
  latency metrics.
- [ ] **Phase 6 — Deployment:** Dockerfile (multi-stage), docker-compose (API + UI),
  health checks, deploy docs.

Update the checkboxes as phases complete.

## Gotchas

- `main.py` currently rebuilds the QA chain on every Streamlit rerun — known perf issue,
  fixed in Phase 2 with caching.
- `CharacterTextSplitter(separator="\n")` frequently produces oversized chunks on PDFs
  (it only splits on newlines); Phase 2 replaces it with `RecursiveCharacterTextSplitter`.
- The docs-hash function `compute_docs_hash()` in `embeddings.py` exists but is never
  called — index staleness is currently manual ("Rebuild index" button).
- Git history still contains the 190 MB venv until the history purge in Phase 1 is done.
