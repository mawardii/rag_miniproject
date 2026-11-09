# RAG Mini-Project -> Production RAG Service

A RAG QA system over corporate SEC filings (10-K, 10-Q, 8-K, and earnings
materials), curated from the FinanceBench benchmark and fetched into `docs/` by
`scripts/download_corpus.py`. The project is mid-refactor from a Streamlit prototype
to a production-grade service.

## Stack

- **LLM:** DeepSeek (`deepseek-v4-flash`) via OpenAI-compatible API
  (`base_url=https://api.deepseek.com`)
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector store:** FAISS, persisted to `faiss_index/`
- **Orchestration:** LangChain
- **UI:** Streamlit (`ui/app.py`)
- **Python:** 3.10+

## Commands

```bash
# Install dependencies and dev tools
pip install -e ".[dev]"

# Fetch the corpus
python scripts/download_corpus.py

# Lint and format
ruff check .
ruff format .

# Run the app
streamlit run ui/app.py

# Run the API
uvicorn rag.api.app:app --host 127.0.0.1 --port 8000

# Build or refresh the vector index
rag-ingest
rag-ingest --force
rag-ingest --query "some question"

# Smoke-test API keys and connectivity
python scripts/smoke_test.py

# Evaluate the pipeline
rag-eval
rag-eval --answers
RETRIEVAL_MODE=dense RERANK_ENABLED=false rag-eval
```

Secrets live in `.env`, never in git. Required keys: `DEEPSEEK_API_KEY`,
`OPENAI_API_KEY`.

## Local Environment

The developer may use the existing ignored `rag-cloud/` virtualenv:

```powershell
.\rag-cloud\Scripts\Activate.ps1
```

Do not commit files from `rag-cloud/`. Windows virtualenv launchers contain absolute
paths to the base interpreter, so a venv can work on one machine and fail in another
shell with `No Python at ...`. If that happens, recreate the virtualenv from a working
Python 3.10+ install and reinstall with `pip install -e ".[dev]"`.

Useful checks after activation:

```powershell
Get-Command python,pip,ruff,pytest
python --version
python -m pip show rag-miniproject
```

## Repo Rules

- Never commit `.env`, `rag-cloud/`, `faiss_index/`, or `__pycache__/`.
- Never log or print API keys, even partially.
- The FAISS index is derived data. Settings and corpus changes are caught by the
  index fingerprint; code changes to ingestion behavior must bump
  `PIPELINE_VERSION` in `src/rag/retrieval/vectorstore.py`.
- Index rebuilds save to a temporary sibling directory before replacing the live
  `faiss_index/`, with rollback if the final move fails.
- `FAISS.load_local(..., allow_dangerous_deserialization=True)` is acceptable only
  for indexes this code built locally. Never load an index from an untrusted source.
- Pin dependencies in `pyproject.toml`; there is no `requirements.txt`.

## Conventions

- Package code lives in `src/rag/`; scripts in `scripts/`; tests in `tests/`; eval
  sets in `evals/`.
- Configuration lives in `src/rag/config.py`; avoid scattered `os.getenv()` calls.
- Ingestion, retrieval, generation, and serving layers stay separate.
- UI, eval, CLI, and future API consumers go through `rag.pipeline.RAGPipeline`.
- Retrieval and generation changes should be measured with `rag-eval`.
- Commit messages follow Conventional Commits.

## Roadmap

- [x] **Phase 1 - Repo hygiene:** `.gitignore`, `pyproject.toml`, pinned deps, README,
  and ignored runtime artifacts.
- [x] **Phase 2 - Architecture:** `src/rag/` package, `Settings`, ingestion/retrieval/
  generation modules, `rag-ingest`, and thin Streamlit UI.
- [x] **Phase 3 - Eval harness:** golden QA dataset, hit@k, MRR, LLM-judged answer
  quality, and `rag-eval`.
- [x] **Phase 4 - Retrieval options:** dense search, BM25+dense hybrid retrieval with
  RRF, optional LLM listwise reranking, and env-switchable config.
- [x] **Phase 5 - API layer:** FastAPI service with `/query`, `/healthz`, request/
  response models, error handling, and model/client timeouts. Streaming is deferred.
- [ ] **Phase 6 - Observability + tests:** structured logging, request IDs, token/cost
  tracking, pytest unit tests, and CI.
- [ ] **Phase 7 - Deployment:** Dockerfile, docker-compose, health checks, and deploy
  docs.

## Current Retrieval Default

The current default is dense + table boost, fetch 20, DeepSeek rerank with source/page
metadata, return 8 chunks. On the 69-question development set it achieves 98.6%
filing hit@8, 60.9% evidence-page hit@8, 58.7% page recall@8, and 0.450 page MRR.
Mean retrieval time was 1.48 seconds in one run. Before this experiment the top-4,
no-rerank configuration achieved 89.9% filing hit, 33.3% page hit, and 29.7% page
recall. See README's experiment table for controls and known regressions.
These are retrieval metrics; answer correctness has not been remeasured.

## Historical Retrieval Baseline

Reference numbers for the SEC-filings corpus, 69 FinanceBench questions,
`deepseek-v4-flash`, 2026-07-18:

| Config | hit@4 | MRR |
|---|---:|---:|
| dense only (former default) | 0.93 | 0.81 |
| hybrid | 0.86 | 0.77 |
| hybrid + rerank | 0.90 | 0.77 |
| dense + rerank | 0.91 | 0.83 |

Historical answer quality at the former dense-only config: faithfulness 4.96/5, relevance
4.97/5, correctness 3.75/5. The correctness gap appears concentrated in numeric
and table-heavy questions.

## Test Coverage

The initial pytest suite covers pure logic only: prompt/context formatting, rerank
response parsing, golden-file loading, retrieval metric calculation, and index
fingerprint hashing. These tests should not require API keys, PDFs, or a FAISS index.

## Gotchas

- Windows consoles often default to cp1252; CLI entry points call
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`.
- PDF text is NFKC-normalized at load time; skipping this can hurt retrieval for
  plain-ASCII queries.
- The current 18-filing corpus yields roughly 13.8k chunks at 800/120. Large
  deviations after ingestion changes usually mean loader or chunking behavior changed.
- `evals/golden.jsonl` is generated by `scripts/build_golden.py` from FinanceBench's
  open questions. Regenerate it after changing the corpus.
