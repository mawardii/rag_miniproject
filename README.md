# Financial Filings RAG

A Retrieval-Augmented Generation (RAG) QA system that answers analyst-style questions
grounded in corporate SEC filings: 10-Ks, 10-Qs, 8-Ks, and earnings releases from
18 public-company documents curated from the
[FinanceBench](https://github.com/patronus-ai/financebench) benchmark.

> **Status:** under active refactor from Streamlit prototype to production service.
> Roadmap and progress live in [CLAUDE.md](CLAUDE.md).

## How It Works

```text
Offline ingest:
  PDFs in docs/ -> load -> chunk -> embed with OpenAI text-embedding-3-small -> FAISS index

Question answering:
  user question -> dense FAISS search -> optional BM25/RRF hybrid -> optional LLM rerank
                -> top-k excerpts + question -> DeepSeek chat model -> grounded answer + sources
```

Retrieval strategy is config-switchable: dense-only, hybrid dense+BM25 with Reciprocal
Rank Fusion, and optional listwise LLM reranking. The default is chosen by measurement
on the eval set, per corpus. On the current SEC-filings corpus (69 FinanceBench
questions), the current default fetches 20 dense/table-boost candidates, reranks
with DeepSeek using filing/page metadata, and returns up to 8 chunks. It reaches
0.61 evidence-page hit@8 and 0.59 evidence-page recall@8. The earlier dense-only
0.93 filing hit@4 / 0.81 filing MRR baseline is historical, before the table boost.

Two API providers are involved on purpose: OpenAI serves embeddings because DeepSeek
does not provide embeddings, and DeepSeek generates answers.

## Quickstart

```bash
# 1. Create and activate a virtualenv (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies and dev tools
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# 3. Configure secrets
copy example.env .env          # then fill in real keys
# Required: DEEPSEEK_API_KEY (chat), OPENAI_API_KEY (embeddings)

# 4. Fetch the corpus
python scripts/download_corpus.py

# 5. Sanity-check connectivity
python scripts/smoke_test.py

# 6. Build the index, then run the app
rag-ingest
streamlit run ui/app.py

# Optional: run the API instead of the Streamlit UI
uvicorn rag.api.app:app --host 127.0.0.1 --port 8000
```

If you are using the existing local `rag-cloud` virtualenv instead of `.venv`, activate it
first:

```powershell
.\rag-cloud\Scripts\Activate.ps1
```

Then run the same `pip`, `rag-ingest`, `rag-eval`, and `streamlit` commands above.
`rag-cloud/` is intentionally ignored by git because virtualenv launchers contain
machine-specific absolute paths. If commands fail with a missing Python path, recreate
the venv rather than committing fixes inside `rag-cloud/`.

Ask something an analyst would, for example:

```text
What drove AMD's revenue change in FY2022?
```

Answers cite the source filing and page.

## Project Layout

```text
docs/                     corpus PDFs, ignored by git
evals/                    FinanceBench-derived golden QA set
src/rag/
  config.py               settings via pydantic-settings and .env
  ingestion/              loaders and chunking
  retrieval/              FAISS lifecycle, dense/hybrid retrieval, rerank
  generation/             prompts
  pipeline.py             RAGPipeline facade used by UI/eval/future API
  cli.py                  rag-ingest command
ui/app.py                 Streamlit chat UI
scripts/smoke_test.py     API connectivity check
faiss_index/              generated vector index, ignored by git
.claude/                  AI-collaborator skills and agents
```

A FastAPI backend is Phase 5 of the roadmap.

## Development

```bash
ruff check .        # lint
ruff format .       # format
rag-eval            # retrieval metrics against evals/golden.jsonl
rag-eval --answers  # plus LLM-judged answer quality
rag-eval --output evals/results/retrieval.json # save metrics and full retrieved chunks
python -m pytest -q # pure unit tests; no API keys required
uvicorn rag.api.app:app --reload
```

GitHub Actions runs dependency compatibility, lint, formatting, and unit-test checks
on Python 3.10 on Windows and Linux for every push and pull request. These checks
do not need API keys, corpus PDFs, or a generated index. To run the same checks locally:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip check
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Retrieval and generation changes should ship with a `rag-eval` before/after delta.

### Evidence-level evaluation

`python scripts/build_golden.py` preserves FinanceBench's evidence text and page
annotations for all 69 questions. Golden `evidence[].page` values are **one-based PDF
page numbers**, converted from FinanceBench's zero-based `evidence_page_num`; they
are not the page labels printed inside a filing. Chunk metadata remains zero-based
and is converted during evaluation. The 84 original evidence annotations were checked
against the corresponding local PDF pages using normalized extracted full-page text.

The evaluator reports distinct measures:

- **Filing hit@k:** any chunk from an expected filing appears in the top k.
- **Filing MRR (unique sources):** the historical metric, ranking deduplicated filenames.
- **Evidence-page hit@k:** any retrieved chunk matches an annotated source and page.
- **Evidence-page recall@k:** the fraction of distinct annotated source/page pairs
  retrieved, averaged per annotated question. Multiple chunks from one page count once.
- **Evidence-page MRR:** reciprocal rank of the first matching page, using actual
  chunk positions, including repeated filenames.

Older golden files without evidence still work; their questions are excluded from
evidence metrics, and annotation coverage is printed explicitly. Page matches are
an intermediate check: a chunk can come from the correct page and still omit the
required row, year header, or units. Use the JSON report's full `retrieved_chunks`
and the golden set's evidence text to audit whether the answer inputs are present.
In particular, inspect both a narrative question and a calculation requiring several
pages before interpreting a page hit as sufficient evidence.

Reports under `evals/results/` are generated local artifacts and ignored by git.
Evaluation makes query-embedding API calls; `--answers` additionally generates and
judges answers. The historical 0.93 hit@4 / 0.81 MRR figures above are filing-level
results and predate the table boost, which is enabled in the current pipeline.

### Wider retrieval experiment

Measured on the same 69 questions, with unchanged embeddings, corpus, chunking,
and table boost. All reranked runs fetch 20 candidates and use DeepSeek.

| Configuration | Filing hit | Evidence-page hit | Page recall | Page MRR | Mean retrieval time |
|---|---:|---:|---:|---:|---:|
| Previous default, 4 chunks | 89.9% | 33.3% | 29.7% | 0.204 | not recorded |
| 8 chunks, no reranking | 94.2% | 40.6% | 37.7% | 0.205 | 0.29 s |
| 8 chunks, reranking without source metadata | 97.1% | 60.9% | 58.7% | 0.440 | 1.57 s |
| **Current: 8 chunks, reranking with source/page** | **98.6%** | **60.9%** | **58.7%** | **0.450** | **1.48 s** |

The first four chunks of the current reranked result already hit an annotated
page for 39/69 questions (56.5%), versus 23/69 (33.3%) in the previous default.
At eight chunks it gains 20 page hits and loses one versus that baseline. The
remaining regression is Best Buy FY2023 cash-flow comparison: the retrieved
context includes discussion but omits the needed cash-flow totals. Best Buy's
acquisition question now retrieves both acquired company names. AES inventory
turnover still lacks cost of sales, and PepsiCo EBITDA less capex still lacks
capital spending despite retrieving its income and cash-flow pages.

This is retrieval evaluation on the development set, not held-out validation or
an answer-correctness measurement. Context text grows from about 2.8k to 5.8k
characters per question, plus one reranking call, increasing token usage. Timing
is a single-run mean, not a latency guarantee. Rerank errors fall back to the
candidate order. To reproduce the old configuration in PowerShell:

```powershell
$env:RERANK_ENABLED = 'false'
$env:TOP_K = '4'
rag-eval --output evals/results/previous-default.json
```

For the current configuration, set `RERANK_ENABLED=true`, `TOP_K=8`, and
`FETCH_K=20` before running `rag-eval`. These options do not rebuild the index.

The index carries a fingerprint: embedding model, chunk params, corpus hash, and
pipeline version. `rag-ingest` and the app rebuild automatically when it goes stale.
Rebuilds are written to a temporary sibling directory before replacing the live index.
After changing ingestion code behavior, bump `PIPELINE_VERSION` in
`src/rag/retrieval/vectorstore.py`.

## Environment Checks

After activating your virtualenv, these commands should resolve inside that environment:

```powershell
Get-Command python,pip,ruff,pytest
python --version
python -m pip show rag-miniproject
```

If `pytest` or `python` reports a missing base interpreter, verify that the installed
Python is accessible from the current shell (including its sandbox permissions).
If the interpreter was deleted, recreate the virtualenv from a working Python
3.10+ install and rerun `pip install -e ".[dev]"`. Prefer a fresh `.venv` over editing
virtualenv launchers or `pyvenv.cfg`. You can bypass activation by using
`.\.venv\Scripts\python.exe -m pytest -q` (and the same `python.exe -m` prefix for
`pip` and `ruff`).
