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
questions), dense-only is the default measured winner: 0.93 hit@4 / 0.81 MRR.

Two API providers are involved on purpose: OpenAI serves embeddings because DeepSeek
does not provide embeddings, and DeepSeek generates answers.

## Quickstart

```bash
# 1. Create and activate a virtualenv (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies and dev tools
pip install -e ".[dev]"

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
```

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
pytest              # once Phase 6 tests are added
```

Retrieval and generation changes should ship with a `rag-eval` before/after delta.

The index carries a fingerprint: embedding model, chunk params, corpus hash, and
pipeline version. `rag-ingest` and the app rebuild automatically when it goes stale.
After changing ingestion code behavior, bump `PIPELINE_VERSION` in
`src/rag/retrieval/vectorstore.py`.
