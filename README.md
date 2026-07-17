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
 user question ──► embed query ──► top-k similarity search ────────────────────────┘
                                        │
                              retrieved chunks + question
                                        │
                                        ▼
                        DeepSeek chat LLM ──► grounded answer + sources
```

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
python query_test.py

# 5. Run — first launch builds the FAISS index (~1 min, ~$0.02 in embedding calls)
streamlit run main.py
```

Ask something grounded in the corpus, e.g. *"Which YOLO versions are used for road
damage detection and how do they compare?"* — answers cite their source PDFs.

## Project layout

```
docs/           source corpus (PDF research papers)
main.py         Streamlit chat UI (prototype entry point)
embeddings.py   index build/load logic
prototype.py    minimal scripted RAG example
query_test.py   API connectivity smoke test
faiss_index/    generated vector index (git-ignored build artifact)
.claude/        AI-collaborator skills and agents
```

This flat layout is temporary; Phase 2 of the roadmap moves logic into a `src/rag/`
package with separated ingestion / retrieval / generation modules and a FastAPI backend.

## Development

```bash
ruff check .        # lint
ruff format .       # format
pytest              # tests (arrive in Phase 4)
```

Changing chunking, loaders, or the embedding model **requires rebuilding the index**:
delete `faiss_index/` and run `python embeddings.py`.
