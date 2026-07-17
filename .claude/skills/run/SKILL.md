---
name: run
description: Launch the RAG app locally and verify it actually answers a question end-to-end
---

# Running the RAG app

## Preconditions
1. `.env` exists at repo root with `DEEPSEEK_API_KEY` and `OPENAI_API_KEY` set
   (copy `example.env` if missing — but never invent keys).
2. Dependencies installed: `pip install -e ".[dev]"`.
3. If connectivity is in doubt, smoke-test first: `python query_test.py`
   (must print both "Success" lines).

## Launch
```bash
streamlit run main.py --server.headless true --server.port 8501
```
Run in the background; the app is up when `http://localhost:8501` returns 200.
First run builds the FAISS index from `docs/` — this calls the OpenAI embeddings API
and can take a minute for the full PDF corpus.

> Phase 3 of the roadmap adds a FastAPI backend; once it lands, start it with
> `uvicorn rag.api.app:app` and update this skill.

## Verify (don't skip)
A 200 from Streamlit only proves the server started. To verify the RAG path works,
ask a corpus-grounded question and check the answer cites sources, e.g.:
"What datasets are used for road damage detection?" — the answer should mention
RDD/RCD-IIUM-style datasets and list PDF sources in the Sources expander.

## Common failures
- `AuthenticationError` on startup → wrong/missing `OPENAI_API_KEY` (embeddings, not the LLM).
- Empty/garbage answers with tiny chunks → index was built with bad chunk settings; rebuild via the `ingest` skill.
- Port already in use → another Streamlit instance is running; kill it rather than switching ports.
