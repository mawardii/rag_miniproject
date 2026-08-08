from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from rag.api.app import app, get_app_settings, get_pipeline
from rag.pipeline import RAGResult, SourceRef


class FakePipeline:
    def answer(self, question: str) -> RAGResult:
        return RAGResult(
            answer=f"answer to {question}",
            sources=[
                SourceRef(
                    source="AMD_2022_10K.pdf",
                    page=0,
                    snippet="Revenue increased.",
                )
            ],
            latency_s=0.12,
            input_tokens=10,
            output_tokens=20,
        )


def test_healthz_reports_config_and_index_state(tmp_path: Path) -> None:
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "fingerprint.json").write_text("{}", encoding="utf-8")
    settings = SimpleNamespace(
        index_dir=index_dir,
        retrieval_mode="dense",
        rerank_enabled=False,
        top_k=4,
    )
    app.dependency_overrides[get_app_settings] = lambda: settings

    try:
        response = TestClient(app).get("/healthz")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "index_ready": True,
        "retrieval_mode": "dense",
        "rerank_enabled": False,
        "top_k": 4,
    }


def test_query_returns_answer_and_one_indexed_sources() -> None:
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()

    try:
        response = TestClient(app).post("/query", json={"question": "What changed?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "answer": "answer to What changed?",
        "sources": [
            {
                "source": "AMD_2022_10K.pdf",
                "page": 1,
                "snippet": "Revenue increased.",
            }
        ],
        "latency_s": 0.12,
        "input_tokens": 10,
        "output_tokens": 20,
    }
