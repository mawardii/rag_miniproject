import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from rag import evaluation
from rag.evaluation import evaluate_retrieval, load_golden, summarize_retrieval


class FakePipeline:
    def __init__(self, docs: list[Document]):
        self.docs = docs

    def retrieve(self, question: str) -> list[Document]:
        return self.docs


def test_load_golden_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Golden set is empty"):
        load_golden(path)


def test_load_golden_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    row = {"question": "Q?", "expected_sources": ["source.pdf"]}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert load_golden(path) == [row]


def test_evaluate_retrieval_scores_unique_sources_in_order() -> None:
    pipeline = FakePipeline(
        [
            Document(page_content="chunk 1", metadata={"source": "wrong.pdf"}),
            Document(page_content="chunk 2", metadata={"source": "wrong.pdf"}),
            Document(page_content="chunk 3", metadata={"source": "right.pdf"}),
        ]
    )
    golden = [{"question": "Q?", "expected_sources": ["right.pdf"]}]

    rows = evaluate_retrieval(pipeline, golden)

    assert len(rows) == 1
    assert rows[0].hit is True
    assert rows[0].reciprocal_rank == 0.5
    assert rows[0].retrieved == ["wrong.pdf", "right.pdf"]


def test_correct_filing_wrong_page_fails_evidence_check() -> None:
    pipeline = FakePipeline(
        [Document(page_content="Unrelated paragraph", metadata={"source": "a.pdf", "page": 2})]
    )
    golden = [
        {
            "question": "Revenue?",
            "expected_sources": ["a.pdf"],
            "evidence": [{"source": "a.pdf", "page": 10, "text": "Revenue was 42"}],
        }
    ]

    row = evaluate_retrieval(pipeline, golden)[0]

    assert row.hit is True
    assert row.evidence_page_hit is False
    assert row.evidence_page_recall == 0
    assert row.evidence_page_reciprocal_rank == 0


def test_page_recall_deduplicates_pages_and_mrr_uses_actual_chunk_rank() -> None:
    pipeline = FakePipeline(
        [
            Document(page_content="Wrong page", metadata={"source": "a.pdf", "page": 0}),
            Document(page_content="Same filename", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="Revenue was 42", metadata={"source": "a.pdf", "page": 9}),
            Document(page_content="Another chunk", metadata={"source": "a.pdf", "page": 9}),
        ]
    )
    golden = [
        {
            "question": "Revenue growth?",
            "expected_sources": ["a.pdf"],
            "evidence": [
                {"source": "a.pdf", "page": 10, "text": "Revenue was 42"},
                {"source": "a.pdf", "page": 10, "text": "Units are millions"},
                {"source": "a.pdf", "page": 11, "text": "Prior year revenue was 40"},
            ],
        }
    ]

    row = evaluate_retrieval(pipeline, golden)[0]

    assert row.evidence_page_hit is True
    assert row.evidence_page_recall == 0.5
    assert row.evidence_page_reciprocal_rank == pytest.approx(1 / 3)
    assert row.reciprocal_rank == 1
    assert row.retrieved_chunks[2] == {"source": "a.pdf", "page": 10, "text": "Revenue was 42"}


def test_page_match_requires_source_and_does_not_guess_missing_metadata() -> None:
    pipeline = FakePipeline(
        [
            Document(page_content="Wrong filing", metadata={"source": "wrong.pdf", "page": 9}),
            Document(page_content="Unknown page", metadata={"source": "a.pdf"}),
        ]
    )
    golden = [
        {
            "question": "Revenue?",
            "expected_sources": ["a.pdf"],
            "evidence": [{"source": "a.pdf", "page": 10, "text": "Revenue was 42"}],
        }
    ]

    assert evaluate_retrieval(pipeline, golden)[0].evidence_page_hit is False


def test_unannotated_questions_are_excluded_from_evidence_averages() -> None:
    pipeline = FakePipeline([])
    golden = [
        {"question": "Legacy question", "expected_sources": ["a.pdf"]},
        {
            "question": "Annotated question",
            "expected_sources": ["a.pdf"],
            "evidence": [{"source": "a.pdf", "page": 1, "text": "Revenue"}],
        },
    ]

    rows = evaluate_retrieval(pipeline, golden)
    metrics = summarize_retrieval(rows)

    assert rows[0].evidence_page_hit is None
    assert metrics["questions"] == 2
    assert metrics["annotated_questions"] == 1
    assert metrics["evidence_page_recall"] == 0
    assert summarize_retrieval(rows[:1])["evidence_page_recall"] is None


@pytest.mark.parametrize("page", [0, -1, True, "10", None])
def test_load_golden_rejects_invalid_evidence_page(tmp_path: Path, page) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Q?",
                "expected_sources": ["a.pdf"],
                "evidence": [{"source": "a.pdf", "page": page, "text": "Revenue"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one-based integer page"):
        load_golden(path)


def test_committed_golden_preserves_financebench_evidence() -> None:
    evals = Path(__file__).resolve().parents[1] / "evals"
    upstream = {
        row["financebench_id"]: row
        for row in (
            json.loads(line)
            for line in (evals / "financebench_open_source.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    golden = load_golden(evals / "golden.jsonl")

    assert len(golden) == 69
    for row in golden:
        original = upstream[row["financebench_id"]]
        assert row["evidence"] == [
            {
                "source": e["doc_name"] + ".pdf",
                "page": e["evidence_page_num"] + 1,
                "text": e["evidence_text"],
            }
            for e in original["evidence"]
        ]


def test_cli_exports_evidence_metrics_and_full_chunks(tmp_path: Path, monkeypatch, capsys) -> None:
    golden = tmp_path / "golden.jsonl"
    output = tmp_path / "results" / "retrieval.json"
    golden.write_text(
        json.dumps(
            {
                "question": "Revenue?",
                "expected_sources": ["a.pdf"],
                "evidence": [{"source": "a.pdf", "page": 10, "text": "Revenue was 42"}],
            }
        ),
        encoding="utf-8",
    )
    pipeline = FakePipeline(
        [
            Document(page_content="Wrong page", metadata={"source": "a.pdf", "page": 0}),
        ]
    )
    settings = SimpleNamespace(
        retrieval_mode="dense",
        rerank_enabled=False,
        top_k=4,
        fetch_k=20,
        chunk_size=800,
        chunk_overlap=120,
        embedding_model="test-model",
    )
    monkeypatch.setattr(evaluation, "get_settings", lambda: settings)
    monkeypatch.setattr(evaluation, "RAGPipeline", lambda _: pipeline)
    monkeypatch.setattr(evaluation.sys.stdout, "reconfigure", lambda **_: None, raising=False)
    monkeypatch.setattr(
        evaluation.sys,
        "argv",
        [
            "rag-eval",
            "--golden",
            str(golden),
            "--output",
            str(output),
        ],
    )

    evaluation.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["metrics"]["filing_hit_rate"] == 1
    assert report["metrics"]["evidence_page_hit_rate"] == 0
    assert report["rows"][0]["retrieved_chunks"] == [
        {"source": "a.pdf", "page": 1, "text": "Wrong page"}
    ]
    assert "evidence-page hit@4=0.00" in capsys.readouterr().out
