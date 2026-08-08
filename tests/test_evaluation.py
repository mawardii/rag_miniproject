import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag.evaluation import evaluate_retrieval, load_golden


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
