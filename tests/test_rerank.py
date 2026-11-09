from types import SimpleNamespace

from langchain_core.documents import Document

from rag.retrieval.rerank import _parse_ranking, llm_rerank


def test_parse_ranking_accepts_json_object() -> None:
    assert _parse_ranking('{"ranking": [3, 1, 2]}') == [3, 1, 2]


def test_parse_ranking_accepts_json_list() -> None:
    assert _parse_ranking("[2, 1]") == [2, 1]


def test_parse_ranking_falls_back_to_digits() -> None:
    assert _parse_ranking("best passages are 4, then 2") == [4, 2]


def test_reranker_sees_filing_page_and_full_chunk_and_preserves_unranked_candidates() -> None:
    docs = [
        Document(
            page_content="same financial terms", metadata={"source": "OTHER_2021.pdf", "page": 1}
        ),
        Document(
            page_content="x" * 850 + "important final row",
            metadata={"source": "BESTBUY_2023_10K.pdf", "page": 28},
        ),
        Document(page_content="additional candidate", metadata={"source": "other.pdf", "page": 0}),
    ]
    captured = []

    class FakeLLM:
        def bind(self, **kwargs):
            return self

        def invoke(self, messages):
            captured.extend(messages)
            return SimpleNamespace(content='{"ranking": [2, 2, 99]}')

    result = llm_rerank(FakeLLM(), "Best Buy FY2023 cash flows?", docs, top_k=2)

    assert result == [docs[1], docs[0]]
    assert "BESTBUY_2023_10K.pdf, p.29" in captured[1].content
    assert "OTHER_2021.pdf, p.2" in captured[1].content
    assert "important final row" in captured[1].content
