from langchain_core.documents import Document

from rag.retrieval.retrievers import TableBoost, rrf_fuse


def _doc(source: str, page: int, text: str, chunk_type: str = "prose") -> Document:
    return Document(
        page_content=text, metadata={"source": source, "page": page, "chunk_type": chunk_type}
    )


def test_rrf_fuse_dedupes_the_same_chunk_across_lists() -> None:
    shared = _doc("a.pdf", 1, "shared chunk")
    list_a = [shared, _doc("a.pdf", 2, "only in a")]
    list_b = [shared, _doc("a.pdf", 3, "only in b")]

    fused = rrf_fuse([list_a, list_b], k=10)

    sources = [(d.metadata["source"], d.metadata["page"]) for d in fused]
    assert sources.count(("a.pdf", 1)) == 1


def test_rrf_fuse_ranks_docs_appearing_in_both_lists_above_single_list_docs() -> None:
    shared = _doc("a.pdf", 1, "shared chunk")
    only_a = _doc("a.pdf", 2, "only in a, ranked first there")
    only_b = _doc("a.pdf", 3, "only in b, ranked first there")
    list_a = [only_a, shared]
    list_b = [only_b, shared]

    fused = rrf_fuse([list_a, list_b], k=10)

    assert fused[0].metadata["page"] == 1


def test_rrf_fuse_truncates_to_k() -> None:
    docs = [_doc("a.pdf", i, f"chunk {i}") for i in range(5)]

    fused = rrf_fuse([docs], k=2)

    assert len(fused) == 2


def _aes_corpus() -> list[Document]:
    return [
        _doc("aes.pdf", 83, "Selected Financial Data 2022 2021 2020 revenue debt equity", "table"),
        _doc(
            "aes.pdf", 129, "Balance Sheets Inventory 1,055 604 cash equivalents prepaid", "table"
        ),
        _doc(
            "aes.pdf", 200, "Inventory is valued primarily using the average-cost method", "prose"
        ),
        _doc("other.pdf", 40, "Inventory turnover ratio revenue debt equity table", "table"),
    ]


def test_table_boost_ignores_non_numeric_questions() -> None:
    boost = TableBoost(_aes_corpus())
    candidates = [_doc("aes.pdf", 1, "narrative context")]

    result = boost.boost("Who are AES's main competitors in the energy market?", candidates, k=2)

    assert result == candidates


def test_table_boost_scopes_to_candidate_sources_only() -> None:
    boost = TableBoost(_aes_corpus())
    candidates = [_doc("aes.pdf", 1, "narrative context about AES")]

    result = boost.boost("How many times did AES sell its inventory?", candidates, k=3)

    assert all(d.metadata["source"] == "aes.pdf" for d in result)


def test_table_boost_surfaces_the_matching_table_row() -> None:
    boost = TableBoost(_aes_corpus())
    candidates = [
        _doc("aes.pdf", 83, "Selected Financial Data 2022 2021 2020 revenue debt equity", "table")
    ]

    result = boost.boost("How many times did AES sell its inventory?", candidates, k=2)

    assert any(d.metadata["page"] == 129 for d in result)


def test_table_boost_returns_candidates_unchanged_when_no_table_chunks_exist() -> None:
    boost = TableBoost([_doc("aes.pdf", 1, "just prose, no tables here")])
    candidates = [_doc("aes.pdf", 1, "just prose, no tables here")]

    result = boost.boost("How many times did AES sell its inventory?", candidates, k=2)

    assert result == candidates
