from langchain_core.documents import Document

from rag.ingestion.tables import tag_chunk_type


def test_tag_chunk_type_flags_numeric_dense_chunk_as_table() -> None:
    chunk = Document(page_content="Cash and cash equivalents $ 1,374 $ 943 \nInventory 1,055 604")

    tag_chunk_type([chunk])

    assert chunk.metadata["chunk_type"] == "table"


def test_tag_chunk_type_flags_prose_chunk_as_prose() -> None:
    chunk = Document(
        page_content=(
            "The Company's results were driven by continued growth in the Data Center "
            "segment, primarily due to higher sales of server processors."
        )
    )

    tag_chunk_type([chunk])

    assert chunk.metadata["chunk_type"] == "prose"
