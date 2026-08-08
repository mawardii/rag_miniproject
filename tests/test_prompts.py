from langchain_core.documents import Document

from rag.generation.prompts import build_user_message, format_context


def test_format_context_numbers_sources_and_one_indexes_pages() -> None:
    docs = [
        Document(
            page_content="Revenue increased because Data Center sales grew.",
            metadata={"source": "AMD_2022_10K.pdf", "page": 42},
        ),
        Document(
            page_content="Gross margin declined due to one-time charges.",
            metadata={"source": "3M_2022_10K.pdf"},
        ),
    ]

    context = format_context(docs)

    assert "[1] (AMD_2022_10K.pdf, p.43)" in context
    assert "[2] (3M_2022_10K.pdf)" in context
    assert "Revenue increased because Data Center sales grew." in context


def test_build_user_message_includes_context_and_question() -> None:
    docs = [
        Document(
            page_content="Operating income was affected by restructuring costs.",
            metadata={"source": "filing.pdf", "page": 0},
        )
    ]

    message = build_user_message("What changed operating income?", docs)

    assert "Context excerpts:" in message
    assert "Operating income was affected by restructuring costs." in message
    assert "Question: What changed operating income?" in message
