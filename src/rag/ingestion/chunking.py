"""Split documents into retrieval-sized chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Recursive splitting: tries paragraph breaks first, then lines, then words.

    This replaces CharacterTextSplitter(separator="\\n"), which could only cut on
    newlines — PDF paragraphs without newlines became oversized chunks that blew
    past chunk_size and diluted retrieval relevance.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
