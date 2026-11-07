"""Flag financial-statement table chunks for retrieval routing.

Diagnosed on the AES 10-K: the chunk containing "Inventory 1,055 604" ranked
#422/13.8k on dense search and #1746 on BM25 for "how many times did AES sell
its inventory" — nowhere near top_k, even though the chunk itself is intact and
correct. Embeddings and whole-corpus BM25 both treat it as generic financial
text; tagging it lets retrieval route numeric questions toward table chunks
specifically (see retrieval/retrievers.py:TableBoost).

An earlier version of this module also rewrote each detected row into a
narrative sentence ("Inventory was 1,055 in 2022, 604 in 2021.") before
chunking, on the theory that embeddings would match a sentence better than a
bare row. Measured on the 69-question eval set, that cost an *extra* 0.03
hit@4 beyond tagging alone (0.90 -> 0.87) — the same generic phrasing repeated
across every table row in the corpus made unrelated table chunks *more*
semantically similar to each other, actively hurting cross-filing
discrimination. Dropped; tagging + TableBoost's keyword search already surface
the raw row without it.
"""

from langchain_core.documents import Document

_DIGIT_DENSITY_THRESHOLD = 0.12


def tag_chunk_type(chunks: list[Document]) -> None:
    """Flag chunks likely to be financial-statement tables by digit density."""
    for chunk in chunks:
        text = chunk.page_content
        digits = sum(c.isdigit() for c in text)
        density = digits / len(text) if text else 0.0
        chunk.metadata["chunk_type"] = "table" if density > _DIGIT_DENSITY_THRESHOLD else "prose"
