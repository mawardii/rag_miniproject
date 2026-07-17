"""Retrieval strategies: dense-only (FAISS) or hybrid (BM25 + dense, RRF-fused).

Why hybrid is the industry default: embeddings capture meaning but blur exact
terms — dataset names, model versions, acronyms ("RCD-IIUM", "YOLOv7"). BM25
nails exact terms but has zero notion of synonymy. Reciprocal Rank Fusion merges
the two ranked lists using only ranks, so neither scoring scale dominates.
"""

import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

RRF_K = 60  # standard damping constant from the RRF paper; not worth tuning first


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _key(doc: Document) -> tuple:
    """Identity for fusion — the same chunk must merge across both result lists."""
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])


class HybridRetriever:
    def __init__(self, index: FAISS):
        self._index = index
        # FAISS already stores every chunk in its docstore; reuse it so BM25 and
        # dense search share exactly one corpus (no second persistence to drift).
        self._docs: list[Document] = list(index.docstore._dict.values())
        self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in self._docs])

    def retrieve(self, query: str, k: int) -> list[Document]:
        dense = self._index.similarity_search(query, k=k)

        scores = self._bm25.get_scores(_tokenize(query))
        keyword_idx = sorted(range(len(self._docs)), key=scores.__getitem__, reverse=True)[:k]
        keyword = [self._docs[i] for i in keyword_idx]

        fused: dict[tuple, float] = {}
        by_key: dict[tuple, Document] = {}
        for ranked_list in (dense, keyword):
            for rank, doc in enumerate(ranked_list, 1):
                key = _key(doc)
                by_key[key] = doc
                fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)

        ordered = sorted(fused, key=fused.__getitem__, reverse=True)
        return [by_key[key] for key in ordered[:k]]
