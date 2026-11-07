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

from rag.retrieval.query_intent import extract_financial_terms, is_numeric_query

RRF_K = 60  # standard damping constant from the RRF paper; not worth tuning first


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _key(doc: Document) -> tuple:
    """Identity for fusion — the same chunk must merge across both result lists."""
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])


def rrf_fuse(ranked_lists: list[list[Document]], k: int) -> list[Document]:
    """Reciprocal Rank Fusion: merge ranked lists using only ranks, so neither
    scoring scale (cosine distance, BM25 score, ...) dominates.
    """
    fused: dict[tuple, float] = {}
    by_key: dict[tuple, Document] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, 1):
            key = _key(doc)
            by_key[key] = doc
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)

    ordered = sorted(fused, key=fused.__getitem__, reverse=True)
    return [by_key[key] for key in ordered[:k]]


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

        return rrf_fuse([dense, keyword], k)


class TableBoost:
    """Numeric-question boost: keyword search over table-tagged chunks, scoped
    to whatever filings the primary retrieval already surfaced.

    Diagnosed on the AES 10-K ("how many times did AES sell its inventory"):
    dense search buries the balance-sheet row past rank 400 even restricted to
    AES's own table chunks — embeddings don't discriminate well between
    different line items in the same company's tables, they all read as
    generic "financial numbers". BM25 on the specific term ("inventory") does
    discriminate, *but only once scoped to one filing's table chunks* — over
    the whole corpus, generic words ("Corporation", "FY2022") shared by every
    financial question drown it out (verified: unscoped BM25 ranked the same
    row #1746). Primary retrieval reliably finds the right filing even when it
    picks the wrong chunk, so that's the scope to search within.
    """

    def __init__(self, docs: list[Document]):
        self._docs = [d for d in docs if d.metadata.get("chunk_type") == "table"]
        self._bm25 = (
            BM25Okapi([_tokenize(d.page_content) for d in self._docs]) if self._docs else None
        )

    def boost(self, query: str, candidates: list[Document], k: int) -> list[Document]:
        if self._bm25 is None or not is_numeric_query(query):
            return candidates
        sources = {d.metadata.get("source") for d in candidates}
        scoped_idx = [i for i, d in enumerate(self._docs) if d.metadata.get("source") in sources]
        if not scoped_idx:
            return candidates

        terms = extract_financial_terms(query)
        query_tokens = _tokenize(" ".join(terms)) if terms else _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(scoped_idx, key=scores.__getitem__, reverse=True)[:k]
        table_hits = [self._docs[i] for i in ranked]
        return rrf_fuse([candidates, table_hits], k)
