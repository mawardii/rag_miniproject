from pathlib import Path
from types import SimpleNamespace

from rag.retrieval.vectorstore import _corpus_hash, compute_fingerprint


def test_corpus_hash_is_content_based_and_deterministic(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "b.txt").write_text("second", encoding="utf-8")
    (docs_dir / "a.txt").write_text("first", encoding="utf-8")

    first_hash = _corpus_hash(docs_dir)
    second_hash = _corpus_hash(docs_dir)

    assert first_hash == second_hash

    (docs_dir / "a.txt").write_text("changed", encoding="utf-8")

    assert _corpus_hash(docs_dir) != first_hash


def test_compute_fingerprint_includes_rebuild_inputs(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "filing.txt").write_text("content", encoding="utf-8")
    settings = SimpleNamespace(
        docs_dir=docs_dir,
        embedding_model="text-embedding-3-small",
        chunk_size=800,
        chunk_overlap=120,
    )

    fingerprint = compute_fingerprint(settings)

    assert fingerprint["embedding_model"] == "text-embedding-3-small"
    assert fingerprint["chunk_size"] == 800
    assert fingerprint["chunk_overlap"] == 120
    assert fingerprint["corpus_hash"] == _corpus_hash(docs_dir)
