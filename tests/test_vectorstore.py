from pathlib import Path
from types import SimpleNamespace

import pytest

from rag.retrieval import vectorstore
from rag.retrieval.vectorstore import _corpus_hash, _swap_index_dir, compute_fingerprint


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


def test_swap_index_dir_replaces_existing_index(tmp_path: Path) -> None:
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "old.faiss").write_text("old", encoding="utf-8")
    new_dir = tmp_path / ".faiss_index-new"
    new_dir.mkdir()
    (new_dir / "index.faiss").write_text("new", encoding="utf-8")

    _swap_index_dir(new_dir, index_dir)

    assert not new_dir.exists()
    assert not (index_dir / "old.faiss").exists()
    assert (index_dir / "index.faiss").read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob(".faiss_index.backup-*"))


def test_swap_index_dir_rolls_back_existing_index_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "old.faiss").write_text("old", encoding="utf-8")
    new_dir = tmp_path / ".faiss_index-new"
    new_dir.mkdir()
    (new_dir / "index.faiss").write_text("new", encoding="utf-8")
    real_move = vectorstore.shutil.move

    def fail_new_index_move(src: str, dst: str) -> str:
        if Path(src) == new_dir:
            raise OSError("simulated replace failure")
        return str(real_move(src, dst))

    monkeypatch.setattr(vectorstore.shutil, "move", fail_new_index_move)

    with pytest.raises(OSError, match="simulated replace failure"):
        _swap_index_dir(new_dir, index_dir)

    assert (index_dir / "old.faiss").read_text(encoding="utf-8") == "old"
    assert not new_dir.exists()
