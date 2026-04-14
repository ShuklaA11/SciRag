"""Tests for src/retrieval/flat_index.py.

Uses a fake 8-dim embedder so the test stays offline and fast — no
SPECTER2 download. Builds a tiny FAISS index in tmp_path, then exercises
load + search through the FlatIndex wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from src.retrieval.flat_index import FlatIndex


class FakeEmbedder:
    """Deterministic 8-dim embedder. Maps a fixed vocabulary to one-hot
    rows; unknown words contribute nothing. Output is L2-normalized."""

    VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    DIM = 8

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            tokens = t.lower().split()
            for tok in tokens:
                if tok in self.VOCAB:
                    out[i, self.VOCAB.index(tok)] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def _build_tiny_index(tmp_path: Path, chunks: list[dict], embedder: FakeEmbedder) -> Path:
    out_dir = tmp_path / "flat"
    out_dir.mkdir()
    vecs = embedder.encode([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(embedder.DIM)
    index.add(vecs)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with (out_dir / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return out_dir


def _patch_flat_index_dim(monkeypatch):
    """FlatIndex hardcodes EMBED_DIM=768 in the dim check; we relax that
    by stubbing the embedder dim check via the constructor (FlatIndex
    actually compares to self.index.d, so no patch needed). This helper
    exists only to keep tests obvious."""
    pass


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def tiny_chunks():
    return [
        {"chunk_id": "p1::0", "arxiv_id": "p1", "chunk_idx": 0,
         "text": "alpha beta", "token_count": 2},
        {"chunk_id": "p1::1", "arxiv_id": "p1", "chunk_idx": 1,
         "text": "gamma delta", "token_count": 2},
        {"chunk_id": "p2::0", "arxiv_id": "p2", "chunk_idx": 0,
         "text": "epsilon zeta eta", "token_count": 3},
    ]


class TestFlatIndex:
    def test_build_and_search_roundtrip(self, tmp_path, fake_embedder, tiny_chunks):
        idx_dir = _build_tiny_index(tmp_path, tiny_chunks, fake_embedder)
        fi = FlatIndex(idx_dir, embedder=fake_embedder)
        results = fi.search("alpha beta", k=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "p1::0"
        assert results[0]["arxiv_id"] == "p1"
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_search_k(self, tmp_path, fake_embedder, tiny_chunks):
        idx_dir = _build_tiny_index(tmp_path, tiny_chunks, fake_embedder)
        fi = FlatIndex(idx_dir, embedder=fake_embedder)
        results = fi.search("epsilon", k=2)
        assert len(results) == 2
        assert results[0]["chunk_id"] == "p2::0"  # best match
        # scores monotonically decreasing
        assert results[0]["score"] >= results[1]["score"]

    def test_chunks_jsonl_alignment(self, tmp_path, fake_embedder, tiny_chunks):
        idx_dir = _build_tiny_index(tmp_path, tiny_chunks, fake_embedder)
        fi = FlatIndex(idx_dir, embedder=fake_embedder)
        assert fi.index.ntotal == len(tiny_chunks)
        for i, c in enumerate(tiny_chunks):
            assert fi.chunks[i]["chunk_id"] == c["chunk_id"]

    def test_search_caps_k_to_ntotal(self, tmp_path, fake_embedder, tiny_chunks):
        idx_dir = _build_tiny_index(tmp_path, tiny_chunks, fake_embedder)
        fi = FlatIndex(idx_dir, embedder=fake_embedder)
        results = fi.search("alpha", k=999)
        assert len(results) == len(tiny_chunks)

    def test_misalignment_raises(self, tmp_path, fake_embedder, tiny_chunks):
        idx_dir = _build_tiny_index(tmp_path, tiny_chunks, fake_embedder)
        # corrupt jsonl by appending an extra row
        with (idx_dir / "chunks.jsonl").open("a") as f:
            f.write(json.dumps({"chunk_id": "x::0", "arxiv_id": "x",
                                "chunk_idx": 0, "text": "extra",
                                "token_count": 1}) + "\n")
        with pytest.raises(ValueError, match="misalignment"):
            FlatIndex(idx_dir, embedder=fake_embedder)

    def test_missing_index_raises(self, tmp_path, fake_embedder):
        with pytest.raises(FileNotFoundError):
            FlatIndex(tmp_path, embedder=fake_embedder)
