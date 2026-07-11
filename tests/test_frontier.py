"""SB-D1: brainstorm frontier. Fake embedder maps queries to controlled unit
vectors, so cosine dedup is deterministic and model-free. Guards the queue
contract: dedup on add (vs seen, pending, and intra-batch), drain on pop,
exhaustion."""

from __future__ import annotations

import numpy as np

from src.brainstorm import Frontier


class FakeEmbedder:
    """encode_query → looked-up unit vectors. Missing query raises (tests must
    declare every vector, keeping cosine relationships explicit)."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._v = {q: _unit(v) for q, v in vectors.items()}

    def encode_query(self, queries: list[str], batch_size: int = 8) -> np.ndarray:
        return np.array([self._v[q] for q in queries])


def _unit(v: list[float]) -> np.ndarray:
    arr = np.array(v, dtype=float)
    return arr / np.linalg.norm(arr)


# Orthogonal basis 'a','b','c' (cosine 0) + 'a2' near 'a' (cosine ~0.9993).
_VECS = {
    "a": [1, 0, 0],
    "a2": [0.98, 0.02, 0.0],
    "b": [0, 1, 0],
    "c": [0, 0, 1],
}


def test_add_keeps_distinct_and_returns_them():
    f = Frontier(FakeEmbedder(_VECS))
    assert f.add(["a", "b"]) == ["a", "b"]
    assert not f.is_exhausted


def test_add_rejects_near_duplicate():
    f = Frontier(FakeEmbedder(_VECS))
    f.add(["a"])
    assert f.add(["a2"]) == []  # cosine(a, a2) >= 0.85 → dropped


def test_intra_batch_dedup():
    f = Frontier(FakeEmbedder(_VECS))
    assert f.add(["a", "a2", "b"]) == ["a", "b"]  # a2 dropped against a in-batch


def test_blank_candidates_skipped():
    f = Frontier(FakeEmbedder(_VECS))
    assert f.add(["a", "   ", ""]) == ["a"]


def test_pop_batch_drains_and_marks_seen():
    f = Frontier(FakeEmbedder(_VECS))
    f.add(["a", "b", "c"])
    assert f.pop_batch(2) == ["a", "b"]
    assert f.seen == frozenset({"a", "b"})
    assert f.pop_batch(2) == ["c"]  # only one left
    assert f.is_exhausted


def test_dedup_holds_against_dispatched_directions():
    f = Frontier(FakeEmbedder(_VECS))
    f.add(["a"])
    f.pop_batch(1)  # 'a' now dispatched, not pending
    assert f.add(["a2"]) == []  # still deduped against seen
    assert f.is_exhausted  # nothing re-queued


def test_pop_nonpositive_returns_empty():
    f = Frontier(FakeEmbedder(_VECS))
    f.add(["a"])
    assert f.pop_batch(0) == []
    assert not f.is_exhausted
