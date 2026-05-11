"""Tests for the multi-hop retriever.

Multi-hop: decompose → per-subQ retrieve+rerank → merge+dedup → final
rerank-vs-original → top-k. Atomic case (single sub-Q) collapses to
single-hop. All collaborators (decomposer, index, reranker) are faked
so tests pin behavior without loading models.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.retrieval.multihop import MultiHopRetriever


class _FakeDecomposer:
    def __init__(self, sub_qs: list[str]) -> None:
        self.sub_qs = sub_qs
        self.calls: list[str] = []

    def decompose(self, q: str) -> list[str]:
        self.calls.append(q)
        return list(self.sub_qs)


class _FakeIndex:
    """Returns a pre-canned list per query, ignoring k/paper_ids."""

    def __init__(self, results_by_query: dict[str, list[dict]]) -> None:
        self.results_by_query = results_by_query
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query, k=5, paper_ids=None, section_types=None):
        self.search_calls.append((query, k))
        return list(self.results_by_query.get(query, []))


@dataclass(frozen=True)
class _RR:
    chunk: dict
    bi_score: float
    ce_score: float


class _FakeReranker:
    """Reranks by (query, chunk_id) → score lookup table."""

    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self.scores = scores
        self.rerank_calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, candidates, top_k=None, batch_size=32):
        self.rerank_calls.append(
            (query, [c["chunk_id"] for c in candidates])
        )
        scored = [
            _RR(
                chunk=c,
                bi_score=float(c.get("score", 0.0)),
                ce_score=self.scores.get((query, c["chunk_id"]), 0.0),
            )
            for c in candidates
        ]
        scored.sort(key=lambda r: r.ce_score, reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return scored


def _chunk(cid: str, text: str = "x") -> dict:
    return {"chunk_id": cid, "arxiv_id": "p1", "chunk_idx": 0,
            "text": text, "section_type": None, "score": 0.0}


def test_atomic_single_subq_skips_final_rerank():
    """One sub-Q equal to the original → only one rerank pass."""
    q = "What dataset?"
    idx = _FakeIndex({q: [_chunk("c1"), _chunk("c2"), _chunk("c3")]})
    rr = _FakeReranker({(q, "c1"): 0.1, (q, "c2"): 0.9, (q, "c3"): 0.5})
    mh = MultiHopRetriever(
        decomposer=_FakeDecomposer([q]),
        flat_index=idx, reranker=rr,
        retrieve_k=10, top_k=2,
    )
    out, meta = mh.retrieve(q)
    assert [c["chunk_id"] for c in out] == ["c2", "c3"]
    assert meta["sub_questions"] == [q]
    assert meta["n_sub_questions"] == 1
    # exactly one rerank call (per-subQ); no separate "final" rerank
    assert len(rr.rerank_calls) == 1


def test_multi_subq_merges_and_dedupes():
    q = "compound"
    sub_a = "subA"
    sub_b = "subB"
    idx = _FakeIndex({
        sub_a: [_chunk("c1"), _chunk("c2")],
        sub_b: [_chunk("c2"), _chunk("c3")],  # c2 overlaps
    })
    # final rerank scores against original q
    rr = _FakeReranker({
        (sub_a, "c1"): 0.9, (sub_a, "c2"): 0.8,
        (sub_b, "c2"): 0.85, (sub_b, "c3"): 0.7,
        (q, "c1"): 0.4, (q, "c2"): 0.95, (q, "c3"): 0.5,
    })
    mh = MultiHopRetriever(
        decomposer=_FakeDecomposer([sub_a, sub_b]),
        flat_index=idx, reranker=rr,
        retrieve_k=10, top_k=3,
    )
    out, meta = mh.retrieve(q)
    cids = [c["chunk_id"] for c in out]
    # final order is by rerank-vs-original: c2 (0.95) > c3 (0.5) > c1 (0.4)
    assert cids == ["c2", "c3", "c1"]
    assert meta["n_sub_questions"] == 2
    assert meta["n_unique_candidates"] == 3  # c1, c2, c3 after dedup


def test_top_k_cap_honored():
    q = "compound"
    sub_a = "subA"
    sub_b = "subB"
    idx = _FakeIndex({
        sub_a: [_chunk(f"c{i}") for i in range(5)],
        sub_b: [_chunk(f"c{i}") for i in range(5, 10)],
    })
    scores = {(sub_a, f"c{i}"): float(i) for i in range(5)}
    scores.update({(sub_b, f"c{i}"): float(i) for i in range(5, 10)})
    scores.update({(q, f"c{i}"): float(i) for i in range(10)})
    rr = _FakeReranker(scores)
    mh = MultiHopRetriever(
        decomposer=_FakeDecomposer([sub_a, sub_b]),
        flat_index=idx, reranker=rr,
        retrieve_k=10, top_k=3,
    )
    out, _ = mh.retrieve(q)
    assert len(out) == 3
    # highest-scoring three by (q, ci)
    assert [c["chunk_id"] for c in out] == ["c9", "c8", "c7"]


def test_paper_ids_passed_to_index():
    q = "What?"
    idx = _FakeIndex({q: [_chunk("c1")]})
    rr = _FakeReranker({(q, "c1"): 0.5})
    mh = MultiHopRetriever(
        decomposer=_FakeDecomposer([q]),
        flat_index=idx, reranker=rr,
    )
    mh.retrieve(q, paper_ids={"p1", "p2"})
    assert len(idx.search_calls) == 1


def test_no_reranker_uses_bi_encoder_score_only():
    """If reranker is None, dedup + sort by bi-encoder score."""
    q = "compound"
    sub_a = "subA"
    sub_b = "subB"
    c1 = _chunk("c1"); c1["score"] = 0.9
    c2 = _chunk("c2"); c2["score"] = 0.7
    c3 = _chunk("c3"); c3["score"] = 0.8
    idx = _FakeIndex({sub_a: [c1, c2], sub_b: [c3]})
    mh = MultiHopRetriever(
        decomposer=_FakeDecomposer([sub_a, sub_b]),
        flat_index=idx, reranker=None,
        retrieve_k=10, top_k=2,
    )
    out, _ = mh.retrieve(q)
    assert [c["chunk_id"] for c in out] == ["c1", "c3"]


def test_meta_contains_latency_keys():
    q = "What?"
    idx = _FakeIndex({q: [_chunk("c1")]})
    rr = _FakeReranker({(q, "c1"): 0.5})
    mh = MultiHopRetriever(
        decomposer=_FakeDecomposer([q]),
        flat_index=idx, reranker=rr,
    )
    _, meta = mh.retrieve(q)
    assert "decompose_ms" in meta
    assert "retrieve_ms" in meta
    assert "rerank_ms" in meta


def test_empty_retrieval_returns_empty():
    q = "What?"
    idx = _FakeIndex({})  # no results for any sub-Q
    rr = _FakeReranker({})
    mh = MultiHopRetriever(
        decomposer=_FakeDecomposer([q]),
        flat_index=idx, reranker=rr,
    )
    out, meta = mh.retrieve(q)
    assert out == []
    assert meta["n_unique_candidates"] == 0
