"""Tests for the cross-encoder reranker.

The actual MiniLM model load is gated behind ``SCIRAG_RUN_HEAVY=1`` to
keep the default test run fast (mirrors the pattern used in
test_chunker_embedder.py). Light tests use a fake model to verify
ordering, candidate handling, and top-k truncation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from src.retrieval.cross_encoder_reranker import (
    CrossEncoderReranker,
    RerankResult,
)


class _FakeModel:
    """Stand-in for sentence-transformers' CrossEncoder.

    Returns the score baked into each candidate's ``text`` (we encode the
    desired score in the candidate dict and read it back) so tests can
    pin behavior without loading a real model.
    """

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self.scores_by_text = scores_by_text

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return [self.scores_by_text[text] for _, text in pairs]


@pytest.fixture
def fake_reranker(monkeypatch):
    def _factory(scores_by_text: dict[str, float]) -> CrossEncoderReranker:
        # Bypass __init__ so we don't actually load a model.
        r = CrossEncoderReranker.__new__(CrossEncoderReranker)
        r.model_name = "fake"
        r.device = "cpu"
        r.max_length = 512
        r.model = _FakeModel(scores_by_text)
        return r
    return _factory


def test_rerank_orders_by_ce_score(fake_reranker):
    candidates = [
        {"text": "low", "score": 0.9},
        {"text": "high", "score": 0.3},
        {"text": "mid", "score": 0.6},
    ]
    r = fake_reranker({"low": 0.1, "high": 0.9, "mid": 0.5})
    out = r.rerank("q", candidates)
    assert [x.chunk["text"] for x in out] == ["high", "mid", "low"]
    assert out[0].ce_score == pytest.approx(0.9)
    assert out[0].bi_score == pytest.approx(0.3)


def test_rerank_top_k_truncates(fake_reranker):
    candidates = [{"text": f"c{i}", "score": 0.0} for i in range(5)]
    r = fake_reranker({f"c{i}": float(i) for i in range(5)})
    out = r.rerank("q", candidates, top_k=2)
    assert len(out) == 2
    assert [x.chunk["text"] for x in out] == ["c4", "c3"]


def test_rerank_empty_returns_empty(fake_reranker):
    r = fake_reranker({})
    assert r.rerank("q", []) == []


def test_rerankresult_is_immutable():
    rr = RerankResult(chunk={"text": "x"}, bi_score=0.1, ce_score=0.2)
    with pytest.raises(Exception):
        rr.ce_score = 0.5  # frozen dataclass


@pytest.mark.skipif(
    os.environ.get("SCIRAG_RUN_HEAVY") != "1",
    reason="loads MiniLM-L6 (~80MB); set SCIRAG_RUN_HEAVY=1 to run",
)
def test_real_minilm_reranks_obvious_pair():
    r = CrossEncoderReranker()
    candidates = [
        {"text": "Apples are fruit grown on trees.", "score": 0.0},
        {"text": "We trained a transformer on QASPER for retrieval.",
         "score": 0.0},
    ]
    out = r.rerank("How was the QASPER retrieval model trained?", candidates)
    assert "QASPER" in out[0].chunk["text"]
    assert out[0].ce_score > out[1].ce_score
