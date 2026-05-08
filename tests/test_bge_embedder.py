"""Tests for src/pipeline/bge_embedder.py.

All tests are gated by SCIRAG_RUN_HEAVY=1 because they download the
BGE-base-en-v1.5 model (~440 MB) and run a forward pass.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.pipeline.bge_embedder import EMBED_DIM, QUERY_PREFIX, BGEEmbedder


@pytest.mark.skipif(
    os.environ.get("SCIRAG_RUN_HEAVY") != "1",
    reason="heavy: downloads BGE model (~440 MB). Set SCIRAG_RUN_HEAVY=1 to run.",
)
class TestBGEEmbedder:
    def test_encode_shape_and_norm(self):
        emb = BGEEmbedder()
        out = emb.encode(["hello world", "scientific paper about transformers"])
        assert out.shape == (2, EMBED_DIM)
        assert out.dtype == np.float32
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_encode_empty(self):
        emb = BGEEmbedder()
        out = emb.encode([])
        assert out.shape == (0, EMBED_DIM)

    def test_query_prefix_changes_embedding(self):
        # encode_query() must apply the BGE retrieval instruction; the
        # resulting vector must differ from the plain-encoded version.
        emb = BGEEmbedder()
        text = "what dataset is used for evaluation"
        plain = emb.encode([text])
        as_query = emb.encode_query([text])
        assert plain.shape == as_query.shape == (1, EMBED_DIM)
        cos = float(np.dot(plain[0], as_query[0]))
        assert cos < 0.999, "query prefix had no effect on the embedding"

    def test_query_prefix_matches_manual_prefix(self):
        # encode_query(q) must equal encode(QUERY_PREFIX + q) up to fp noise.
        emb = BGEEmbedder()
        q = "how is BLEU computed"
        via_helper = emb.encode_query([q])
        via_manual = emb.encode([QUERY_PREFIX + q])
        cos = float(np.dot(via_helper[0], via_manual[0]))
        assert cos > 0.9999
