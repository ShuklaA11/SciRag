"""Tests for src/pipeline/chunker.py and src/pipeline/embedder.py.

Chunker tests run always (they need the SPECTER2 tokenizer — ~2 MB,
downloaded + cached on first run).

Embedder tests are gated by SCIRAG_RUN_HEAVY=1 because they download the
full SPECTER2 model (~440 MB) and run a forward pass. Skipped by default
so `pytest tests/ -v` stays fast and offline-friendly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.chunker import chunk_paper
from src.pipeline.embedder import EMBED_DIM, MODEL_NAME, Specter2Embedder

FIXTURE_XML = Path("data/grobid_output/qasper/1503.00841.xml")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def sample_tei() -> str:
    if not FIXTURE_XML.exists():
        pytest.skip(f"TEI fixture missing: {FIXTURE_XML}")
    return FIXTURE_XML.read_text()


class TestChunker:
    def test_real_paper_chunks(self, tokenizer, sample_tei):
        chunks = chunk_paper(sample_tei, tokenizer, chunk_size=512, overlap=64)
        assert len(chunks) > 0
        for c in chunks:
            assert c["token_count"] <= 512
            assert c["text"].strip() != ""
        assert [c["chunk_idx"] for c in chunks] == list(range(len(chunks)))

    def test_deterministic(self, tokenizer, sample_tei):
        a = chunk_paper(sample_tei, tokenizer)
        b = chunk_paper(sample_tei, tokenizer)
        assert a == b

    def test_empty_body_returns_empty(self, tokenizer):
        empty_tei = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            "<text><body/></text></TEI>"
        )
        assert chunk_paper(empty_tei, tokenizer) == []

    def test_short_paper_single_chunk(self, tokenizer):
        tei = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            "<teiHeader><fileDesc><titleStmt><title>Tiny Paper</title>"
            "</titleStmt></fileDesc><profileDesc><abstract><p>This is a short abstract "
            "about transformers and retrieval augmented generation.</p></abstract>"
            "</profileDesc></teiHeader>"
            "<text><body><div><head>Intro</head><p>We propose a method. "
            "It is very simple and easy to evaluate.</p></div></body></text></TEI>"
        )
        chunks = chunk_paper(tei, tokenizer, chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert "tiny paper" in chunks[0]["text"].lower()
        assert chunks[0]["chunk_idx"] == 0

    def test_overlap_honored(self, tokenizer, sample_tei):
        chunks = chunk_paper(sample_tei, tokenizer, chunk_size=128, overlap=32)
        if len(chunks) < 2:
            pytest.skip("fixture too short to test overlap")
        ids_a = tokenizer.encode(chunks[0]["text"], add_special_tokens=False)
        ids_b = tokenizer.encode(chunks[1]["text"], add_special_tokens=False)
        # the tail of chunk 0 should share tokens with the head of chunk 1.
        # after decode/re-encode, exact equality is fragile, so assert the
        # first token of chunk 1 also appears in the last 48 tokens of chunk 0.
        assert ids_b[0] in ids_a[-48:]

    def test_overlap_validation(self, tokenizer):
        with pytest.raises(ValueError):
            chunk_paper("<TEI/>", tokenizer, chunk_size=100, overlap=100)

    def test_preserves_case_hyphens_and_bibref(self, tokenizer):
        # Regression for the encode/decode round-trip bug that lowercased
        # text, spaced out hyphens, and stripped BIBREF tokens — corrupting
        # both the chunk text and the embedded representation.
        tei = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            "<teiHeader><fileDesc><titleStmt><title>"
            "Cross-Lingual Pre-Training for NMT</title>"
            "</titleStmt></fileDesc></teiHeader>"
            "<text><body><div><head>Background</head>"
            "<p>We compare with related approaches of pivoting BIBREF19 "
            "and cross-lingual transfer without pretraining BIBREF16.</p>"
            "</div></body></text></TEI>"
        )
        chunks = chunk_paper(tei, tokenizer, chunk_size=512, overlap=64)
        assert len(chunks) >= 1
        joined = " ".join(c["text"] for c in chunks)
        assert "Cross-Lingual" in joined
        assert "Pre-Training" in joined
        assert "BIBREF19" in joined
        assert "BIBREF16" in joined


@pytest.mark.skipif(
    os.environ.get("SCIRAG_RUN_HEAVY") != "1",
    reason="heavy: downloads SPECTER2 model (~440 MB). Set SCIRAG_RUN_HEAVY=1 to run.",
)
class TestEmbedder:
    def test_encode_shape_and_norm(self):
        emb = Specter2Embedder()
        out = emb.encode(["hello world", "scientific paper about transformers"])
        assert out.shape == (2, EMBED_DIM)
        assert out.dtype == np.float32
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_encode_empty(self):
        emb = Specter2Embedder()
        out = emb.encode([])
        assert out.shape == (0, EMBED_DIM)
