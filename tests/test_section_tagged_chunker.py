"""Tests for chunk_paper_section_tagged.

Two key properties:
  1. Parity with chunk_paper (flat): same chunk count, same token_count
     per chunk, same chunk text. Only added fields are section_type and
     section_head.
  2. Section tagging is correct: the dominant section by character
     overlap wins, ties go to the earliest section in document order.
"""

from __future__ import annotations

import pytest

from src.pipeline.chunker import chunk_paper
from src.pipeline.section_chunker import (
    chunk_paper_section_tagged,
    _dominant_section,
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    from src.pipeline.embedder import MODEL_NAME
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def _build_tei(title: str, abstract: str, sections: list[tuple[str, str]]) -> str:
    abs_xml = (
        f"<profileDesc><abstract><p>{abstract}</p></abstract></profileDesc>"
        if abstract else "<profileDesc/>"
    )
    sec_xml = "".join(
        f"<div><head>{head}</head><p>{body}</p></div>" if head
        else f"<div><p>{body}</p></div>"
        for head, body in sections
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
        f"<teiHeader><fileDesc><titleStmt><title>{title}</title>"
        f"</titleStmt></fileDesc>{abs_xml}</teiHeader>"
        f"<text><body>{sec_xml}</body></text></TEI>"
    )


class TestDominantSection:
    def test_picks_max_overlap(self):
        spans = [(0, 100, "abstract", "Abstract"),
                 (100, 500, "method", "Methodology")]
        sec, _ = _dominant_section(50, 400, spans)
        assert sec == "method"

    def test_ties_break_earliest(self):
        spans = [(0, 100, "introduction", "Intro"),
                 (100, 200, "method", "Method")]
        sec, _ = _dominant_section(0, 200, spans)
        assert sec == "introduction"

    def test_no_overlap_returns_other(self):
        spans = [(0, 100, "abstract", "Abstract")]
        sec, head = _dominant_section(200, 300, spans)
        assert sec == "other"
        assert head == "[untitled]"


class TestParityWithFlatChunker:
    def test_chunk_count_and_text_match_flat(self, tokenizer):
        tei = _build_tei(
            "A Cool Paper",
            "We propose a new method for retrieval.",
            [
                ("Introduction", "Background context goes here. " * 10),
                ("Methodology", "We use a transformer model. " * 30),
                ("Experiments", "We train on QASPER. " * 20),
                ("Results", "Recall@5 is high. " * 15),
                ("Conclusion", "We conclude well. " * 5),
            ],
        )
        flat_chunks = chunk_paper(tei, tokenizer, chunk_size=128, overlap=16)
        tagged_chunks = chunk_paper_section_tagged(
            tei, tokenizer, chunk_size=128, overlap=16,
        )
        assert len(flat_chunks) == len(tagged_chunks)
        for f, t in zip(flat_chunks, tagged_chunks):
            assert f["chunk_idx"] == t["chunk_idx"]
            assert f["token_count"] == t["token_count"]
            assert f["text"] == t["text"]

    def test_each_tagged_chunk_has_section_metadata(self, tokenizer):
        tei = _build_tei(
            "T", "abs.",
            [("Methodology", "we propose a method. " * 100)],
        )
        chunks = chunk_paper_section_tagged(tei, tokenizer)
        for c in chunks:
            assert c["section_type"] in {
                "abstract", "introduction", "related_work", "method",
                "experiments", "results", "conclusion", "other",
            }
            assert isinstance(c["section_head"], str)


class TestSectionTagging:
    def test_long_method_chunks_tagged_method(self, tokenizer):
        # Method body large enough that chunks past the first are pure-method.
        tei = _build_tei(
            "T", "Short.",
            [("Methodology", "we use a transformer model. " * 200)],
        )
        chunks = chunk_paper_section_tagged(
            tei, tokenizer, chunk_size=128, overlap=16,
        )
        # At least the latter half of chunks should be unambiguously method.
        method_chunks = [c for c in chunks if c["section_type"] == "method"]
        assert len(method_chunks) >= len(chunks) // 2

    def test_first_chunk_typically_abstract(self, tokenizer):
        # Long abstract followed by long body — first chunk should be tagged
        # abstract (it dominates the title + start-of-doc region).
        tei = _build_tei(
            "Title",
            "We present a careful study of retrieval. " * 30,
            [("Introduction", "Background. " * 100)],
        )
        chunks = chunk_paper_section_tagged(
            tei, tokenizer, chunk_size=128, overlap=16,
        )
        assert chunks[0]["section_type"] == "abstract"

    def test_no_sections_returns_other_or_abstract(self, tokenizer):
        tei = _build_tei("T", "Just an abstract here.", [])
        chunks = chunk_paper_section_tagged(tei, tokenizer)
        assert all(c["section_type"] in {"abstract", "other"} for c in chunks)
