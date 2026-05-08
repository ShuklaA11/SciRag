"""Tests for src/pipeline/section_chunker.py.

Heuristic head->section_type mapping is pure-string and always runs.
chunk_paper_by_section tests use the SPECTER2 tokenizer (~2 MB cached
on first run) for parity with test_chunker_embedder.py.
"""

from __future__ import annotations

import pytest

from src.pipeline.section_chunker import (
    SECTION_TYPES,
    chunk_paper_by_section,
    resolve_section_types,
    section_type_for_head,
)


class TestSectionTypeForHead:
    @pytest.mark.parametrize("head,expected", [
        # introduction
        ("Introduction", "introduction"),
        ("1. Introduction", "introduction"),
        ("1 Introduction", "introduction"),
        ("INTRODUCTION", "introduction"),
        ("Motivation", "introduction"),
        # related work
        ("Related Work", "related_work"),
        ("2. Background", "related_work"),
        ("Prior Work", "related_work"),
        ("Literature Review", "related_work"),
        # method
        ("Methodology", "method"),
        ("3 Methodology", "method"),
        ("Proposed Approach", "method"),
        ("Model Architecture", "method"),
        ("Our Algorithm", "method"),
        # experiments
        ("Experimental Setup", "experiments"),
        ("4.1 Datasets", "experiments"),
        ("Implementation Details", "experiments"),
        ("Training", "experiments"),
        # results
        ("Results", "results"),
        ("5. Results and Analysis", "results"),
        ("Ablation Study", "results"),
        ("Evaluation", "results"),
        # conclusion
        ("Conclusion", "conclusion"),
        ("Discussion", "conclusion"),
        ("Future Work", "conclusion"),
        ("Limitations", "conclusion"),
        # other
        ("Acknowledgments", "other"),
        ("[untitled]", "other"),
        ("", "other"),
        ("References", "other"),
    ])
    def test_known_heads(self, head, expected):
        assert section_type_for_head(head) == expected

    def test_specificity_order_method_beats_introduction(self):
        # "Model Overview" contains "overview" (introduction) and "model"
        # (method); method must win because it's listed earlier.
        assert section_type_for_head("Model Overview") == "method"

    def test_returns_only_canonical_types(self):
        for head in ["Random gibberish 12345", "", "[untitled]", "xyz"]:
            assert section_type_for_head(head) in SECTION_TYPES

    def test_strips_roman_numerals(self):
        assert section_type_for_head("IV. Experiments") == "experiments"

    def test_strips_nested_numbering(self):
        assert section_type_for_head("4.1.2 Datasets") == "experiments"


class TestResolveSectionTypes:
    def test_subsection_inherits_parent(self):
        # 3 = Method (matches), 3.1 = Parser (no match -> inherit method)
        sections = [
            {"head": "Methods", "n": "3", "text": "x"},
            {"head": "Parser", "n": "3.1", "text": "x"},
        ]
        assert resolve_section_types(sections) == ["method", "method"]

    def test_grandchild_inherits_through_chain(self):
        sections = [
            {"head": "Experiments", "n": "4", "text": "x"},
            {"head": "Stance Detection", "n": "4.2", "text": "x"},
            {"head": "Beam Search", "n": "4.2.1", "text": "x"},
        ]
        assert resolve_section_types(sections) == [
            "experiments", "experiments", "experiments",
        ]

    def test_own_classification_overrides_inheritance(self):
        # 3 = Method, 3.1 = Datasets (matches experiments -> stays experiments)
        sections = [
            {"head": "Methods", "n": "3", "text": "x"},
            {"head": "Datasets", "n": "3.1", "text": "x"},
        ]
        assert resolve_section_types(sections) == ["method", "experiments"]

    def test_top_level_other_stays_other(self):
        sections = [{"head": "Glossary", "n": "1", "text": "x"}]
        assert resolve_section_types(sections) == ["other"]

    def test_no_n_means_no_inheritance(self):
        # No `n`, no parent lookup possible -> stays other.
        sections = [
            {"head": "Methods", "n": "3", "text": "x"},
            {"head": "Parser", "n": "", "text": "x"},
        ]
        assert resolve_section_types(sections) == ["method", "other"]

    def test_orphan_subsection_no_ancestor_seen(self):
        # 3.1 appears with no prior 3 in the list -> can't inherit.
        sections = [{"head": "Parser", "n": "3.1", "text": "x"}]
        assert resolve_section_types(sections) == ["other"]

    def test_does_not_cross_top_level_boundaries(self):
        # 3 = Method; 4 = Experiments; 4.1 = Beam Search must inherit
        # from 4 (experiments), not from 3 (method).
        sections = [
            {"head": "Methods", "n": "3", "text": "x"},
            {"head": "Experiments", "n": "4", "text": "x"},
            {"head": "Beam Search", "n": "4.1", "text": "x"},
        ]
        assert resolve_section_types(sections) == [
            "method", "experiments", "experiments",
        ]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    from src.pipeline.embedder import MODEL_NAME
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def _build_tei(title: str, abstract: str, sections: list[tuple[str, str]]) -> str:
    """Build a minimal TEI XML for testing."""
    abs_xml = (
        f"<profileDesc><abstract><p>{abstract}</p></abstract></profileDesc>"
        if abstract
        else "<profileDesc/>"
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


class TestChunkPaperBySection:
    def test_abstract_chunk_tagged_and_includes_title(self, tokenizer):
        tei = _build_tei(
            "Cool Paper Title",
            "We propose a new approach to retrieval.",
            [("Introduction", "Background and motivation here.")],
        )
        chunks = chunk_paper_by_section(tei, tokenizer)
        abs_chunks = [c for c in chunks if c["section_type"] == "abstract"]
        assert len(abs_chunks) == 1
        assert "cool paper title" in abs_chunks[0]["text"].lower()
        assert abs_chunks[0]["section_head"] == "Abstract"

    def test_section_types_are_tagged(self, tokenizer):
        tei = _build_tei(
            "T", "Short abstract.",
            [
                ("Introduction", "Intro body text."),
                ("Related Work", "Related work body."),
                ("Methodology", "Method body."),
                ("Experiments", "Experiments body."),
                ("Results", "Results body."),
                ("Conclusion", "Conclusion body."),
            ],
        )
        chunks = chunk_paper_by_section(tei, tokenizer)
        types = [c["section_type"] for c in chunks]
        for expected in ["abstract", "introduction", "related_work",
                         "method", "experiments", "results", "conclusion"]:
            assert expected in types, f"missing {expected} in {types}"

    def test_long_section_splits_with_same_section_type(self, tokenizer):
        # Build a long method body that exceeds 512 tokens.
        long_body = ("we propose a method. " * 400)
        tei = _build_tei(
            "T", "abs.",
            [("Methodology", long_body)],
        )
        chunks = chunk_paper_by_section(tei, tokenizer, chunk_size=128, overlap=16)
        method_chunks = [c for c in chunks if c["section_type"] == "method"]
        assert len(method_chunks) > 1
        for c in method_chunks:
            assert c["section_type"] == "method"

    def test_chunk_idx_monotonic_global(self, tokenizer):
        tei = _build_tei(
            "T", "abs body.",
            [("Introduction", "intro body."), ("Methodology", "method body.")],
        )
        chunks = chunk_paper_by_section(tei, tokenizer)
        idxs = [c["chunk_idx"] for c in chunks]
        assert idxs == list(range(len(chunks)))

    def test_no_bleed_across_sections(self, tokenizer):
        tei = _build_tei(
            "T", "alpha alpha alpha.",
            [("Introduction", "beta beta beta."),
             ("Methodology", "gamma gamma gamma.")],
        )
        chunks = chunk_paper_by_section(tei, tokenizer)
        by_type = {c["section_type"]: c["text"].lower() for c in chunks}
        # Each section's body tokens should appear only in its own chunk.
        assert "alpha" in by_type["abstract"] and "beta" not in by_type["abstract"]
        assert "beta" in by_type["introduction"] and "gamma" not in by_type["introduction"]
        assert "gamma" in by_type["method"] and "beta" not in by_type["method"]

    def test_no_abstract_title_attached_to_first_section(self, tokenizer):
        tei = _build_tei(
            "Unique Title String", "",
            [("Introduction", "intro body here.")],
        )
        chunks = chunk_paper_by_section(tei, tokenizer)
        assert len(chunks) >= 1
        assert chunks[0]["section_type"] == "introduction"
        assert "unique title string" in chunks[0]["text"].lower()

    def test_empty_paper_returns_empty(self, tokenizer):
        tei = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            "<text><body/></text></TEI>"
        )
        assert chunk_paper_by_section(tei, tokenizer) == []

    def test_section_without_head_is_other(self, tokenizer):
        tei = _build_tei(
            "T", "abs body.",
            [("", "headless section content here.")],
        )
        chunks = chunk_paper_by_section(tei, tokenizer)
        non_abs = [c for c in chunks if c["section_type"] != "abstract"]
        assert len(non_abs) >= 1
        assert non_abs[0]["section_type"] == "other"
