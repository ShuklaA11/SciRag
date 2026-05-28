"""Tests for src/wiki/indices.py — light, no LLM, no model loads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.wiki.indices import (
    _extract_year,
    _parse_depends_on,
    build_all,
    load_concept,
    render_glossary,
    render_graph,
    render_index,
    render_questions,
    render_timeline,
)
from src.wiki.concept_extractor import Summary


PAPER_FIXTURE = """---
arxiv_id: 1601.06738
title: "Example QASPER Paper"
generated_by: llama3.1:8b
generated_at: 2026-05-27T00:00:00+00:00
status: ok
---

# Example QASPER Paper

## TL;DR
A short summary.

## Problem
Some problem.

## Method
Some method.

## Results
Some results.

## Limitations
Unknown
"""

CANONICAL_FIXTURE = """---
arxiv_id: bert
title: "BERT: Bidirectional Transformers"
generated_by: llama3.1:8b
generated_at: 2026-05-27T00:00:00+00:00
status: ok
---

# BERT: Bidirectional Transformers

## TL;DR
A.

## Problem
B.

## Method
C.

## Results
D.

## Limitations
Unknown
"""

CONCEPT_FIXTURE = """---
concept: "Masked Language Model"
generated_by: llama3.1:8b
generated_at: 2026-05-27T00:00:00+00:00
status: ok
depends_on: [bert, 1601.06738]
---

# Masked Language Model

## Definition
An objective that masks tokens and predicts them from context.

## Origin
Introduced by BERT.

## Key Papers
- [[bert]]
- [[1601.06738]]

## Variants
Span MLM, whole-word masking.

## Open Questions
Optimal mask ratio remains contested.
"""


@pytest.fixture
def wiki(tmp_path: Path) -> dict[str, Path]:
    papers = tmp_path / "papers"
    concepts = tmp_path / "concepts"
    indices = tmp_path / "indices"
    papers.mkdir()
    concepts.mkdir()
    (papers / "1601.06738.md").write_text(PAPER_FIXTURE)
    (papers / "bert.md").write_text(CANONICAL_FIXTURE)
    (concepts / "masked_language_model.md").write_text(CONCEPT_FIXTURE)
    return {"papers": papers, "concepts": concepts, "indices": indices}


# ------------------------------------------------------------------- helpers


def test_extract_year_parses_yymm() -> None:
    assert _extract_year("1601.06738") == 2016
    assert _extract_year("2003.12345") == 2020


def test_extract_year_returns_none_for_non_yymm() -> None:
    assert _extract_year("bert") is None
    assert _extract_year("attention_is_all_you_need") is None


def test_parse_depends_on_handles_brackets_and_commas() -> None:
    assert _parse_depends_on("[bert, 1601.06738]") == ["bert", "1601.06738"]
    assert _parse_depends_on("[]") == []
    assert _parse_depends_on("") == []


# ------------------------------------------------------------------- concepts


def test_load_concept_parses_frontmatter_and_sections(wiki: dict[str, Path]) -> None:
    c = load_concept(wiki["concepts"] / "masked_language_model.md")
    assert c is not None
    assert c.slug == "masked_language_model"
    assert c.name == "Masked Language Model"
    assert "masks tokens" in c.definition
    assert c.depends_on == ["bert", "1601.06738"]
    assert "Optimal mask ratio" in c.open_questions


# ------------------------------------------------------------------- renders


def test_render_index_lists_papers_and_concepts(wiki: dict[str, Path]) -> None:
    counts = build_all(wiki["papers"], wiki["concepts"], wiki["indices"])
    index = (wiki["indices"] / "INDEX.md").read_text()
    assert "## Papers" in index
    assert "## Concepts" in index
    assert "[[bert]]" in index
    assert "[[1601.06738]]" in index
    assert "[[masked_language_model]]" in index
    assert counts["n_papers"] == 2
    assert counts["n_concepts"] == 1


def test_render_glossary_includes_definition(wiki: dict[str, Path]) -> None:
    build_all(wiki["papers"], wiki["concepts"], wiki["indices"])
    g = (wiki["indices"] / "GLOSSARY.md").read_text()
    assert "Masked Language Model" in g
    assert "masks tokens" in g


def test_render_timeline_groups_by_year(wiki: dict[str, Path]) -> None:
    build_all(wiki["papers"], wiki["concepts"], wiki["indices"])
    t = (wiki["indices"] / "TIMELINE.md").read_text()
    assert "## 2016" in t
    assert "## Undated" in t  # bert has no YYMM
    assert "[[1601.06738]]" in t
    assert "[[bert]]" in t


def test_render_questions_aggregates_concept_open_questions(wiki: dict[str, Path]) -> None:
    build_all(wiki["papers"], wiki["concepts"], wiki["indices"])
    q = (wiki["indices"] / "QUESTIONS.md").read_text()
    assert "Optimal mask ratio" in q
    assert "Masked Language Model" in q


def test_render_questions_handles_no_concepts(tmp_path: Path) -> None:
    papers = tmp_path / "papers"; papers.mkdir()
    concepts = tmp_path / "concepts"; concepts.mkdir()
    out = render_questions([])
    assert "no open questions" in out.lower()


def test_render_graph_emits_paper_concept_edges(wiki: dict[str, Path]) -> None:
    build_all(wiki["papers"], wiki["concepts"], wiki["indices"])
    g = json.loads((wiki["indices"] / "GRAPH.json").read_text())
    assert {n["id"] for n in g["nodes"]} == {"bert", "1601.06738", "masked_language_model"}
    edge_pairs = {(e["source"], e["target"]) for e in g["edges"]}
    assert ("masked_language_model", "bert") in edge_pairs
    assert ("masked_language_model", "1601.06738") in edge_pairs


def test_render_graph_drops_edges_to_missing_papers(tmp_path: Path) -> None:
    papers = tmp_path / "papers"; papers.mkdir()
    concepts = tmp_path / "concepts"; concepts.mkdir()
    # concept references a paper that doesn't exist
    (concepts / "ghost.md").write_text(
        '---\nconcept: "Ghost"\ngenerated_by: x\ngenerated_at: x\n'
        'status: ok\ndepends_on: [does_not_exist]\n---\n\n## Definition\nX\n'
    )
    out = tmp_path / "out"
    build_all(papers, concepts, out)
    g = json.loads((out / "GRAPH.json").read_text())
    assert g["edges"] == []


def test_build_all_is_idempotent(wiki: dict[str, Path]) -> None:
    counts1 = build_all(wiki["papers"], wiki["concepts"], wiki["indices"])
    counts2 = build_all(wiki["papers"], wiki["concepts"], wiki["indices"])
    assert counts1 == counts2


def test_build_all_handles_empty_concepts_dir(tmp_path: Path) -> None:
    papers = tmp_path / "papers"; papers.mkdir()
    (papers / "bert.md").write_text(CANONICAL_FIXTURE)
    out = tmp_path / "out"
    counts = build_all(papers, tmp_path / "no_concepts_dir", out)
    assert counts["n_concepts"] == 0
    assert (out / "INDEX.md").exists()
