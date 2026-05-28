"""Tests for src/wiki/concept_extractor.py.

All tests are light (no model loads, no LLM calls). Fixtures are tiny
inline markdown strings written to tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.wiki.concept_extractor import (
    ConceptEvidence,
    Summary,
    _normalize,
    _parse_sections,
    extract_candidates,
    load_summaries_dir,
    load_summary,
    rank_concepts,
)


FIXTURE_BERT = """---
arxiv_id: bert
title: "BERT: Pre-training of Deep Bidirectional Transformers"
generated_by: llama3.1:8b
generated_at: 2026-04-28T22:51:50+00:00
status: ok
---

# BERT: Pre-training of Deep Bidirectional Transformers

## TL;DR
BERT is a pre-trained language model achieving state-of-the-art on GLUE and SQuAD.

## Problem
Current models are unidirectional.

## Method
The authors propose BERT, which uses a Masked Language Model objective and a
Next Sentence Prediction task to enable deep bidirectional Transformer representations.

## Results
BERT obtains 80.5% on GLUE and 93.2 F1 on SQuAD v1.1.

## Limitations
Unknown
"""

FIXTURE_GPT2 = """---
arxiv_id: gpt2
title: "Language Models are Unsupervised Multitask Learners"
generated_by: llama3.1:8b
generated_at: 2026-04-28T22:51:50+00:00
status: ok
---

# Language Models are Unsupervised Multitask Learners

## TL;DR
GPT-2 demonstrates that Language Models can perform many NLP tasks zero-shot.

## Problem
Supervised learning requires large labeled datasets.

## Method
The authors train a large Transformer on WebText, then evaluate zero-shot on
multiple benchmarks including SQuAD and CoQA. The Language Model approach
generalizes across tasks.

## Results
GPT-2 achieves SOTA on 7 of 8 zero-shot benchmarks.

## Limitations
Unknown
"""

FIXTURE_ELMO = """---
arxiv_id: elmo
title: "Deep contextualized word representations"
generated_by: llama3.1:8b
generated_at: 2026-04-28T22:51:50+00:00
status: ok
---

# Deep contextualized word representations

## TL;DR
ELMo learns Language Model representations from biLM and improves downstream tasks.

## Problem
Static word embeddings ignore context.

## Method
The authors train a biLM on a large corpus and use a learned linear
combination of internal states as contextualized embeddings. The Language Model
provides deep word representations.

## Results
20% relative error reductions across six NLP tasks.

## Limitations
Unknown
"""


@pytest.fixture
def papers_dir(tmp_path: Path) -> Path:
    d = tmp_path / "papers"
    d.mkdir()
    (d / "bert.md").write_text(FIXTURE_BERT)
    (d / "gpt2.md").write_text(FIXTURE_GPT2)
    (d / "elmo.md").write_text(FIXTURE_ELMO)
    (d / "REVIEW_NOTES.md").write_text("# review notes — should be skipped")
    return d


# --------------------------------------------------------------------- parsing


def test_load_summary_parses_frontmatter(papers_dir: Path) -> None:
    s = load_summary(papers_dir / "bert.md")
    assert s is not None
    assert s.arxiv_id == "bert"
    assert "BERT" in s.title


def test_load_summary_parses_sections(papers_dir: Path) -> None:
    s = load_summary(papers_dir / "bert.md")
    assert s is not None
    assert "Masked Language Model" in s.sections["Method"]
    assert "GLUE" in s.sections["TL;DR"]
    assert s.sections["Limitations"] == "Unknown"


def test_load_summary_returns_none_for_missing_frontmatter(tmp_path: Path) -> None:
    p = tmp_path / "broken.md"
    p.write_text("# No frontmatter here\n\nJust a body.")
    assert load_summary(p) is None


def test_parse_sections_handles_empty_body() -> None:
    assert _parse_sections("") == {}


# ----------------------------------------------------------------- extraction


def test_extract_candidates_finds_acronyms() -> None:
    text = "BERT obtains 80.5% on GLUE and SQuAD v1.1."
    cands = extract_candidates(text)
    assert "BERT" in cands
    assert "GLUE" in cands
    assert "SQuAD" in cands


def test_extract_candidates_finds_capitalized_phrases() -> None:
    text = "The authors propose Masked Language Model and Next Sentence Prediction."
    cands = extract_candidates(text)
    assert "Masked Language Model" in cands
    assert "Next Sentence Prediction" in cands


def test_extract_candidates_drops_acronym_stopwords() -> None:
    text = "Convert PDF to JSON via API endpoint."
    cands = extract_candidates(text)
    assert "PDF" not in cands
    assert "JSON" not in cands
    assert "API" not in cands


def test_extract_candidates_empty_text() -> None:
    assert extract_candidates("") == []


# ----------------------------------------------------------------- normalize


def test_normalize_lowercases() -> None:
    assert _normalize("Language Model") == "language model"


def test_normalize_strips_plural_on_long_last_word() -> None:
    assert _normalize("Language Models") == "language model"
    assert _normalize("Transformers") == "transformer"


def test_normalize_preserves_short_or_ss_endings() -> None:
    assert _normalize("Loss") == "loss"
    assert _normalize("MLPs") == "mlps"  # short word, plural not stripped


# ----------------------------------------------------------------- ranking


def test_rank_concepts_aggregates_across_papers(papers_dir: Path) -> None:
    summaries = load_summaries_dir(papers_dir)
    assert len(summaries) == 3
    ranked = rank_concepts(summaries, top_n=20, min_paper_count=2)
    concepts = {c for c, _, _ in ranked}
    # "Language Model" appears in all 3
    assert "language model" in concepts


def test_rank_concepts_filters_singletons(papers_dir: Path) -> None:
    summaries = load_summaries_dir(papers_dir)
    ranked = rank_concepts(summaries, top_n=50, min_paper_count=2)
    # "Next Sentence Prediction" only in bert → filtered
    concepts = {c for c, _, _ in ranked}
    assert "next sentence prediction" not in concepts


def test_rank_concepts_returns_evidence_with_snippets(papers_dir: Path) -> None:
    summaries = load_summaries_dir(papers_dir)
    ranked = rank_concepts(summaries, top_n=20, min_paper_count=2)
    by_name = {c: (count, evs) for c, count, evs in ranked}
    # gpt2 + elmo each say "Language Model" (cap) in TL;DR or Method;
    # bert says "Masked Language Model" which normalizes differently.
    assert "language model" in by_name
    count, evs = by_name["language model"]
    assert count == 2
    assert all(isinstance(e, ConceptEvidence) for e in evs)
    assert all(e.snippet for e in evs)
    paper_ids = {e.arxiv_id for e in evs}
    assert paper_ids == {"gpt2", "elmo"}


def test_rank_concepts_respects_top_n(papers_dir: Path) -> None:
    summaries = load_summaries_dir(papers_dir)
    ranked = rank_concepts(summaries, top_n=1, min_paper_count=2)
    assert len(ranked) <= 1


def test_rank_concepts_sorted_by_paper_count_desc(papers_dir: Path) -> None:
    summaries = load_summaries_dir(papers_dir)
    ranked = rank_concepts(summaries, top_n=20, min_paper_count=2)
    counts = [c for _, c, _ in ranked]
    assert counts == sorted(counts, reverse=True)


def test_load_summaries_dir_skips_review_notes(papers_dir: Path) -> None:
    summaries = load_summaries_dir(papers_dir)
    arxiv_ids = {s.arxiv_id for s in summaries}
    assert "REVIEW_NOTES" not in arxiv_ids
    assert len(summaries) == 3


def test_load_summaries_dir_empty(tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    assert load_summaries_dir(d) == []


def test_summary_signal_text_combines_tldr_and_method() -> None:
    s = Summary(
        arxiv_id="x",
        title="X",
        sections={"TL;DR": "A.", "Method": "B.", "Results": "C."},
    )
    sig = s.signal_text()
    assert "A." in sig
    assert "B." in sig
    assert "C." not in sig  # Results is not a signal section
