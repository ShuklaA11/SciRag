"""SB-W11.1: wiki BM25 search. Unit tests build WikiDocs directly (no wiki dir);
one light integration test runs over the real compiled wiki."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.wiki.search import WikiDoc, WikiHit, WikiSearchIndex

_DOCS = [
    WikiDoc("paper", "1810.04805", "BERT", "bert masked language model bidirectional transformer pretraining"),
    WikiDoc("paper", "1503.00841", "Text Classification", "regularization prior knowledge text classification robust"),
    WikiDoc("concept", "attention", "Attention", "attention mechanism weights context alignment transformer"),
]


def test_search_ranks_best_match_first():
    idx = WikiSearchIndex(_DOCS)
    hits = idx.search("masked language model bert pretraining", k=2)
    assert isinstance(hits[0], WikiHit)
    assert hits[0].ident == "1810.04805"
    assert len(hits) == 2


def test_search_matches_concept_docs_too():
    idx = WikiSearchIndex(_DOCS)
    top = idx.search("attention mechanism context", k=1)[0]
    assert (top.kind, top.ident) == ("concept", "attention")


def test_k_caps_results_and_nonpositive_returns_empty():
    idx = WikiSearchIndex(_DOCS)
    assert len(idx.search("transformer", k=2)) == 2
    assert idx.search("transformer", k=0) == []


def test_empty_index_returns_empty():
    idx = WikiSearchIndex([])
    assert len(idx) == 0
    assert idx.search("anything") == []


@pytest.mark.skipif(not Path("wiki/papers").is_dir(), reason="compiled wiki not present")
def test_integration_over_real_wiki():
    idx = WikiSearchIndex.from_wiki("wiki")
    assert len(idx) > 0
    hits = idx.search("bert language model", k=5)
    assert hits
    assert any("bert" in (h.ident + h.title).lower() for h in hits)
