"""Tests for the BM25 SciFact evidence retriever.

No external models — rank_bm25 is a pure-Python dep and loads instantly,
so these tests run against the real BM25 implementation rather than a
fake. Corpus is small synthetic data with hand-picked term overlap so
the expected ordering is deterministic.
"""

from __future__ import annotations

import pytest

from src.verification.evidence_retriever import (
    BM25EvidenceRetriever,
    RetrievedDoc,
    tokenize,
)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello World, FooBar!") == ["hello", "world", "foobar"]


def test_tokenize_keeps_digits():
    assert tokenize("CD4+ T cells in 2020") == ["cd4", "t", "cells", "in", "2020"]


def test_tokenize_empty():
    assert tokenize("") == []


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_corpus():
    return {
        1: {"title": "Diabetes overview", "abstract": ["Insulin resistance in type 2 diabetes."]},
        2: {"title": "Cancer immunotherapy", "abstract": ["T cells attack tumors."]},
        3: {"title": "Diabetic neuropathy", "abstract": ["Nerve damage from diabetes."]},
        4: {"title": "Unrelated", "abstract": ["The cat sat on the mat."]},
    }


def test_retrieve_returns_relevant_doc_first(tiny_corpus):
    retriever = BM25EvidenceRetriever(tiny_corpus)
    hits = retriever.retrieve("diabetes insulin", k=2)
    assert len(hits) == 2
    assert hits[0].doc_id == 1  # title + abstract both match
    assert hits[0].score > hits[1].score


def test_retrieve_unrelated_query_has_low_score(tiny_corpus):
    retriever = BM25EvidenceRetriever(tiny_corpus)
    top = retriever.retrieve("quantum gravity", k=1)
    assert top[0].score == 0.0  # no overlapping tokens


def test_retrieve_k_zero_returns_empty(tiny_corpus):
    retriever = BM25EvidenceRetriever(tiny_corpus)
    assert retriever.retrieve("diabetes", k=0) == []


def test_retrieve_k_greater_than_corpus(tiny_corpus):
    """Requesting more docs than exist should clamp without crashing."""
    retriever = BM25EvidenceRetriever(tiny_corpus)
    hits = retriever.retrieve("diabetes", k=99)
    assert len(hits) == 4
    assert {h.doc_id for h in hits} == {1, 2, 3, 4}


def test_retrieve_ordering_is_score_descending(tiny_corpus):
    retriever = BM25EvidenceRetriever(tiny_corpus)
    hits = retriever.retrieve("diabetes nerve damage", k=4)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_many(tiny_corpus):
    retriever = BM25EvidenceRetriever(tiny_corpus)
    out = retriever.retrieve_many(["diabetes", "cat mat"], k=1)
    assert out[0][0].doc_id == 1
    assert out[1][0].doc_id == 4


def test_doc_ids_stable_sort(tiny_corpus):
    """Doc ordering inside the BM25 index is sorted by doc_id, so two
    builds over the same corpus return identical scores for the same
    query (no insertion-order non-determinism)."""
    r1 = BM25EvidenceRetriever(tiny_corpus)
    r2 = BM25EvidenceRetriever(dict(reversed(list(tiny_corpus.items()))))
    h1 = r1.retrieve("diabetes nerve", k=4)
    h2 = r2.retrieve("diabetes nerve", k=4)
    assert [h.doc_id for h in h1] == [h.doc_id for h in h2]
    for a, b in zip(h1, h2):
        assert a.score == pytest.approx(b.score)


def test_retrieve_handles_doc_with_no_abstract():
    """Title-only docs should still be retrievable. BM25 IDF can be
    negative on degenerate single-doc corpora, so we only assert
    identity, not score sign."""
    corpus = {
        1: {"title": "title-only doc about diabetes", "abstract": []},
        2: {"title": "unrelated", "abstract": ["nothing here"]},
    }
    retriever = BM25EvidenceRetriever(corpus)
    hits = retriever.retrieve("diabetes", k=1)
    assert hits[0].doc_id == 1


def test_retrieved_doc_is_frozen():
    rd = RetrievedDoc(doc_id=1, score=0.5)
    with pytest.raises(Exception):
        rd.doc_id = 2  # type: ignore[misc]
