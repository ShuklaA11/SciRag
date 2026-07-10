"""SB-C2: per-claim verdict aggregation. Deterministic, model-free — the
intellectual core of the idea-evaluation engine, so it gets exhaustive rules
coverage: direction, novelty gating, threshold edges, tie-break, evidence pick."""

from __future__ import annotations

import pytest

from src.ideas import CONTRADICTED, ENTAILED, NOVEL, assess_claim
from src.verification.nli_classifier import NLIPrediction


def _pred(support: float, contradict: float) -> NLIPrediction:
    nei = max(0.0, 1.0 - support - contradict)
    label = "SUPPORT" if support >= contradict else "CONTRADICT"
    return NLIPrediction(label, support, contradict, nei)


def test_strong_support_is_entailed():
    v = assess_claim("c", [("doc1", _pred(0.9, 0.05))])
    assert v.bucket == ENTAILED
    assert v.top_evidence == "doc1"
    assert v.best_support == 0.9
    assert v.n_evidence == 1


def test_strong_contradiction_is_contrarian():
    v = assess_claim("c", [("doc1", _pred(0.05, 0.88))])
    assert v.bucket == CONTRADICTED
    assert v.top_evidence == "doc1"
    assert v.best_contradict == 0.88


def test_all_nei_is_novel_with_no_top_evidence():
    v = assess_claim("c", [("d1", _pred(0.2, 0.1)), ("d2", _pred(0.3, 0.25))])
    assert v.bucket == NOVEL
    assert v.top_evidence is None
    assert v.n_evidence == 2


def test_empty_evidence_is_novel():
    v = assess_claim("c", [])
    assert v.bucket == NOVEL
    assert (v.best_support, v.best_contradict, v.n_evidence) == (0.0, 0.0, 0)
    assert v.top_evidence is None


def test_stronger_signal_wins_when_corpus_is_split():
    strong_con = [("sup", _pred(0.6, 0.1)), ("con", _pred(0.1, 0.95))]
    assert assess_claim("c", strong_con).bucket == CONTRADICTED

    strong_sup = [("sup", _pred(0.97, 0.1)), ("con", _pred(0.1, 0.6))]
    assert assess_claim("c", strong_sup).bucket == ENTAILED


def test_threshold_is_inclusive_lower_bound():
    at = assess_claim("c", [("d", _pred(0.5, 0.0))], novelty_threshold=0.5)
    assert at.bucket == ENTAILED  # support_prob == threshold clears it

    below = assess_claim("c", [("d", _pred(0.49, 0.0))], novelty_threshold=0.5)
    assert below.bucket == NOVEL


def test_support_ties_contradict_resolves_to_entailed():
    v = assess_claim("c", [("d", _pred(0.7, 0.7))])
    assert v.bucket == ENTAILED  # best_support >= best_contradict


def test_best_scores_taken_across_all_evidence_not_first():
    ev = [("weak", _pred(0.3, 0.2)), ("strong", _pred(0.85, 0.4)), ("mid", _pred(0.5, 0.6))]
    v = assess_claim("c", ev)
    assert v.best_support == 0.85
    assert v.best_contradict == 0.6
    assert v.top_evidence == "strong"  # entailed, driven by max-support doc
