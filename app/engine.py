"""Composition root for the idea-evaluation engine (app edge, not core).

Assembles the real IdeaEvaluator — FlatIndex(bge) + zero-shot DeBERTa NLI +
Ollama decomposer — once per session via ``st.cache_resource``. Unlike the
thread-affine sqlite connection in HubStore, model objects are cache-safe
(read-only inference), so caching them here is correct and avoids reloading
~1GB of weights on every rerun.

``SCIRAG_FAKE_ENGINE=1`` returns a deterministic, model-free fake — mirroring the
``SCIRAG_HUB_DB`` / ``SCIRAG_DOMAIN`` env idiom so the heavy view stays
smoke-testable. Wiring concrete components lives here, not in ``src.ideas``: the
engine stays pure and dependency-injected; the app is where it gets bolted on.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.ideas import (
    CONTRADICTED,
    ENTAILED,
    NOVEL,
    ClaimDecomposer,
    ClaimVerdict,
    Evidence,
    IdeaEvaluator,
    IdeaReport,
    Provenance,
)

INDEX_DIR = Path("data/index/flat_bge_tagged")


def build_evaluator():
    """The evaluator the view calls: fake (cheap, per-run) or real (cached)."""
    if os.getenv("SCIRAG_FAKE_ENGINE") == "1":
        return _FakeEvaluator()
    return _build_real_evaluator()


@st.cache_resource(show_spinner="Loading retrieval index + NLI model …")
def _build_real_evaluator() -> IdeaEvaluator:
    from src.llm.client import OllamaProvider
    from src.retrieval.flat_index import FlatIndex
    from src.verification.nli_classifier import DEFAULT_MODEL, NLIClassifier

    index = FlatIndex(INDEX_DIR, embedder_name="bge")
    return IdeaEvaluator(
        ClaimDecomposer(OllamaProvider()),
        _FlatRetriever(index),
        NLIClassifier(),
        k=5,
        model=DEFAULT_MODEL,
    )


class _FlatRetriever:
    """FlatIndex → engine Evidence (the text NLI reads)."""

    def __init__(self, index) -> None:
        self.index = index

    def retrieve(self, query: str, k: int) -> list[Evidence]:
        return [
            Evidence(ref=c["arxiv_id"], text=c["text"], score=c["score"])
            for c in self.index.search(query, k=k)
        ]


class _FakeEvaluator:
    """Deterministic, model-free — one verdict per bucket, for smoke tests."""

    def evaluate(self, idea: str) -> IdeaReport:
        verdicts = (
            ClaimVerdict("A known claim.", ENTAILED, 0.91, 0.03, 3, Evidence("d1", "ev", 1.0)),
            ClaimVerdict("A refuted claim.", CONTRADICTED, 0.04, 0.88, 2, Evidence("d2", "ev", 1.0)),
            ClaimVerdict("An untested claim.", NOVEL, 0.2, 0.1, 4, None),
        )
        return IdeaReport(idea=idea, verdicts=verdicts, provenance=Provenance("fake", 5, 0.5, 3))
