"""Composition root for the research crew (app edge, not core).

Assembles the real ResearchCrew — Ollama supervisor + CrewTools over
FlatIndex(bge) retrieval, an IdeaEvaluator using the V2b DeBERTa-LARGE NLI
(the +8pt verifier), and a BrainstormLoop — once per session via
``st.cache_resource`` (model objects are read-only and cache-safe).

``SCIRAG_FAKE_ENGINE=1`` returns a deterministic model-free fake, mirroring the
engine's env idiom so the Streamlit view stays smoke-testable without models.
Wiring concrete components lives here; ``src.crew`` stays pure and DI'd.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

INDEX_DIR = Path("data/index/flat_bge_tagged")
# V2b win: DeBERTa-large zero-shot (dev acc 0.774) is the crew's verifier.
LARGE_NLI = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NEI_THRESHOLD = 0.55


def build_crew():
    """The crew the view calls: fake (cheap, per-run) or real (cached)."""
    if os.getenv("SCIRAG_FAKE_ENGINE") == "1":
        return _FakeCrew()
    return _build_real_crew()


@st.cache_resource(show_spinner="Loading crew (index + large NLI + Ollama) …")
def _build_real_crew():
    from src.brainstorm.directions import DirectionProposer
    from src.brainstorm.frontier import Frontier
    from src.brainstorm.loop import BrainstormLoop
    from src.crew import CrewTools
    from src.crew.crew import ResearchCrew
    from src.ideas import ClaimDecomposer, IdeaEvaluator
    from src.llm.client import OllamaProvider
    from src.retrieval.flat_index import FlatIndex
    from src.verification.nli_classifier import NLIClassifier

    llm = OllamaProvider()
    index = FlatIndex(INDEX_DIR, embedder_name="bge")
    nli = NLIClassifier(model_name=LARGE_NLI, nei_threshold=NEI_THRESHOLD)
    evaluator = IdeaEvaluator(
        ClaimDecomposer(llm), _FlatRetriever(index), nli, k=5, model=LARGE_NLI
    )
    proposer = DirectionProposer(llm)
    brainstorm = BrainstormLoop(evaluator, proposer, lambda: Frontier(index.embedder))
    tools = CrewTools(index, evaluator, brainstorm)
    return ResearchCrew(llm, tools)


class _FlatRetriever:
    """FlatIndex → engine Evidence (the text the NLI reads)."""

    def __init__(self, index) -> None:
        self.index = index

    def retrieve(self, query: str, k: int):
        from src.ideas import Evidence

        return [
            Evidence(ref=c["arxiv_id"], text=c["text"], score=c["score"])
            for c in self.index.search(query, k=k)
        ]


class _FakeCrew:
    """Deterministic, model-free crew for smoke tests."""

    def run(self, query: str):
        from src.crew.crew import CrewResult
        from src.crew.tools import ToolResult

        findings = (
            ToolResult("search_corpus", "Retrieved 2 chunk(s): [1901.001/method] …", {}),
            ToolResult("verify_idea", "Evaluated 2 claim(s) — 1 ENTAILED, 1 NOVEL.", {}),
        )
        return CrewResult(query, f"Fake crew answer for: {query}", findings)
