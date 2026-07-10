"""Idea-evaluation orchestrator (v3 Phase C, SB-C3).

Wires the Phase-C pieces into ``idea → IdeaReport``:

    decompose idea → claims          (ClaimDecomposer, SB-C1; LLM narrator)
    per claim: retrieve evidence      (retriever protocol; BM25/SPECTER2 at call site)
    per claim: NLI over evidence      (NLIClassifier, fine-tuned DeBERTa)
    per claim: bucket the verdict      (assess_claim, SB-C2; the actual decision)

Dependencies are injected as structural protocols, so the engine is decoupled
from the SciFact corpus shape and testable with fakes (no models). The real
``BM25EvidenceRetriever`` (which returns ``doc_id``/``score`` only) is adapted
into ``Evidence`` — carrying the text NLI needs — at the call site.

The report is always per-claim. There is deliberately no single novelty scalar
(plan-v3 Phase C: a scalar is gameable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.ideas.verdict import DEFAULT_NOVELTY_THRESHOLD, ClaimVerdict, assess_claim
from src.verification.nli_classifier import NLIPrediction

DEFAULT_K = 5


@dataclass(frozen=True)
class Evidence:
    """One retrieved evidence unit. ``ref`` is an opaque handle back to the
    source (e.g. a doc id); ``text`` is what NLI reads."""

    ref: object
    text: str
    score: float = 0.0


@dataclass(frozen=True)
class Provenance:
    """What produced a report, for auditability. Records only what the engine
    controls; git commit is stamped at persistence time (SB-C4)."""

    model: str
    k: int
    novelty_threshold: float
    n_claims: int


@dataclass(frozen=True)
class IdeaReport:
    idea: str
    verdicts: tuple[ClaimVerdict, ...]
    provenance: Provenance


class _DecomposerLike(Protocol):
    def decompose(self, idea: str) -> list[str]: ...


class _RetrieverLike(Protocol):
    def retrieve(self, query: str, k: int) -> list[Evidence]: ...


class _NLILike(Protocol):
    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]: ...


class IdeaEvaluator:
    def __init__(
        self,
        decomposer: _DecomposerLike,
        retriever: _RetrieverLike,
        nli: _NLILike,
        *,
        k: int = DEFAULT_K,
        novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD,
        model: str = "",
    ) -> None:
        self.decomposer = decomposer
        self.retriever = retriever
        self.nli = nli
        self.k = k
        self.novelty_threshold = novelty_threshold
        self.model = model

    def evaluate(self, idea: str) -> IdeaReport:
        """Decompose ``idea`` into claims, then evaluate each."""
        return self.evaluate_claims(self.decomposer.decompose(idea), idea=idea)

    def evaluate_claims(self, claims: list[str], *, idea: str = "") -> IdeaReport:
        """Evaluate pre-split claims — the LLM-free path (temporal eval, SciFact)."""
        claims = list(claims)
        verdicts = tuple(self._assess(c) for c in claims)
        provenance = Provenance(
            model=self.model,
            k=self.k,
            novelty_threshold=self.novelty_threshold,
            n_claims=len(claims),
        )
        return IdeaReport(idea=idea, verdicts=verdicts, provenance=provenance)

    def _assess(self, claim: str) -> ClaimVerdict:
        evidences = self.retriever.retrieve(claim, self.k)
        preds = self.nli.predict_batch([(claim, e.text) for e in evidences])
        paired: list[tuple[object, NLIPrediction]] = list(zip(evidences, preds))
        return assess_claim(claim, paired, novelty_threshold=self.novelty_threshold)
