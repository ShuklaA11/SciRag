"""Idea-evaluation engine (v3 Phase C).

Decompose a research idea into atomic claims, retrieve corpus evidence per
claim, run NLI, and bucket each claim as ENTAILED (not novel) / CONTRADICTED
(contrarian) / NOVEL (untested). Retrieval + NLI compute the verdict; any LLM
is a narrator only. Reports are always per-claim — never a single novelty
scalar, which would be gameable.
"""

from src.ideas.claims import ClaimDecomposer
from src.ideas.engine import Evidence, IdeaEvaluator, IdeaReport, Provenance
from src.ideas.verdict import (
    BUCKETS,
    CONTRADICTED,
    DEFAULT_NOVELTY_THRESHOLD,
    ENTAILED,
    NOVEL,
    ClaimVerdict,
    assess_claim,
)

__all__ = [
    "BUCKETS",
    "CONTRADICTED",
    "DEFAULT_NOVELTY_THRESHOLD",
    "ENTAILED",
    "NOVEL",
    "ClaimDecomposer",
    "ClaimVerdict",
    "Evidence",
    "IdeaEvaluator",
    "IdeaReport",
    "Provenance",
    "assess_claim",
]
