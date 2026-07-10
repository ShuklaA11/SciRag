"""Per-claim verdict aggregation (v3 Phase C, SB-C2).

The model-free core of the idea-evaluation engine. Given one claim and the NLI
predictions for each retrieved evidence doc, decide which of three buckets the
claim falls in:

  ENTAILED     — the corpus already supports it (not novel)
  CONTRADICTED — the corpus refutes it (contrarian)
  NOVEL        — nothing in the corpus strongly engages it (untested)

Rule: take the strongest SUPPORT signal and the strongest CONTRADICT signal
across all evidence. If neither clears ``novelty_threshold`` the claim is NOVEL;
otherwise the stronger directional signal wins. This is symmetric,
deterministic, and reports both component scores — deliberately not collapsed
to a single novelty scalar, which would be gameable (plan-v3 Phase C).

Takes ``NLIPrediction`` objects as input, so it has no model dependency and is
fully unit-testable. The engine (SB-C3) supplies the predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.verification.nli_classifier import NLIPrediction

ENTAILED = "ENTAILED"
CONTRADICTED = "CONTRADICTED"
NOVEL = "NOVEL"
BUCKETS = (ENTAILED, CONTRADICTED, NOVEL)

# Directional-probability floor below which the corpus is treated as not
# engaging the claim (→ NOVEL). Mirrors the NLI classifier's own NEI gating
# (max directional prob < 0.5 → NEI) so a claim whose every evidence read as
# NEI lands in NOVEL rather than being forced into a direction.
DEFAULT_NOVELTY_THRESHOLD = 0.5


@dataclass(frozen=True)
class ClaimVerdict:
    """Auditable per-claim result. ``top_evidence`` is the caller-supplied
    evidence reference that drove the decision (``None`` when no evidence)."""

    claim: str
    bucket: str
    best_support: float
    best_contradict: float
    n_evidence: int
    top_evidence: Any = None


def assess_claim(
    claim: str,
    evidence: list[tuple[Any, NLIPrediction]],
    *,
    novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD,
) -> ClaimVerdict:
    """Bucket ``claim`` from its per-evidence NLI predictions.

    ``evidence`` is a list of ``(evidence_ref, prediction)`` pairs; the ref is
    opaque (doc id, ``RetrievedDoc``, text) and passed through to
    ``top_evidence`` untouched.
    """
    if not evidence:
        return ClaimVerdict(claim, NOVEL, 0.0, 0.0, 0, None)

    sup_ref, sup_pred = max(evidence, key=lambda e: e[1].support_prob)
    con_ref, con_pred = max(evidence, key=lambda e: e[1].contradict_prob)
    best_support = sup_pred.support_prob
    best_contradict = con_pred.contradict_prob

    if max(best_support, best_contradict) < novelty_threshold:
        bucket, top_ref = NOVEL, None
    elif best_support >= best_contradict:
        bucket, top_ref = ENTAILED, sup_ref
    else:
        bucket, top_ref = CONTRADICTED, con_ref

    return ClaimVerdict(
        claim=claim,
        bucket=bucket,
        best_support=best_support,
        best_contradict=best_contradict,
        n_evidence=len(evidence),
        top_evidence=top_ref,
    )
