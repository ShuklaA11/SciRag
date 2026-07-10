"""SB-C3: idea-evaluation orchestrator. All deps faked (no models) — the guard
is that idea → per-claim IdeaReport wires decompose → retrieve → NLI → bucket
correctly, carries provenance, and never collapses to a scalar."""

from __future__ import annotations

import pytest

from src.ideas import (
    CONTRADICTED,
    ENTAILED,
    NOVEL,
    Evidence,
    IdeaEvaluator,
    IdeaReport,
)
from src.verification.nli_classifier import NLIPrediction


class FakeDecomposer:
    def __init__(self, claims: list[str]) -> None:
        self.claims = claims

    def decompose(self, idea: str) -> list[str]:
        return list(self.claims)


class FakeRetriever:
    """Returns canned evidence per query; records the k it was asked for."""

    def __init__(self, by_query: dict[str, list[Evidence]]) -> None:
        self.by_query = by_query
        self.last_k: int | None = None

    def retrieve(self, query: str, k: int) -> list[Evidence]:
        self.last_k = k
        return self.by_query.get(query, [])


class FakeNLI:
    """Maps evidence text → prediction; identity on the claim (premise-based)."""

    def __init__(self, by_evidence: dict[str, NLIPrediction]) -> None:
        self.by_evidence = by_evidence

    def predict_batch(self, pairs):
        return [self.by_evidence[evidence] for _claim, evidence in pairs]


def _sup() -> NLIPrediction:
    return NLIPrediction("SUPPORT", 0.9, 0.05, 0.05)


def _con() -> NLIPrediction:
    return NLIPrediction("CONTRADICT", 0.05, 0.92, 0.03)


def test_end_to_end_buckets_each_claim():
    ev_sup = Evidence(ref="d1", text="supports claim one", score=3.0)
    retriever = FakeRetriever({"claim one": [ev_sup]})  # "claim two" → no evidence
    nli = FakeNLI({"supports claim one": _sup()})
    evaluator = IdeaEvaluator(
        FakeDecomposer(["claim one", "claim two"]), retriever, nli, k=5, model="deberta-x"
    )

    report = evaluator.evaluate("my idea")

    assert isinstance(report, IdeaReport)
    assert report.idea == "my idea"
    assert [v.bucket for v in report.verdicts] == [ENTAILED, NOVEL]
    assert report.verdicts[0].top_evidence is ev_sup  # auditable, not a scalar
    assert report.verdicts[1].n_evidence == 0
    assert retriever.last_k == 5


def test_provenance_records_engine_config():
    evaluator = IdeaEvaluator(
        FakeDecomposer(["a", "b", "c"]),
        FakeRetriever({}),
        FakeNLI({}),
        k=7,
        novelty_threshold=0.6,
        model="deberta-x",
    )
    prov = evaluator.evaluate("idea").provenance
    assert (prov.model, prov.k, prov.novelty_threshold, prov.n_claims) == ("deberta-x", 7, 0.6, 3)


def test_contradiction_bucket():
    ev = Evidence(ref="d9", text="refutes it", score=2.0)
    evaluator = IdeaEvaluator(
        FakeDecomposer(["the claim"]),
        FakeRetriever({"the claim": [ev]}),
        FakeNLI({"refutes it": _con()}),
    )
    report = evaluator.evaluate("idea")
    assert report.verdicts[0].bucket == CONTRADICTED
    assert report.verdicts[0].top_evidence is ev


def test_evaluate_claims_is_llm_free_path():
    # No decomposer call: pre-split claims go straight in (temporal eval / SciFact).
    evaluator = IdeaEvaluator(
        FakeDecomposer(["should-not-be-used"]),
        FakeRetriever({"pre-split": [Evidence("d", "supports claim one", 1.0)]}),
        FakeNLI({"supports claim one": _sup()}),
    )
    report = evaluator.evaluate_claims(["pre-split"], idea="tagged")
    assert report.idea == "tagged"
    assert [v.claim for v in report.verdicts] == ["pre-split"]
    assert report.verdicts[0].bucket == ENTAILED


def test_no_claims_yields_empty_report():
    evaluator = IdeaEvaluator(FakeDecomposer([]), FakeRetriever({}), FakeNLI({}))
    report = evaluator.evaluate("   ")
    assert report.verdicts == ()
    assert report.provenance.n_claims == 0
