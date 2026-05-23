"""Tests for the NLI classifier.

Real HF model load is gated behind ``SCIRAG_RUN_HEAVY=1`` (mirrors the
pattern in test_cross_encoder_reranker.py). Light tests use a fake
model + tokenizer to verify label mapping, NEI thresholding, and batch
plumbing without downloading 750MB.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from src.verification.nli_classifier import (
    CONTRADICT,
    DEFAULT_MODEL,
    NEI,
    SUPPORT,
    NLIClassifier,
    NLIPrediction,
    _build_label_map,
)


# ---------------------------------------------------------------------------
# Label-mapping unit tests
# ---------------------------------------------------------------------------


def test_label_map_upper_case_mnli():
    assert _build_label_map(
        {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
    ) == {0: SUPPORT, 1: NEI, 2: CONTRADICT}


def test_label_map_lower_case():
    assert _build_label_map(
        {0: "entailment", 1: "neutral", 2: "contradiction"}
    ) == {0: SUPPORT, 1: NEI, 2: CONTRADICT}


def test_label_map_index_independent():
    """DeBERTa-mnli-fever-anli ships CONTRADICTION at index 0."""
    mapping = _build_label_map({0: "contradiction", 1: "neutral", 2: "entailment"})
    assert mapping == {0: CONTRADICT, 1: NEI, 2: SUPPORT}


def test_label_map_unknown_label_raises():
    with pytest.raises(ValueError, match="Unrecognised MNLI label"):
        _build_label_map({0: "ENTAILMENT", 1: "FRUIT", 2: "CONTRADICTION"})


def test_label_map_incomplete_raises():
    with pytest.raises(ValueError, match="does not cover"):
        _build_label_map({0: "ENTAILMENT", 1: "entailment"})


# ---------------------------------------------------------------------------
# Threshold + decode tests using a fake model
# ---------------------------------------------------------------------------


@dataclass
class _FakeLogits:
    logits: object


class _FakeModel:
    """Stand-in for a HF AutoModelForSequenceClassification.

    ``logits_by_hypothesis`` maps each hypothesis string to a 3-tuple of
    pre-softmax logits at the (ENTAIL, NEUTRAL, CONTRADICT) indices.
    The production path applies softmax, so tests should pass values
    with enough gap that the resulting probability survives the
    threshold check.
    """

    def __init__(self, logits_by_hypothesis: dict[str, tuple[float, float, float]]):
        self.scores = logits_by_hypothesis

    def __call__(self, **kwargs):
        import torch

        # We stash hypotheses on the tokenizer call's return dict so
        # the fake tokenizer can pass them through.
        hyps = kwargs.pop("_hypotheses")
        rows = [self.scores[h] for h in hyps]
        return _FakeLogits(logits=torch.tensor(rows))

    def to(self, _device):
        return self

    def eval(self):
        return self


class _Passthrough:
    """List wrapper that satisfies the ``.to(device)`` contract."""

    def __init__(self, value):
        self.value = value

    def to(self, _device):
        return self.value


class _FakeTokenizer:
    """Echo the hypothesis list back so the fake model can look up scores."""

    def __call__(self, premises, hypotheses, **_kwargs):
        return {"_hypotheses": _Passthrough(hypotheses)}


@pytest.fixture
def fake_classifier():
    def _factory(
        logits_by_hypothesis: dict[str, tuple[float, float, float]],
        *,
        nei_threshold: float = 0.5,
    ) -> NLIClassifier:
        # MNLI layout: index 0 ENTAILMENT, 1 NEUTRAL, 2 CONTRADICTION
        label_map = {0: SUPPORT, 1: NEI, 2: CONTRADICT}
        return NLIClassifier._from_parts(
            model=_FakeModel(logits_by_hypothesis),
            tokenizer=_FakeTokenizer(),
            label_map=label_map,
            device="cpu",
            nei_threshold=nei_threshold,
        )

    return _factory


def test_predict_returns_support_when_entailment_dominant(fake_classifier):
    # logits (5, 0, 0) -> softmax ~ (0.985, 0.007, 0.007)
    clf = fake_classifier({"claim A": (5.0, 0.0, 0.0)})
    pred = clf.predict("claim A", "evidence text")
    assert pred.label == SUPPORT
    assert pred.support_prob == pytest.approx(0.9866, abs=1e-3)
    assert pred.contradict_prob == pytest.approx(0.0067, abs=1e-3)


def test_predict_returns_contradict_when_contradiction_dominant(fake_classifier):
    clf = fake_classifier({"claim B": (0.0, 0.0, 5.0)})
    assert clf.predict("claim B", "evidence text").label == CONTRADICT


def test_predict_returns_nei_when_both_directional_below_threshold(fake_classifier):
    # logits (1.0, 0.0, 1.1) -> softmax ~ (0.32, 0.12, 0.36) -- both directional < 0.5
    clf = fake_classifier({"claim C": (1.0, 0.0, 1.1)}, nei_threshold=0.5)
    assert clf.predict("claim C", "evidence").label == NEI


def test_predict_returns_nei_when_argmax_is_neutral_below_threshold(fake_classifier):
    # NEI clearly dominant; SUP/CON below threshold so NEI gate fires
    clf = fake_classifier({"claim D": (0.0, 5.0, 0.0)}, nei_threshold=0.5)
    pred = clf.predict("claim D", "evidence")
    assert pred.label == NEI
    assert pred.nei_prob == pytest.approx(0.9866, abs=1e-3)


def test_predict_batch_preserves_order(fake_classifier):
    clf = fake_classifier(
        {
            "c1": (5.0, 0.0, 0.0),  # SUPPORT
            "c2": (0.0, 0.0, 5.0),  # CONTRADICT
            "c3": (1.0, 0.0, 1.1),  # NEI by threshold
        },
        nei_threshold=0.5,
    )
    preds = clf.predict_batch([("c1", "e"), ("c2", "e"), ("c3", "e")])
    assert [p.label for p in preds] == [SUPPORT, CONTRADICT, NEI]


def test_predict_batch_empty(fake_classifier):
    clf = fake_classifier({})
    assert clf.predict_batch([]) == []


def test_max_directional_prob():
    pred = NLIPrediction(label=SUPPORT, support_prob=0.7, contradict_prob=0.2, nei_prob=0.1)
    assert pred.max_directional_prob == 0.7


# ---------------------------------------------------------------------------
# Heavy integration test (gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("SCIRAG_RUN_HEAVY") != "1",
    reason="Set SCIRAG_RUN_HEAVY=1 to run real-model integration tests",
)
def test_real_model_smoke():
    clf = NLIClassifier(model_name=DEFAULT_MODEL)
    pred = clf.predict(
        claim="The sky is blue.",
        evidence="On a clear day the sky appears blue due to Rayleigh scattering.",
    )
    assert pred.label == SUPPORT
    contra = clf.predict(
        claim="The sky is green.",
        evidence="On a clear day the sky appears blue due to Rayleigh scattering.",
    )
    assert contra.label in (CONTRADICT, NEI)
