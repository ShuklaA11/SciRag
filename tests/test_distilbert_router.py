"""Unit tests for DistilBertRouter decision logic — no transformer in the loop.

The transformer forward/training is exercised by the real train + eval runs
(E-R1c.2/.3). Here we canned-inject predict_proba to test that the
threshold/top_n decision maps proba rows to RouterPrediction correctly, plus
the fit/predict guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.router.distilbert_router import DistilBertRouter
from src.router.tfidf_classifier import SECTION_TYPES, RouterPrediction

# ('abstract','introduction','related_work','method','experiments',
#  'results','conclusion','other') -> indices 0..7
CLASSES = SECTION_TYPES


def _canned(row: list[float]):
    arr = np.array([row], dtype=np.float32)
    return lambda questions: arr


def test_predict_threshold_and_top_n_union():
    router = DistilBertRouter()
    router._fitted = True
    # results=0.9 (idx5), method=0.6 (idx3), rest 0.1
    row = [0.1] * len(CLASSES)
    row[5] = 0.9
    row[3] = 0.6
    router.predict_proba = _canned(row)

    pred = router.predict(["q"], threshold=0.5, top_n=1)[0]

    assert isinstance(pred, RouterPrediction)
    # threshold>=0.5 -> {method, results}; top_1 -> {results}; union
    assert set(pred.labels) == {"method", "results"}
    assert set(pred.probabilities) == set(CLASSES)


def test_predict_top_n_pulls_in_subthreshold_label():
    router = DistilBertRouter()
    router._fitted = True
    # all below threshold; top_2 should still surface the two highest
    row = [0.1] * len(CLASSES)
    row[1] = 0.4  # introduction
    row[4] = 0.3  # experiments
    router.predict_proba = _canned(row)

    pred = router.predict(["q"], threshold=0.5, top_n=2)[0]

    assert set(pred.labels) == {"introduction", "experiments"}


def test_predict_before_fit_raises():
    router = DistilBertRouter()
    with pytest.raises(RuntimeError):
        router.predict(["anything"])


def test_fit_length_mismatch_raises_before_model_load():
    router = DistilBertRouter()
    with pytest.raises(ValueError):
        router.fit(["only one"], [["method"], ["results"]])
