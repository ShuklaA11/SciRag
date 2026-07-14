"""Unit tests for EmbeddingRouter — fake embedder, no models in the loop.

The fake maps each question to a per-class one-hot direction based on the
class name it mentions, so a linear head can separate them perfectly. This
exercises the fit/predict/save/load interface without touching real BGE.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.router.embedding_router import EmbeddingRouter
from src.router.tfidf_classifier import SECTION_TYPES, RouterPrediction

CLASSES = SECTION_TYPES


class FakeEmbedder:
    """Deterministic embedder: one dimension per class, set when the class
    name appears in the question. Linearly separable by construction."""

    def __init__(self, classes: tuple[str, ...] = CLASSES) -> None:
        self.classes = classes

    def encode_query(self, queries: list[str], batch_size: int = 16) -> np.ndarray:
        vecs = np.zeros((len(queries), len(self.classes)), dtype=np.float32)
        for i, q in enumerate(queries):
            ql = q.lower()
            for j, c in enumerate(self.classes):
                if c in ql:
                    vecs[i, j] = 1.0
        return vecs


def _training_data() -> tuple[list[str], list[list[str]]]:
    """A few separable examples per class."""
    questions: list[str] = []
    labels: list[list[str]] = []
    for c in CLASSES:
        for template in ("Tell me about the {}.", "What is in the {} part?",
                         "A question concerning the {}."):
            questions.append(template.format(c))
            labels.append([c])
    return questions, labels


def test_fit_returns_self_and_predictions_use_known_classes():
    q, y = _training_data()
    router = EmbeddingRouter(embedder=FakeEmbedder()).fit(q, y)

    preds = router.predict(["Tell me about the results."], threshold=0.5, top_n=1)

    assert len(preds) == 1
    assert isinstance(preds[0], RouterPrediction)
    assert set(preds[0].labels) <= set(CLASSES)
    assert set(preds[0].probabilities) == set(CLASSES)


def test_predict_separates_obvious_classes():
    q, y = _training_data()
    router = EmbeddingRouter(embedder=FakeEmbedder()).fit(q, y)

    preds = router.predict(
        ["A question concerning the results."], threshold=0.5, top_n=1
    )

    assert "results" in preds[0].labels


def test_save_load_roundtrip_preserves_predictions(tmp_path):
    q, y = _training_data()
    router = EmbeddingRouter(embedder=FakeEmbedder()).fit(q, y)
    path = tmp_path / "embedding.joblib"
    router.save(path)

    loaded = EmbeddingRouter.load(path, embedder=FakeEmbedder())

    query = ["What is in the method part?"]
    before = router.predict(query, threshold=0.3, top_n=2)[0]
    after = loaded.predict(query, threshold=0.3, top_n=2)[0]
    assert before.labels == after.labels
    assert before.probabilities == pytest.approx(after.probabilities)


def test_fit_length_mismatch_raises():
    router = EmbeddingRouter(embedder=FakeEmbedder())
    with pytest.raises(ValueError):
        router.fit(["only one question"], [["method"], ["results"]])


def test_predict_before_fit_raises():
    router = EmbeddingRouter(embedder=FakeEmbedder())
    with pytest.raises(RuntimeError):
        router.predict(["anything"])
