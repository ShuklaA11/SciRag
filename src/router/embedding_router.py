"""Multi-label BGE-embedding + one-vs-rest logistic regression router.

Drop-in replacement for :class:`TfidfRouter` with the same interface
(``fit`` / ``predict`` / ``predict_proba`` / ``save`` / ``load`` and the
shared :class:`RouterPrediction` result). Only the features differ: dense
BGE question embeddings instead of sparse TF-IDF vectors.

R1a showed the TF-IDF router can't discriminate sections (routes ``other``
for ~98% of questions); dense embeddings put semantically-similar questions
near each other, giving a linear head a real signal to separate on. No
transformer fine-tuning happens here — the BGE embedder is frozen and only
the logistic-regression head is trained — so there is no MPS-NaN risk.

The embedder is dependency-injected. Questions are encoded with
``encode_query`` (the asymmetric query side), matching how the retrieval
index encodes questions. ``save``/``load`` persist only the trained head;
the embedder is reconstructed lazily so artifacts stay light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.router.tfidf_classifier import SECTION_TYPES, RouterPrediction


class QueryEmbedder(Protocol):
    """Minimal seam the router needs from an embedder (BGE or a fake)."""

    def encode_query(self, queries: list[str], batch_size: int = ...) -> np.ndarray: ...


class EmbeddingRouter:
    """Multi-label BGE-embedding + OvR logistic regression router."""

    def __init__(
        self,
        embedder: QueryEmbedder | None = None,
        C: float = 1.0,
        classes: tuple[str, ...] = SECTION_TYPES,
    ) -> None:
        self.classes = classes
        self._embedder = embedder
        self.binarizer = MultiLabelBinarizer(classes=list(classes))
        self.binarizer.fit([list(classes)])
        self.model = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
            )
        )
        self._fitted = False

    @property
    def embedder(self) -> QueryEmbedder:
        """Lazily build the real BGE embedder if none was injected."""
        if self._embedder is None:
            from src.pipeline.bge_embedder import BGEEmbedder

            self._embedder = BGEEmbedder()
        return self._embedder

    def fit(self, questions: list[str], labels: list[list[str]]) -> "EmbeddingRouter":
        if len(questions) != len(labels):
            raise ValueError("questions and labels must have the same length")
        x = self.embedder.encode_query(questions)
        y = self.binarizer.transform(labels)
        self.model.fit(x, y)
        self._fitted = True
        return self

    def predict_proba(self, questions: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EmbeddingRouter is not fitted")
        x = self.embedder.encode_query(questions)
        return self.model.predict_proba(x)

    def predict(
        self,
        questions: list[str],
        threshold: float = 0.5,
        top_n: int = 2,
    ) -> list[RouterPrediction]:
        proba = self.predict_proba(questions)
        out: list[RouterPrediction] = []
        for row in proba:
            probs = {self.classes[i]: float(row[i]) for i in range(len(self.classes))}
            by_threshold = {c for c, p in probs.items() if p >= threshold}
            top_idx = np.argsort(row)[::-1][:top_n]
            by_top = {self.classes[int(i)] for i in top_idx}
            labels = tuple(sorted(by_threshold | by_top))
            out.append(RouterPrediction(labels=labels, probabilities=probs))
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classes": self.classes,
                "binarizer": self.binarizer,
                "model": self.model,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: Path, embedder: QueryEmbedder | None = None
    ) -> "EmbeddingRouter":
        blob = joblib.load(path)
        obj = cls(embedder=embedder, classes=tuple(blob["classes"]))
        obj.binarizer = blob["binarizer"]
        obj.model = blob["model"]
        obj._fitted = blob["fitted"]
        return obj
