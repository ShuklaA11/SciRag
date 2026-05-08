"""Multi-label TF-IDF + one-vs-rest logistic regression classifier.

Maps a QASPER-style question to a set of canonical section types
(``abstract``, ``introduction``, ``related_work``, ``method``,
``experiments``, ``results``, ``conclusion``, ``other``). Used as the
floor model for the Week 5 query router; if this can't beat random,
neither will DistilBERT.

Predictions are returned as ``(labels, probabilities)``: the labels with
``prob >= threshold`` plus the top-``top_n`` by probability (union). The
caller decides how to interpret an empty set (Week 5 falls back to flat
retrieval).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

SECTION_TYPES: tuple[str, ...] = (
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiments",
    "results",
    "conclusion",
    "other",
)


@dataclass(frozen=True)
class RouterPrediction:
    labels: tuple[str, ...]
    probabilities: dict[str, float]


class TfidfRouter:
    """Multi-label TF-IDF + OvR logistic regression router."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        C: float = 1.0,
        classes: tuple[str, ...] = SECTION_TYPES,
    ) -> None:
        self.classes = classes
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            lowercase=True,
        )
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

    def fit(self, questions: list[str], labels: list[list[str]]) -> "TfidfRouter":
        if len(questions) != len(labels):
            raise ValueError("questions and labels must have the same length")
        x = self.vectorizer.fit_transform(questions)
        y = self.binarizer.transform(labels)
        self.model.fit(x, y)
        self._fitted = True
        return self

    def predict_proba(self, questions: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfRouter is not fitted")
        x = self.vectorizer.transform(questions)
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
                "vectorizer": self.vectorizer,
                "binarizer": self.binarizer,
                "model": self.model,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TfidfRouter":
        blob = joblib.load(path)
        obj = cls(classes=tuple(blob["classes"]))
        obj.vectorizer = blob["vectorizer"]
        obj.binarizer = blob["binarizer"]
        obj.model = blob["model"]
        obj._fitted = blob["fitted"]
        return obj
