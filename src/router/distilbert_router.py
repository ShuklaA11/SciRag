"""Fine-tuned DistilBERT multi-label section router.

Same interface as :class:`TfidfRouter` / :class:`EmbeddingRouter`
(``fit`` / ``predict`` / ``predict_proba`` / ``save`` / ``load`` and the
shared :class:`RouterPrediction`), but instead of a linear head on frozen
features it fine-tunes the encoder end-to-end for the section-routing task.

R1b's ceiling (~0.745 with union routing) came from a linear head on frozen
BGE embeddings: the classifier only sees whatever section-signal BGE already
encodes, which is too weak to confidently pick small section sets. R1c
unfreezes the encoder so its representations are trained for this task —
the bet being it can filter harder (higher threshold) without losing
evidence, realizing headroom above 0.745 when paired with union routing.

Multi-label via HuggingFace ``problem_type="multi_label_classification"``
(sigmoid per class + BCE loss). Training defaults to CPU: the documented
MPS-NaN risk was on a larger fine-tune, but DistilBERT-base on ~2k examples
is small enough that CPU is safe and fast; MPS is opt-in for speed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from src.router.tfidf_classifier import SECTION_TYPES, RouterPrediction

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256


class DistilBertRouter:
    """Fine-tuned DistilBERT multi-label section router."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        classes: tuple[str, ...] = SECTION_TYPES,
        device: str = "cpu",
        epochs: int = 3,
        lr: float = 5e-5,
        batch_size: int = 16,
        max_len: int = MAX_LEN,
    ) -> None:
        self.model_name = model_name
        self.classes = classes
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_len = max_len
        self.binarizer = MultiLabelBinarizer(classes=list(classes))
        self.binarizer.fit([list(classes)])
        self._tokenizer = None
        self._model = None
        self._fitted = False

    def _ensure_model(self, from_pretrained: str | None = None) -> None:
        if self._model is not None:
            return
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        src = from_pretrained or self.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(src)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            src,
            num_labels=len(self.classes),
            problem_type="multi_label_classification",
        ).to(self.device)

    def fit(
        self, questions: list[str], labels: list[list[str]]
    ) -> "DistilBertRouter":
        if len(questions) != len(labels):
            raise ValueError("questions and labels must have the same length")
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self._ensure_model()
        y = self.binarizer.transform(labels).astype(np.float32)
        enc = self._tokenizer(
            questions,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        ds = TensorDataset(
            enc["input_ids"], enc["attention_mask"], torch.from_numpy(y)
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr)

        self._model.train()
        for _ in range(self.epochs):
            for input_ids, attn, target in loader:
                opt.zero_grad()
                out = self._model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attn.to(self.device),
                    labels=target.to(self.device),
                )
                out.loss.backward()
                opt.step()
        self._fitted = True
        return self

    def predict_proba(self, questions: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("DistilBertRouter is not fitted")
        import torch

        self._model.eval()
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(questions), self.batch_size):
                batch = questions[i : i + self.batch_size]
                enc = self._tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                logits = self._model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                ).logits
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(probs) if probs else np.zeros((0, len(self.classes)))

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
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        (path / "router_meta.json").write_text(
            json.dumps({"classes": list(self.classes), "max_len": self.max_len})
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "DistilBertRouter":
        path = Path(path)
        meta = json.loads((path / "router_meta.json").read_text())
        obj = cls(
            classes=tuple(meta["classes"]),
            device=device,
            max_len=meta["max_len"],
        )
        obj._ensure_model(from_pretrained=str(path))
        obj._fitted = True
        return obj
