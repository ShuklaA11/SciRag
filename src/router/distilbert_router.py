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
        epochs: int = 6,
        lr: float = 5e-5,
        batch_size: int = 16,
        max_len: int = MAX_LEN,
        pos_weight_cap: float = 5.0,
        warmup_frac: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.classes = classes
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_len = max_len
        self.pos_weight_cap = pos_weight_cap
        self.warmup_frac = warmup_frac
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

        # Class-balanced BCE, but MILD: sqrt(neg/pos) capped at pos_weight_cap.
        # Raw neg/pos (~54x for 'abstract') pushes the model to predict every
        # class positive (degenerate 'predict-all'); unweighted collapses rare
        # classes to zero. sqrt + cap sits between: lift rare classes without
        # steamrolling to predict-all.
        pos = y.sum(axis=0)
        neg = len(y) - pos
        raw = np.sqrt(neg / np.clip(pos, 1.0, None))
        pos_weight = torch.tensor(
            np.clip(raw, None, self.pos_weight_cap), dtype=torch.float32
        ).to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Linear lr warmup then decay — standard for stable transformer
        # fine-tuning (avoids early large steps wrecking pretrained weights).
        from transformers import get_linear_schedule_with_warmup

        total_steps = self.epochs * len(loader)
        scheduler = get_linear_schedule_with_warmup(
            opt, int(self.warmup_frac * total_steps), total_steps
        )

        self._model.train()
        for _ in range(self.epochs):
            for input_ids, attn, target in loader:
                opt.zero_grad()
                logits = self._model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attn.to(self.device),
                ).logits
                loss = criterion(logits, target.to(self.device))
                loss.backward()
                opt.step()
                scheduler.step()
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
