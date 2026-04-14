"""SPECTER2 embedder for scientific paper chunks.

Loads `allenai/specter2_base` via plain transformers (no `adapters` dep),
pools the [CLS] token of the last hidden state, and L2-normalizes the
output so dot-product == cosine.

The SPECTER2 proximity adapter would give a small quality bump but
requires the `adapters` package. The adapter-free base model is a fine
flat baseline — and a deliberately weaker one gives Week 4's
section-aware variant more headroom to win.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "allenai/specter2_base"
MAX_LEN = 512
EMBED_DIM = 768


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Specter2Embedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or _pick_device()
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def _ensure_model(self):
        if self._model is None:
            model = AutoModel.from_pretrained(self.model_name)
            model.eval()
            try:
                model.to(self.device)
            except (RuntimeError, NotImplementedError):
                self.device = "cpu"
                model.to(self.device)
            self._model = model

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        self._ensure_model()
        tok = self.tokenizer
        out = np.empty((len(texts), EMBED_DIM), dtype=np.float32)

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(self.device)
            hidden = self._model(**enc).last_hidden_state
            cls = hidden[:, 0, :]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            out[start : start + len(batch)] = cls.cpu().numpy().astype(np.float32)

        return out
