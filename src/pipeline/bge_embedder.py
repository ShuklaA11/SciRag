"""BGE-base-en-v1.5 embedder for query→paragraph retrieval.

Drop-in replacement for Specter2Embedder. Same 768d output, [CLS]
pooling, L2-normalized, so the FAISS IndexFlatIP and `chunks.jsonl`
formats stay identical — only the index directory differs.

Asymmetric encoding: BGE-v1.5 expects an instruction prefix on the
*query* side only. Passages are encoded plain. Skipping the query
prefix costs 2-4 points of recall on retrieval tasks, so callers
must use `encode_query()` for queries and `encode()` for passages.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
MAX_LEN = 512
EMBED_DIM = 768
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BGEEmbedder:
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
    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
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

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Encode passages (no instruction prefix)."""
        return self._encode(texts, batch_size)

    def encode_query(self, queries: list[str], batch_size: int = 16) -> np.ndarray:
        """Encode queries with the BGE retrieval instruction prefix."""
        prefixed = [QUERY_PREFIX + q for q in queries]
        return self._encode(prefixed, batch_size)
