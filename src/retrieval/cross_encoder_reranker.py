"""Cross-encoder reranker over top-K bi-encoder candidates.

Wraps a sentence-transformers ``CrossEncoder`` so callers can hand it
``(question, candidate_chunks)`` and get back the same chunks reordered
by relevance score. Default model is the MiniLM-L6 MS-MARCO checkpoint
specified in PLAN.md Week 6 (~80MB, ~50ms/batch on M-series MPS).

The reranker is intentionally state-light: load once, reuse across
queries. Latency-sensitive callers should batch all candidates for a
single question into one ``predict`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RerankResult:
    chunk: dict
    bi_score: float
    ce_score: float


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        if device is None:
            try:
                import torch

                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except ImportError:  # pragma: no cover
                device = "cpu"

        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict],
        top_k: int | None = None,
        batch_size: int = 32,
    ) -> list[RerankResult]:
        """Return ``candidates`` reordered by cross-encoder score, descending.

        ``candidates`` are dicts with at least a ``text`` field and an
        optional ``score`` field (the bi-encoder score; preserved for
        downstream introspection).
        """
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(
            pairs, batch_size=batch_size, show_progress_bar=False,
        )
        ranked = sorted(
            (
                RerankResult(
                    chunk=c,
                    bi_score=float(c.get("score", 0.0)),
                    ce_score=float(s),
                )
                for c, s in zip(candidates, scores)
            ),
            key=lambda r: r.ce_score,
            reverse=True,
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked
