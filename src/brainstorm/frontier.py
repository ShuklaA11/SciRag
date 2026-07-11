"""Subquery frontier for the brainstorm loop (v3 Phase D, SB-D1).

Bookkeeping for the agentic loop: a dedup'd queue of directions to explore.
``add`` rejects any candidate whose BGE cosine similarity to an
already-accepted direction (pending or dispatched) is ``>= dedup_threshold``
(the plan's 0.85), so the loop never revisits a near-duplicate. ``pop_batch``
hands the next directions to the loop and marks them dispatched.

Intentionally a plain stateful queue — the iteration cap lives in the loop
(SB-D3), not here. The embedder is injected as a protocol, so this is fully
testable with a fake (no model). Mirrors ``QueryDecomposer._dedup``'s
cosine-on-normalized-vectors approach; the duplication is deliberate to keep
this decoupled from the retrieval decomposer.
"""

from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np

DEFAULT_DEDUP_THRESHOLD = 0.85


class _EmbedderLike(Protocol):
    def encode_query(self, queries: list[str], batch_size: int = ...) -> np.ndarray: ...


class Frontier:
    def __init__(
        self, embedder: _EmbedderLike, *, dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD
    ) -> None:
        self._embedder = embedder
        self._threshold = dedup_threshold
        self._pending: list[str] = []
        self._seen: set[str] = set()
        self._vecs: list[np.ndarray] = []  # accepted (pending + dispatched) vectors

    def add(self, candidates: Iterable[str]) -> list[str]:
        """Add non-duplicate candidates to the queue. Returns the ones kept.

        Dedups against everything accepted so far *and* within this batch
        (a candidate's vector is registered before the next is checked)."""
        added: list[str] = []
        for cand in candidates:
            cand = cand.strip()
            if not cand:
                continue
            vec = self._embedder.encode_query([cand])[0]
            if self._is_duplicate(vec):
                continue
            self._vecs.append(vec)
            self._pending.append(cand)
            added.append(cand)
        return added

    def pop_batch(self, n: int) -> list[str]:
        """Dispatch up to ``n`` pending directions, marking them seen."""
        if n <= 0:
            return []
        batch = self._pending[:n]
        self._pending = self._pending[n:]
        self._seen.update(batch)
        return batch

    @property
    def is_exhausted(self) -> bool:
        return not self._pending

    @property
    def seen(self) -> frozenset[str]:
        return frozenset(self._seen)

    def _is_duplicate(self, vec: np.ndarray) -> bool:
        # Vectors are L2-normalized, so cosine == dot product.
        return any(float(np.dot(vec, v)) >= self._threshold for v in self._vecs)
