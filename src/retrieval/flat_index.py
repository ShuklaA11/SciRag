"""Load + search wrapper around the flat FAISS index.

The flat index is a FAISS IndexFlatIP over L2-normalized SPECTER2
embeddings (so inner product == cosine). Each row corresponds 1:1 with a
line in chunks.jsonl, written together by scripts/build_flat_index.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.pipeline.embedder import EMBED_DIM, Specter2Embedder

DEFAULT_INDEX_DIR = Path("data/index/flat")


class FlatIndex:
    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        embedder: Specter2Embedder | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.embedder = embedder or Specter2Embedder()

        index_path = self.index_dir / "index.faiss"
        chunks_path = self.index_dir / "chunks.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks sidecar not found: {chunks_path}")

        self.index = faiss.read_index(str(index_path))
        self.chunks: list[dict] = [
            json.loads(line) for line in chunks_path.read_text().splitlines() if line
        ]

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Index/chunks misalignment: index has {self.index.ntotal} rows, "
                f"chunks.jsonl has {len(self.chunks)} lines"
            )

    def search(
        self,
        query: str,
        k: int = 5,
        paper_ids: set[str] | None = None,
    ) -> list[dict]:
        """Search the index for top-k chunks matching `query`.

        paper_ids: if given, restrict results to chunks whose arxiv_id is
        in the set. Implemented by overfetching (k * OVERFETCH) and
        filtering in Python — correct for our scale (≤1M chunks).
        """
        if self.index.ntotal == 0:
            return []
        q_vec = self.embedder.encode([query])
        if q_vec.shape[1] != self.index.d:
            raise ValueError(
                f"Query dim {q_vec.shape[1]} != index dim {self.index.d}"
            )

        # When paper_ids filter is active, scan the full index so we
        # don't miss in-paper chunks that rank low globally. Flat
        # IndexFlatIP is O(ntotal * dim) either way; only sort cost
        # scales with fetch_k, which stays tiny.
        fetch_k = self.index.ntotal if paper_ids is not None else min(k, self.index.ntotal)
        scores, ids = self.index.search(q_vec.astype(np.float32), fetch_k)
        results: list[dict] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if paper_ids is not None and chunk["arxiv_id"] not in paper_ids:
                continue
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "arxiv_id": chunk["arxiv_id"],
                    "chunk_idx": chunk["chunk_idx"],
                    "text": chunk["text"],
                    "score": float(score),
                }
            )
            if len(results) >= k:
                break
        return results
