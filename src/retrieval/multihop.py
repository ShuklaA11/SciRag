"""Multi-hop retriever: decompose → fan-out → merge → re-rank → top-k.

Wraps the W6 retrieve+rerank pipeline so each sub-question runs its own
retrieval, then candidates are merged (dedup by chunk_id) and re-ranked
against the *original* question for the final top-k. The atomic case
(single sub-Q) skips the second rerank since the per-sub-Q rerank is
already against the only query.

This is intentionally a plain class — not a LangGraph state machine —
because the flow is linear with one fan-out and no conditional
branches. Revisit if W8 polish needs retries or per-hop branching.
"""

from __future__ import annotations

import time
from typing import Protocol, Sequence


class _Decomposer(Protocol):
    def decompose(self, question: str) -> list[str]: ...


class _Index(Protocol):
    def search(
        self,
        query: str,
        k: int = ...,
        paper_ids: set[str] | None = ...,
        section_types: set[str] | None = ...,
    ) -> list[dict]: ...


class _Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[dict],
        top_k: int | None = ...,
        batch_size: int = ...,
    ): ...


class MultiHopRetriever:
    def __init__(
        self,
        decomposer: _Decomposer,
        flat_index: _Index,
        reranker: _Reranker | None = None,
        retrieve_k: int = 10,
        top_k: int = 5,
    ) -> None:
        self.decomposer = decomposer
        self.flat_index = flat_index
        self.reranker = reranker
        self.retrieve_k = retrieve_k
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        paper_ids: set[str] | None = None,
        section_types: set[str] | None = None,
    ) -> tuple[list[dict], dict]:
        t0 = time.perf_counter()
        sub_qs = self.decomposer.decompose(question)
        decompose_ms = (time.perf_counter() - t0) * 1000

        merged: list[dict] = []
        seen: set[str] = set()
        retrieve_ms = 0.0
        rerank_ms = 0.0

        for sub_q in sub_qs:
            t_r = time.perf_counter()
            cands = self.flat_index.search(
                sub_q, k=self.retrieve_k,
                paper_ids=paper_ids, section_types=section_types,
            )
            retrieve_ms += (time.perf_counter() - t_r) * 1000

            if self.reranker is not None and cands:
                t_ce = time.perf_counter()
                ranked = self.reranker.rerank(
                    sub_q, cands, top_k=self.retrieve_k,
                )
                rerank_ms += (time.perf_counter() - t_ce) * 1000
                cands = [r.chunk for r in ranked]

            for c in cands:
                cid = c["chunk_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                merged.append(c)

        if not merged:
            return [], self._meta(sub_qs, 0, decompose_ms, retrieve_ms, rerank_ms)

        # Atomic: per-sub-Q rerank is already against the original. Skip
        # the second pass and just truncate.
        if len(sub_qs) == 1:
            final = merged[: self.top_k]
        elif self.reranker is not None:
            t_ce = time.perf_counter()
            ranked = self.reranker.rerank(question, merged, top_k=self.top_k)
            rerank_ms += (time.perf_counter() - t_ce) * 1000
            final = [r.chunk for r in ranked]
        else:
            # No reranker → fall back to bi-encoder score sort.
            final = sorted(
                merged, key=lambda c: float(c.get("score", 0.0)), reverse=True,
            )[: self.top_k]

        return final, self._meta(
            sub_qs, len(merged), decompose_ms, retrieve_ms, rerank_ms,
        )

    @staticmethod
    def _meta(
        sub_qs: list[str],
        n_unique: int,
        decompose_ms: float,
        retrieve_ms: float,
        rerank_ms: float,
    ) -> dict:
        return {
            "sub_questions": sub_qs,
            "n_sub_questions": len(sub_qs),
            "n_unique_candidates": n_unique,
            "decompose_ms": round(decompose_ms, 2),
            "retrieve_ms": round(retrieve_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
        }
