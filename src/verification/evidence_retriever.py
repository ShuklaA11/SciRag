"""BM25 evidence retrieval over the SciFact corpus.

Indexes the corpus at abstract granularity: each document is one "doc"
in the BM25 sense, with text = title + " " + concatenated abstract
sentences. Retrieval returns the top-k doc_ids per claim, scored
against the full corpus (no oracle).

This is the SB9.2 swap-in for SB9.1's oracle premise: instead of
feeding the NLI model the cited abstract directly, we feed it the
top-1 (or top-k merged) BM25-retrieved abstract. The harness then
reports two metrics side by side:

  * retrieval recall@k against the gold ``cited_doc_ids`` set
  * end-to-end label accuracy on the same (claim, doc) pairs as SB9.1

Tokenisation is intentionally minimal (lowercase + word-character
split) — rank_bm25 expects pre-tokenised lists. SciFact abstracts are
clean English so no stemming or stopword removal is layered in; that
is a knob to revisit only if retrieval recall is the bottleneck.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase + alnum-word tokeniser. Cheap and adequate for SciFact abstracts."""
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def _doc_text(doc: dict[str, Any]) -> str:
    """Concatenate title + abstract sentences into a single BM25 document."""
    title = doc.get("title", "") or ""
    abstract = doc.get("abstract", []) or []
    return (title + " " + " ".join(abstract)).strip()


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: int
    score: float


class BM25EvidenceRetriever:
    """BM25 retriever over a SciFact-style corpus.

    ``corpus`` is the dict produced by ``scifact_eval.load_corpus``:
    ``{doc_id: {title, abstract: [sent, ...]}}``.
    """

    def __init__(self, corpus: dict[int, dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi

        # Preserve a stable ordering so BM25 indices map cleanly back to
        # doc_ids — sort by doc_id for reproducibility.
        self._doc_ids: list[int] = sorted(corpus.keys())
        tokenised = [tokenize(_doc_text(corpus[d])) for d in self._doc_ids]
        self._bm25 = BM25Okapi(tokenised)

    def __len__(self) -> int:
        return len(self._doc_ids)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        if k <= 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        # Argpartition + sort the top-k for O(N + k log k) instead of O(N log N).
        n = len(scores)
        if k >= n:
            order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        else:
            # numpy is a transitive dep via rank_bm25; use it for argpartition.
            import numpy as np

            partial = np.argpartition(-scores, k - 1)[:k]
            order = sorted(partial.tolist(), key=lambda i: scores[i], reverse=True)
        return [RetrievedDoc(doc_id=self._doc_ids[i], score=float(scores[i])) for i in order]

    def retrieve_many(
        self, queries: Iterable[str], k: int = 5
    ) -> list[list[RetrievedDoc]]:
        return [self.retrieve(q, k=k) for q in queries]
