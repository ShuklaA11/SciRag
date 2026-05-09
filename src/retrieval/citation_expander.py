"""1-hop citation expansion over the SciRAG citation graph.

Loads ``data/citation_graph/graph.pickle`` (a NetworkX DiGraph with
``in_corpus`` flags on nodes) and exposes the in-corpus 1-hop neighbor
set of any paper. "1-hop" means immediate successors (papers this one
cites) plus immediate predecessors (papers that cite this one). External
papers (no chunks indexed) are filtered out so callers can hand the
result straight to ``FlatIndex.search(paper_ids=...)``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

DEFAULT_GRAPH_PATH = Path("data/citation_graph/graph.pickle")


class CitationExpander:
    def __init__(self, graph_path: str | Path = DEFAULT_GRAPH_PATH) -> None:
        self.graph_path = Path(graph_path)
        with self.graph_path.open("rb") as f:
            self.graph = pickle.load(f)
        self.in_corpus: frozenset[str] = frozenset(
            n for n, d in self.graph.nodes(data=True) if d.get("in_corpus")
        )

    def neighbors(
        self,
        paper_id: str,
        *,
        in_corpus_only: bool = True,
        directions: str = "both",
    ) -> set[str]:
        """Return the 1-hop neighbor set of ``paper_id``.

        ``directions``:
            - "out"  : papers this one cites (successors)
            - "in"   : papers that cite this one (predecessors)
            - "both" : union of out and in
        """
        if paper_id not in self.graph:
            return set()
        if directions not in {"out", "in", "both"}:
            raise ValueError(f"unknown directions: {directions!r}")
        out: set[str] = set()
        if directions in {"out", "both"}:
            out |= set(self.graph.successors(paper_id))
        if directions in {"in", "both"}:
            out |= set(self.graph.predecessors(paper_id))
        out.discard(paper_id)
        if in_corpus_only:
            out &= self.in_corpus
        return out

    def expanded_paper_ids(
        self,
        paper_id: str,
        *,
        in_corpus_only: bool = True,
        directions: str = "both",
    ) -> set[str]:
        """Convenience: neighbors ∪ {paper_id} for retrieval filters."""
        nbrs = self.neighbors(
            paper_id, in_corpus_only=in_corpus_only, directions=directions,
        )
        if not in_corpus_only or paper_id in self.in_corpus:
            nbrs.add(paper_id)
        return nbrs
