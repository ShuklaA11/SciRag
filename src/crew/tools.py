"""Grounded specialist capabilities for the research crew (SB-X1).

Each method wraps one injected SciRAG component and returns a uniform
:class:`ToolResult` — a short LLM-readable ``summary`` (what the supervisor
reads to decide the next hop) plus a structured ``data`` payload (the auditable
findings). No LLM here: this is pure, deterministic dispatch, so tests inject
fake components and assert exact outputs with no model in the loop.

The three capabilities mirror the crew's specialists:
  - ``search_corpus``  -> retrieval (FlatIndex)
  - ``verify_idea``    -> idea evaluation (IdeaEvaluator + NLI)
  - ``find_gaps``      -> novelty/gap discovery (BrainstormLoop)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolResult:
    """Uniform specialist output: a short summary for the supervisor to read
    and a structured payload for synthesis / auditing."""

    tool: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)


class _RetrieverLike(Protocol):
    def search(self, query: str, k: int = ...) -> list[dict]: ...


class _EvaluatorLike(Protocol):
    def evaluate(self, idea: str) -> Any: ...


class _BrainstormLike(Protocol):
    def run(self, seed: str, seed_directions: list[str]) -> Any: ...


def _snippet(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


class CrewTools:
    """Dependency-injected wrappers over the benchmarked components."""

    def __init__(
        self,
        retriever: _RetrieverLike,
        evaluator: _EvaluatorLike,
        brainstorm: _BrainstormLike,
    ) -> None:
        self.retriever = retriever
        self.evaluator = evaluator
        self.brainstorm = brainstorm

    def search_corpus(self, query: str, k: int = 5) -> ToolResult:
        chunks = self.retriever.search(query, k=k)
        if not chunks:
            return ToolResult(
                "search_corpus", f'No corpus matches for "{_snippet(query, 80)}".',
                {"query": query, "chunks": []},
            )
        lines = [
            f"[{c.get('arxiv_id', '?')}/{c.get('section_type', '?')}] "
            f"{_snippet(c.get('text', ''))}"
            for c in chunks
        ]
        summary = f"Retrieved {len(chunks)} chunk(s):\n" + "\n".join(lines)
        return ToolResult("search_corpus", summary, {"query": query, "chunks": chunks})

    def verify_idea(self, idea: str) -> ToolResult:
        report = self.evaluator.evaluate(idea)
        verdicts = list(report.verdicts)
        counts: dict[str, int] = {}
        for v in verdicts:
            counts[v.bucket] = counts.get(v.bucket, 0) + 1
        dist = ", ".join(f"{n} {b}" for b, n in sorted(counts.items())) or "no claims"
        summary = (
            f"Evaluated {len(verdicts)} claim(s) — {dist}.\n"
            + "\n".join(f"- [{v.bucket}] {_snippet(v.claim, 120)}" for v in verdicts)
        )
        return ToolResult(
            "verify_idea",
            summary,
            {
                "idea": idea,
                "bucket_counts": counts,
                "verdicts": [
                    {
                        "claim": v.claim,
                        "bucket": v.bucket,
                        "best_support": v.best_support,
                        "best_contradict": v.best_contradict,
                    }
                    for v in verdicts
                ],
            },
        )

    def find_gaps(
        self, seed: str, seed_directions: list[str] | None = None
    ) -> ToolResult:
        report = self.brainstorm.run(seed, seed_directions or [])
        directions = list(report.directions)
        summary = (
            f"Discovered {len(directions)} novel direction(s) from "
            f'"{_snippet(seed, 80)}" ({report.iterations} iteration(s)):\n'
            + "\n".join(f"- {_snippet(d.claim, 120)}" for d in directions)
        )
        return ToolResult(
            "find_gaps",
            summary,
            {
                "seed": seed,
                "iterations": report.iterations,
                "directions": [d.claim for d in directions],
            },
        )
