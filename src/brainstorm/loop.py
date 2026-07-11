"""Brainstorm loop orchestrator (v3 Phase D, SB-D3).

The capped agentic loop, as a plain iterative loop (not LangGraph — linear flow
with a stop condition, per the codebase precedent):

    seed directions → frontier
    repeat up to max_iters, until the frontier drains:
        pop a batch → assess each as a claim (Phase C) → collect NOVEL as gaps
        → LLM proposes next directions from the gaps → dedup into the frontier

Output is the set of discovered gaps (NOVEL directions) — per-direction
ClaimVerdicts, never a scalar. Convergence is natural: when a batch yields no
gaps, the proposer returns nothing, the frontier drains, and the loop stops
before max_iters.

Citation-expansion is not a loop step — it lives in the retriever adapter
injected into the evaluator (a real-run wiring detail), keeping the loop linear.
All deps are injected protocols → fake-testable, no models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from src.brainstorm.directions import gaps_from_verdicts
from src.ideas import NOVEL, ClaimVerdict

DEFAULT_MAX_ITERS = 3
DEFAULT_BATCH_SIZE = 5


@dataclass(frozen=True)
class BrainstormReport:
    seed: str
    directions: tuple[ClaimVerdict, ...]  # discovered NOVEL gaps (the output)
    iterations: int
    n_assessed: int
    max_iters: int
    batch_size: int


class _EvaluatorLike(Protocol):
    def evaluate_claims(self, claims: list[str], *, idea: str = ...) -> Any: ...


class _ProposerLike(Protocol):
    def propose(self, seed: str, gaps: list[str]) -> list[str]: ...


class _FrontierLike(Protocol):
    def add(self, candidates: Any) -> list[str]: ...
    def pop_batch(self, n: int) -> list[str]: ...
    @property
    def is_exhausted(self) -> bool: ...


class BrainstormLoop:
    def __init__(
        self,
        evaluator: _EvaluatorLike,
        proposer: _ProposerLike,
        frontier_factory: Callable[[], _FrontierLike],
        *,
        max_iters: int = DEFAULT_MAX_ITERS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.evaluator = evaluator
        self.proposer = proposer
        self.frontier_factory = frontier_factory
        self.max_iters = max_iters
        self.batch_size = batch_size

    def run(self, seed: str, seed_directions: list[str]) -> BrainstormReport:
        frontier = self.frontier_factory()
        frontier.add(seed_directions)

        discovered: list[ClaimVerdict] = []
        seen_gaps: set[str] = set()
        iterations = 0
        n_assessed = 0

        for _ in range(self.max_iters):
            if frontier.is_exhausted:
                break
            batch = frontier.pop_batch(self.batch_size)
            if not batch:
                break
            iterations += 1

            verdicts = self.evaluator.evaluate_claims(batch).verdicts
            n_assessed += len(verdicts)
            for v in verdicts:
                if v.bucket == NOVEL and v.claim not in seen_gaps:
                    seen_gaps.add(v.claim)
                    discovered.append(v)

            proposals = self.proposer.propose(seed, gaps_from_verdicts(verdicts))
            frontier.add(proposals)

        return BrainstormReport(
            seed=seed,
            directions=tuple(discovered),
            iterations=iterations,
            n_assessed=n_assessed,
            max_iters=self.max_iters,
            batch_size=self.batch_size,
        )
