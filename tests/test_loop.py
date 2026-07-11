"""SB-D3: brainstorm loop orchestrator. All deps faked (no models). Guards
control flow: cap termination, natural convergence, gap accumulation, and that
proposals feed back into the frontier."""

from __future__ import annotations

from src.brainstorm import BrainstormLoop, BrainstormReport
from src.ideas import CONTRADICTED, ENTAILED, NOVEL, ClaimVerdict


class _Report:
    def __init__(self, verdicts) -> None:
        self.verdicts = tuple(verdicts)


class FakeEvaluator:
    def __init__(self, buckets: dict[str, str]) -> None:
        self.buckets = buckets

    def evaluate_claims(self, claims, *, idea: str = "") -> _Report:
        return _Report(
            ClaimVerdict(c, self.buckets.get(c, ENTAILED), 0.0, 0.0, 0, None) for c in claims
        )


class FakeProposer:
    """propose → concat of per-gap follow-ups (a chain, to exercise iteration)."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping

    def propose(self, seed: str, gaps: list[str]) -> list[str]:
        out: list[str] = []
        for g in gaps:
            out.extend(self.mapping.get(g, []))
        return out


class FakeFrontier:
    def __init__(self) -> None:
        self._pending: list[str] = []
        self._seen: set[str] = set()

    def add(self, candidates) -> list[str]:
        added = []
        for c in candidates:
            if c and c not in self._seen and c not in self._pending:
                self._pending.append(c)
                added.append(c)
        return added

    def pop_batch(self, n: int) -> list[str]:
        batch = self._pending[:n]
        self._pending = self._pending[n:]
        self._seen.update(batch)
        return batch

    @property
    def is_exhausted(self) -> bool:
        return not self._pending


def test_stops_at_max_iters_when_gaps_keep_appearing():
    # d0→d1→d2→d3 all NOVEL, one per iteration; proposer chains the next.
    ev = FakeEvaluator({f"d{i}": NOVEL for i in range(5)})
    proposer = FakeProposer({f"d{i}": [f"d{i + 1}"] for i in range(5)})
    loop = BrainstormLoop(ev, proposer, FakeFrontier, max_iters=3, batch_size=1)

    report = loop.run("seed", ["d0"])

    assert isinstance(report, BrainstormReport)
    assert report.iterations == 3  # capped
    assert [v.claim for v in report.directions] == ["d0", "d1", "d2"]
    assert report.n_assessed == 3


def test_converges_before_cap_when_no_gaps():
    ev = FakeEvaluator({"d0": ENTAILED})  # not a gap → no proposals
    loop = BrainstormLoop(ev, FakeProposer({}), FakeFrontier, max_iters=5, batch_size=1)

    report = loop.run("seed", ["d0"])

    assert report.iterations == 1  # frontier drained, stopped early
    assert report.directions == ()


def test_only_novel_directions_are_reported():
    ev = FakeEvaluator({"a": NOVEL, "b": ENTAILED, "c": CONTRADICTED})
    loop = BrainstormLoop(ev, FakeProposer({}), FakeFrontier, max_iters=2, batch_size=5)

    report = loop.run("seed", ["a", "b", "c"])

    assert [v.claim for v in report.directions] == ["a"]
    assert report.n_assessed == 3
    assert report.iterations == 1


def test_empty_seed_yields_empty_report():
    loop = BrainstormLoop(FakeEvaluator({}), FakeProposer({}), FakeFrontier, max_iters=3)
    report = loop.run("seed", [])
    assert report.iterations == 0
    assert report.directions == ()
    assert report.n_assessed == 0


def test_report_carries_config():
    loop = BrainstormLoop(
        FakeEvaluator({"d0": ENTAILED}), FakeProposer({}), FakeFrontier, max_iters=4, batch_size=2
    )
    report = loop.run("my seed", ["d0"])
    assert (report.seed, report.max_iters, report.batch_size) == ("my seed", 4, 2)
