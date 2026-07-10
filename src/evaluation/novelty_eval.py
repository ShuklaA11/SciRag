"""Temporal-novelty eval harness (v3 Phase C, SB-C5).

The empirical wedge: if the idea-evaluation engine's NOVEL bucket tracks real
novelty at all, then contributions already *in* the corpus should score low
novelty, and contributions from the *next* year (not yet in the corpus) should
score higher. We measure that gap.

Substrate: QASPER papers, whose arXiv ids encode the year (``YYMM.NNNNN`` →
``2000+YY``). Split papers at a cutoff year Y: the corpus is built from papers
``≤Y``; test contributions are papers ``≤Y`` (in-corpus) vs ``Y+1`` (held-out).

PROXY CAVEAT (on record): "a paper's claim entailed by the earlier corpus" is a
*proxy* for novelty, not ground truth. Related prior work can entail a genuinely
novel contribution, and NLI entailment ≠ semantic novelty. The validated signal
is the *directional gap* (held-out novelty rate > in-corpus), not a per-paper
novelty oracle. Reported honestly as a proxy.

This module is the model-free harness: year parsing, the temporal split, and
rate aggregation over an injected evaluator. The real run (real NLI over the
QASPER split) lives in a run script.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from src.ideas import BUCKETS, NOVEL


def arxiv_year(paper_id: str) -> int:
    """Year from a new-scheme arXiv id (``1912.01214`` → 2019).

    All QASPER papers use the post-2007 ``YYMM.NNNNN`` scheme, so ``2000+YY``.
    Old-scheme / malformed ids (``cs/0501001``) fail loudly rather than being
    silently mis-dated.
    """
    head = paper_id.split(".", 1)[0]
    if len(head) < 2 or not head[:2].isdigit():
        raise ValueError(f"cannot derive arXiv year from paper id {paper_id!r}")
    return 2000 + int(head[:2])


def temporal_split(
    paper_ids: Iterable[str], cutoff_year: int
) -> tuple[list[str], list[str]]:
    """Split ids into (in-corpus ``≤Y``, held-out ``Y+1``).

    Papers after ``Y+1`` are dropped — the eval contrasts adjacent years to
    keep distribution shift minimal (the pre-registered design).
    """
    in_corpus: list[str] = []
    held_out: list[str] = []
    for pid in paper_ids:
        year = arxiv_year(pid)
        if year <= cutoff_year:
            in_corpus.append(pid)
        elif year == cutoff_year + 1:
            held_out.append(pid)
    return in_corpus, held_out


class _EvaluatorLike(Protocol):
    def evaluate_claims(self, claims: list[str], *, idea: str = ...) -> Any: ...


def _bucket_counts(verdicts: Iterable[Any]) -> dict[str, int]:
    counts = {b: 0 for b in BUCKETS}
    for v in verdicts:
        counts[v.bucket] += 1
    return counts


def _group_metrics(evaluator: _EvaluatorLike, claims: Iterable[str]) -> dict[str, Any]:
    verdicts = evaluator.evaluate_claims(list(claims)).verdicts
    counts = _bucket_counts(verdicts)
    n = len(verdicts)
    return {
        "n": n,
        "novel_rate": counts[NOVEL] / n if n else 0.0,
        "buckets": counts,
    }


def novelty_rates(
    evaluator: _EvaluatorLike,
    in_corpus_claims: Iterable[str],
    held_out_claims: Iterable[str],
    *,
    cutoff_year: int | None = None,
) -> dict[str, Any]:
    """NOVEL-bucket rate for each temporal group and the gap between them.

    Validation direction: ``novelty_gap > 0`` (held-out novelty exceeds
    in-corpus). Bucket distributions are reported per group for transparency.
    """
    in_corpus = _group_metrics(evaluator, in_corpus_claims)
    held_out = _group_metrics(evaluator, held_out_claims)
    return {
        "cutoff_year": cutoff_year,
        "in_corpus": in_corpus,
        "held_out": held_out,
        "novelty_gap": held_out["novel_rate"] - in_corpus["novel_rate"],
    }
