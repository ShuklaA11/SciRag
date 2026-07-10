"""SB-C5: temporal-novelty eval harness. Model-free — a fake evaluator maps
claims to buckets, so the guard is the harness logic (year parsing, split,
rate/gap aggregation), not NLI. The real run produces the actual number."""

from __future__ import annotations

import pytest

from src.evaluation.novelty_eval import arxiv_year, novelty_rates, temporal_split
from src.ideas import ENTAILED, NOVEL


class _Verdict:
    def __init__(self, bucket: str) -> None:
        self.bucket = bucket


class _Report:
    def __init__(self, verdicts) -> None:
        self.verdicts = tuple(verdicts)


class FakeEvaluator:
    """evaluate_claims → buckets via a claim→bucket map (default ENTAILED)."""

    def __init__(self, buckets: dict[str, str]) -> None:
        self.buckets = buckets

    def evaluate_claims(self, claims, *, idea: str = "") -> _Report:
        return _Report(_Verdict(self.buckets.get(c, ENTAILED)) for c in claims)


# --- arxiv_year -------------------------------------------------------------


@pytest.mark.parametrize(
    "pid,year",
    [("1912.01214", 2019), ("1704.05119", 2017), ("2005.14165", 2020)],
)
def test_arxiv_year_parses_new_scheme(pid, year):
    assert arxiv_year(pid) == year


@pytest.mark.parametrize("pid", ["cs/0501001", "abc", "", "x9.1"])
def test_arxiv_year_rejects_malformed(pid):
    with pytest.raises(ValueError):
        arxiv_year(pid)


# --- temporal_split ---------------------------------------------------------


def test_temporal_split_groups_by_year():
    ids = ["1712.001", "1801.002", "1903.003", "1905.004", "2001.005"]
    in_corpus, held_out = temporal_split(ids, cutoff_year=2018)
    assert in_corpus == ["1712.001", "1801.002"]  # <= 2018
    assert held_out == ["1903.003", "1905.004"]  # == 2019
    # 2020 paper dropped (beyond Y+1)


# --- novelty_rates ----------------------------------------------------------


def test_novelty_gap_is_positive_when_held_out_scores_more_novel():
    ev = FakeEvaluator({"held1": NOVEL, "held2": NOVEL})  # in-corpus default ENTAILED
    out = novelty_rates(ev, ["in1", "in2"], ["held1", "held2"], cutoff_year=2018)

    assert out["in_corpus"]["novel_rate"] == 0.0
    assert out["held_out"]["novel_rate"] == 1.0
    assert out["novelty_gap"] == 1.0
    assert out["cutoff_year"] == 2018
    assert out["in_corpus"]["buckets"][ENTAILED] == 2


def test_novelty_rates_reports_partial_rates():
    ev = FakeEvaluator({"a": NOVEL, "c": NOVEL})  # b defaults ENTAILED
    out = novelty_rates(ev, ["a", "b"], ["c"])
    assert out["in_corpus"]["novel_rate"] == 0.5
    assert out["held_out"]["novel_rate"] == 1.0
    assert out["novelty_gap"] == 0.5


def test_empty_group_has_zero_rate_not_crash():
    out = novelty_rates(FakeEvaluator({}), [], [])
    assert out["in_corpus"]["n"] == 0
    assert out["in_corpus"]["novel_rate"] == 0.0
    assert out["novelty_gap"] == 0.0
