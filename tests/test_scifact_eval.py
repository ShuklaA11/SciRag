"""Tests for the SciFact eval harness.

No HF model load — the harness takes a ``predict_batch`` callable, so
tests pin behavior with a fake that returns scripted labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.evaluation.scifact_eval import (
    EvalPair,
    build_pairs,
    evaluate,
    gold_label_for,
    load_claims,
    load_corpus,
    write_results,
)
from src.verification.nli_classifier import CONTRADICT, NEI, SUPPORT


@dataclass
class _FakePred:
    label: str
    support_prob: float = 0.0
    contradict_prob: float = 0.0
    nei_prob: float = 0.0


# ---------------------------------------------------------------------------
# Gold label extraction
# ---------------------------------------------------------------------------


def test_gold_label_support():
    claim = {"id": 1, "evidence": {"42": [{"sentences": [0], "label": "SUPPORT"}]}}
    assert gold_label_for(claim, 42) == SUPPORT


def test_gold_label_contradict():
    claim = {
        "id": 1,
        "evidence": {"42": [{"sentences": [0], "label": "CONTRADICT"}]},
    }
    assert gold_label_for(claim, 42) == CONTRADICT


def test_gold_label_missing_doc_is_nei():
    """Doc cited but absent from evidence dict -> NEI."""
    claim = {"id": 1, "evidence": {}}
    assert gold_label_for(claim, 42) == NEI


def test_gold_label_multiple_entries_same_label():
    claim = {
        "id": 1,
        "evidence": {
            "42": [
                {"sentences": [0], "label": "SUPPORT"},
                {"sentences": [2, 3], "label": "SUPPORT"},
            ]
        },
    }
    assert gold_label_for(claim, 42) == SUPPORT


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------


def test_build_pairs_uses_full_abstract_as_premise():
    corpus = {
        7: {"title": "T", "abstract": ["sent1.", "sent2.", "sent3."]},
    }
    claims = [
        {
            "id": 1,
            "claim": "hypothesis",
            "evidence": {"7": [{"sentences": [0], "label": "SUPPORT"}]},
            "cited_doc_ids": [7],
        }
    ]
    pairs, missing = build_pairs(claims, corpus)
    assert missing == 0
    assert len(pairs) == 1
    assert pairs[0].premise == "sent1. sent2. sent3."
    assert pairs[0].gold_label == SUPPORT


def test_build_pairs_expands_multi_cited_docs():
    corpus = {
        1: {"title": "A", "abstract": ["a1."]},
        2: {"title": "B", "abstract": ["b1."]},
    }
    claims = [
        {
            "id": 99,
            "claim": "c",
            "evidence": {"1": [{"sentences": [0], "label": "CONTRADICT"}]},
            "cited_doc_ids": [1, 2],
        }
    ]
    pairs, missing = build_pairs(claims, corpus)
    assert missing == 0
    assert [(p.doc_id, p.gold_label) for p in pairs] == [(1, CONTRADICT), (2, NEI)]


def test_build_pairs_drops_missing_docs():
    corpus = {1: {"title": "A", "abstract": ["a1."]}}
    claims = [{"id": 1, "claim": "c", "evidence": {}, "cited_doc_ids": [1, 999]}]
    pairs, missing = build_pairs(claims, corpus)
    assert missing == 1
    assert {p.doc_id for p in pairs} == {1}


def test_build_pairs_falls_back_to_title_when_abstract_empty():
    corpus = {3: {"title": "title only", "abstract": []}}
    claims = [{"id": 1, "claim": "c", "evidence": {}, "cited_doc_ids": [3]}]
    pairs, _ = build_pairs(claims, corpus)
    assert pairs[0].premise == "title only"


# ---------------------------------------------------------------------------
# Evaluate aggregation
# ---------------------------------------------------------------------------


def _pair(claim_id: int, gold: str) -> EvalPair:
    return EvalPair(
        claim_id=claim_id,
        claim=f"claim {claim_id}",
        doc_id=claim_id * 10,
        premise="premise",
        gold_label=gold,
    )


def test_evaluate_perfect_predictions():
    pairs = [_pair(1, SUPPORT), _pair(2, CONTRADICT), _pair(3, NEI)]

    def predict_batch(pairs_in):
        # Echo the gold label for each pair via claim_id lookup
        return [_FakePred(label=p[0].split()[-1]) for p in pairs_in if False] or [
            _FakePred(label=SUPPORT),
            _FakePred(label=CONTRADICT),
            _FakePred(label=NEI),
        ]

    out = evaluate(pairs, predict_batch, batch_size=10)
    assert out["accuracy"] == 1.0
    assert out["n_pairs"] == 3
    assert out["macro_f1"] == pytest.approx(1.0)
    assert out["per_class_f1"] == {SUPPORT: 1.0, CONTRADICT: 1.0, NEI: 1.0}


def test_evaluate_all_wrong():
    pairs = [_pair(1, SUPPORT), _pair(2, CONTRADICT)]

    def predict_batch(_):
        return [_FakePred(label=NEI), _FakePred(label=NEI)]

    out = evaluate(pairs, predict_batch)
    assert out["accuracy"] == 0.0
    assert out["per_class_f1"][NEI] == 0.0  # no true positives


def test_evaluate_returns_per_row_data():
    pairs = [_pair(1, SUPPORT)]

    def predict_batch(_):
        return [_FakePred(label=SUPPORT, support_prob=0.8, contradict_prob=0.1, nei_prob=0.1)]

    out = evaluate(pairs, predict_batch)
    row = out["rows"][0]
    assert row["pred_label"] == SUPPORT
    assert row["gold_label"] == SUPPORT
    assert row["support_prob"] == pytest.approx(0.8)


def test_evaluate_batching_respected():
    pairs = [_pair(i, SUPPORT) for i in range(7)]
    calls: list[int] = []

    def predict_batch(p):
        calls.append(len(p))
        return [_FakePred(label=SUPPORT)] * len(p)

    evaluate(pairs, predict_batch, batch_size=3)
    assert calls == [3, 3, 1]


def test_evaluate_raises_on_pred_length_mismatch():
    pairs = [_pair(1, SUPPORT), _pair(2, SUPPORT)]

    def predict_batch(_):
        return [_FakePred(label=SUPPORT)]  # wrong length

    with pytest.raises(RuntimeError, match="returned 1 preds"):
        evaluate(pairs, predict_batch)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def test_load_corpus_and_claims_roundtrip(tmp_path: Path):
    corpus_path = tmp_path / "corpus.json"
    claims_path = tmp_path / "claims.json"
    corpus_path.write_text(
        json.dumps(
            [
                {"doc_id": 11, "title": "t1", "abstract": ["x1.", "x2."]},
                {"doc_id": 22, "title": "t2", "abstract": []},
            ]
        )
    )
    claims_path.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "claim": "c",
                    "evidence": {"11": [{"sentences": [0], "label": "SUPPORT"}]},
                    "cited_doc_ids": [11],
                }
            ]
        )
    )
    corpus = load_corpus(corpus_path)
    claims = load_claims(claims_path)
    assert corpus[11]["abstract"] == ["x1.", "x2."]
    assert claims[0]["claim"] == "c"


def test_write_results_emits_rows_and_summary(tmp_path: Path):
    summary = {
        "n_pairs": 1,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "per_class_f1": {SUPPORT: 1.0, CONTRADICT: 0.0, NEI: 0.0},
        "gold_label_dist": {SUPPORT: 1},
        "pred_label_dist": {SUPPORT: 1},
        "runtime_sec": 0.01,
        "rows": [
            {
                "claim_id": 1,
                "doc_id": 10,
                "gold_label": SUPPORT,
                "pred_label": SUPPORT,
                "support_prob": 0.9,
                "contradict_prob": 0.05,
                "nei_prob": 0.05,
            }
        ],
    }
    rows_path, summary_path = write_results(
        tmp_path, "test_run", summary, extra_summary_fields={"model_name": "stub"}
    )
    assert rows_path.read_text().strip().startswith('{"claim_id": 1')
    loaded = json.loads(summary_path.read_text())
    assert loaded["run_name"] == "test_run"
    assert loaded["model_name"] == "stub"
    assert "rows" not in loaded
