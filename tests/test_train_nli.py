"""Smoke + unit tests for scripts/train_nli.py.

Heavy tests (actually invoke the HF Trainer) are gated behind
``SCIRAG_RUN_HEAVY=1`` and use a 30s 1-epoch run on a tiny subset to
verify the training loop saves a checkpoint that NLIClassifier can
reload. The lighter tests cover stratified splitting and label-map
inversion without any model load.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_nli.py"

# Load the script as a module so we can unit-test internal helpers
# without invoking the CLI.
_spec = importlib.util.spec_from_file_location("train_nli", SCRIPT_PATH)
train_nli = importlib.util.module_from_spec(_spec)
sys.modules["train_nli"] = train_nli
_spec.loader.exec_module(train_nli)  # type: ignore[union-attr]

from src.evaluation.scifact_eval import EvalPair  # noqa: E402
from src.verification.nli_classifier import (  # noqa: E402
    CONTRADICT,
    NEI,
    SUPPORT,
)


def _pair(claim_id: int, gold: str) -> EvalPair:
    return EvalPair(
        claim_id=claim_id,
        claim=f"claim {claim_id}",
        doc_id=claim_id * 100,
        premise=f"premise for claim {claim_id}",
        gold_label=gold,
    )


# ---------------------------------------------------------------------------
# Label-map inversion
# ---------------------------------------------------------------------------


def test_invert_label_map_round_trips():
    # Same shape NLIClassifier reads from model.config.id2label
    label_map = {0: SUPPORT, 1: NEI, 2: CONTRADICT}
    inv = train_nli._invert_label_map(label_map)
    assert inv == {SUPPORT: 0, NEI: 1, CONTRADICT: 2}
    # Round-trip via dict comprehension
    assert {v: k for k, v in inv.items()} == label_map


def test_invert_label_map_handles_arbitrary_index_layout():
    # FEVER-ANLI ships CONTRADICTION at index 0
    label_map = {0: CONTRADICT, 1: NEI, 2: SUPPORT}
    assert train_nli._invert_label_map(label_map) == {
        CONTRADICT: 0,
        NEI: 1,
        SUPPORT: 2,
    }


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def test_stratified_split_keeps_all_classes_in_val():
    pairs = (
        [_pair(i, SUPPORT) for i in range(50)]
        + [_pair(i, CONTRADICT) for i in range(50, 75)]
        + [_pair(i, NEI) for i in range(75, 125)]
    )
    train, val = train_nli._stratified_split(pairs, val_frac=0.10, seed=0)
    val_labels = {p.gold_label for p in val}
    # All three classes survived the split
    assert val_labels == {SUPPORT, CONTRADICT, NEI}
    # Sizes are roughly right (10% of each class, rounded with min 1)
    assert len(train) + len(val) == len(pairs)
    # Disjoint
    train_ids = {p.claim_id for p in train}
    val_ids = {p.claim_id for p in val}
    assert train_ids.isdisjoint(val_ids)


def test_stratified_split_small_class_still_gets_at_least_one():
    """A class with only 2 examples at 10% val_frac should still
    contribute >=1 to val (the `max(1, ...)` floor)."""
    pairs = (
        [_pair(i, SUPPORT) for i in range(100)]
        + [_pair(101, CONTRADICT), _pair(102, CONTRADICT)]
        + [_pair(i, NEI) for i in range(200, 220)]
    )
    train, val = train_nli._stratified_split(pairs, val_frac=0.10, seed=1)
    val_labels = [p.gold_label for p in val]
    assert val_labels.count(CONTRADICT) >= 1


def test_stratified_split_is_seed_deterministic():
    pairs = [_pair(i, SUPPORT) for i in range(40)] + [
        _pair(i, CONTRADICT) for i in range(40, 60)
    ]
    a = train_nli._stratified_split(pairs, val_frac=0.10, seed=7)
    b = train_nli._stratified_split(pairs, val_frac=0.10, seed=7)
    assert [p.claim_id for p in a[1]] == [p.claim_id for p in b[1]]


# ---------------------------------------------------------------------------
# compute_metrics closure
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect_predictions():
    import numpy as np

    label_to_id = {SUPPORT: 0, NEI: 1, CONTRADICT: 2}
    fn = train_nli._compute_metrics(label_to_id)

    # Three examples, one per class, all correct
    @dataclass
    class _EP:
        predictions: object
        label_ids: object

    logits = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    labels = np.array([0, 1, 2])
    metrics = fn((logits, labels))
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["f1_contradict"] == pytest.approx(1.0)


def test_compute_metrics_handles_zero_predictions_for_class():
    import numpy as np

    label_to_id = {SUPPORT: 0, NEI: 1, CONTRADICT: 2}
    fn = train_nli._compute_metrics(label_to_id)

    # Model predicts everything as SUPPORT; CONTRADICT and NEI get F1=0
    logits = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    labels = np.array([0, 1, 2])
    metrics = fn((logits, labels))
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["f1_contradict"] == 0.0
    assert metrics["f1_nei"] == 0.0
    assert metrics["f1_support"] > 0.0


# ---------------------------------------------------------------------------
# Heavy end-to-end smoke (gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("SCIRAG_RUN_HEAVY") != "1",
    reason="Set SCIRAG_RUN_HEAVY=1 to run the train-loop smoke",
)
def test_train_loop_smoke_writes_loadable_checkpoint(tmp_path: Path):
    """1-epoch micro-run that proves the train loop produces a
    checkpoint NLIClassifier can reload and predict from."""
    import subprocess

    out_dir = tmp_path / "tiny_ckpt"
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--limit",
        "12",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--val-frac",
        "0.25",
        "--output-dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert (out_dir / "training_metadata.json").exists()

    # Reload via the production interface; verify a forward pass works.
    from src.verification.nli_classifier import NLIClassifier

    clf = NLIClassifier(model_name=str(out_dir))
    pred = clf.predict(claim="any claim", evidence="any evidence")
    assert pred.label in (SUPPORT, CONTRADICT, NEI)
