"""Regression guard for the frozen v2 baseline (eval/baseline_v2.json).

Re-aggregates each resolved cell's stored per-query jsonl and asserts the
mean matches the manifest. This locks the ablation spine's aggregation/metric
code during the v3 Phase A refactor. It does NOT re-run the pipeline (that
needs Docker + Ollama + a full re-eval) — only that the recorded per-query
results still aggregate to the frozen numbers.

Aggregation mirrors scripts/retrieval_only_eval.py: mean of non-None
recall_at_k / recall_at_k_strict over rows (rows with empty gold evidence
carry None and are skipped). SciFact accuracy = fraction of matching labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "eval" / "baseline_v2.json"
TOL = 1e-9


def _load_jsonl(rel_path: str) -> list[dict]:
    path = REPO_ROOT / rel_path
    return [json.loads(line) for line in path.open() if line.strip()]


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _resolved_retrieval_cells() -> list[tuple[str, dict]]:
    cells = _manifest()["retrieval_cells"]
    return [(name, c) for name, c in cells.items() if c.get("source_jsonl")]


def test_manifest_exists_and_parses():
    m = _manifest()
    assert m["manifest_version"] == 1
    assert m["metric"]["version"] == 2


@pytest.mark.parametrize("name,cell", _resolved_retrieval_cells())
def test_retrieval_cell_reaggregates(name, cell):
    rows = _load_jsonl(cell["source_jsonl"])
    fuzzy = [r["recall_at_k"] for r in rows if r.get("recall_at_k") is not None]
    strict = [r["recall_at_k_strict"] for r in rows if r.get("recall_at_k_strict") is not None]

    assert len(fuzzy) == cell["n_with_evidence"], (
        f"{name}: n_with_evidence drifted ({len(fuzzy)} vs {cell['n_with_evidence']})"
    )
    assert abs(mean(fuzzy) - cell["mean_recall_at_k_fuzzy"]) < TOL, f"{name}: fuzzy recall drifted"
    assert abs(mean(strict) - cell["mean_recall_at_k_strict"]) < TOL, f"{name}: strict recall drifted"


def test_scifact_zeroshot_reaggregates():
    cell = _manifest()["verification"]["scifact_zeroshot"]
    rows = _load_jsonl(cell["source_jsonl"])
    assert len(rows) == cell["n_pairs"]
    acc = mean([1.0 if r["gold_label"] == r["pred_label"] else 0.0 for r in rows])
    assert abs(acc - cell["accuracy"]) < TOL, "scifact zero-shot accuracy drifted"
