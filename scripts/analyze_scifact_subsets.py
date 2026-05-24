"""Diagnostic slices on the Week 9 SciFact eval JSONLs (no model load).

Reads the rows written by ``run_scifact_baseline.py`` and
``run_scifact_with_retrieval.py`` and prints three breakdowns that are
not in the headline summaries:

  1. Per-class retrieval recall@k -- is BM25 systematically worse at
     finding CONTRADICT- vs SUPPORT- vs NEI-annotated docs?
  2. Confusion matrix on the k=5 hit subset -- where does NLI fail
     once retrieval has done its job?
  3. SB9.1 oracle accuracy restricted to the k=5 hit subset -- the
     clean apples-to-apples comparison vs SB9.2 hit-only accuracy.

Pure stdout. Nothing is written.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from src.verification.nli_classifier import CONTRADICT, NEI, SCIFACT_LABELS, SUPPORT

ORACLE_PATH = Path("eval/results/week9_scifact_zeroshot.jsonl")
K_PATHS = {
    1: Path("eval/results/week9_scifact_bm25_k1.jsonl"),
    3: Path("eval/results/week9_scifact_bm25_k3.jsonl"),
    5: Path("eval/results/week9_scifact_bm25_k5.jsonl"),
    10: Path("eval/results/week9_scifact_bm25_k10.jsonl"),
}


def _load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _print_per_class_retrieval_recall(k_to_rows: dict[int, list[dict]]) -> None:
    print("=" * 72)
    print("(1) Per-class retrieval recall@k -- fraction of gold pairs with retrieved=True")
    print("=" * 72)
    header = f"{'gold':<12}" + "".join(f"k={k:>3}    " for k in k_to_rows)
    print(header)
    # gold totals are k-invariant; grab from any run
    any_rows = next(iter(k_to_rows.values()))
    gold_counts = Counter(r["gold_label"] for r in any_rows)
    for label in (SUPPORT, CONTRADICT, NEI):
        total = gold_counts.get(label, 0)
        cells = []
        for k, rows in k_to_rows.items():
            hits = sum(1 for r in rows if r["gold_label"] == label and r["retrieved"])
            rate = hits / total if total else 0.0
            cells.append(f"{rate:.3f} ({hits}/{total})")
        # Pad with spaces to keep columns aligned
        cell_str = "".join(f"{c:<10}" for c in cells)
        print(f"{label:<12}{cell_str}")
    # Overall row
    overall_cells = []
    total = sum(gold_counts.values())
    for k, rows in k_to_rows.items():
        hits = sum(1 for r in rows if r["retrieved"])
        rate = hits / total if total else 0.0
        overall_cells.append(f"{rate:.3f} ({hits}/{total})")
    print(f"{'overall':<12}" + "".join(f"{c:<10}" for c in overall_cells))


def _print_confusion_matrix(rows: Iterable[dict], title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    rows = list(rows)
    n = len(rows)
    if n == 0:
        print("  (no rows)")
        return

    by_pair = defaultdict(int)
    for r in rows:
        by_pair[(r["gold_label"], r["pred_label"])] += 1

    # Header
    print(f"{'gold \\ pred':<14}" + "".join(f"{p:<12}" for p in SCIFACT_LABELS) + f"{'total':<8}")
    for gold in SCIFACT_LABELS:
        row_total = sum(by_pair[(gold, p)] for p in SCIFACT_LABELS)
        cells = []
        for p in SCIFACT_LABELS:
            count = by_pair[(gold, p)]
            pct = (100 * count / row_total) if row_total else 0.0
            cells.append(f"{count:>3} ({pct:4.1f}%)")
        print(f"{gold:<14}" + "".join(f"{c:<12}" for c in cells) + f"{row_total:<8}")
    correct = sum(by_pair[(g, g)] for g in SCIFACT_LABELS)
    print(f"\n  n={n}  accuracy={correct / n:.4f}  correct={correct}/{n}")


def _print_oracle_on_hit_subset(
    oracle_rows: list[dict], k5_rows: list[dict]
) -> None:
    """Restrict SB9.1 oracle predictions to the (claim_id, doc_id) keys
    that BM25 successfully retrieved at k=5, then compute accuracy.

    This is the apples-to-apples comparison vs SB9.2's hit-only accuracy
    on the same hit subset.
    """
    hit_keys = {(r["claim_id"], r["doc_id"]) for r in k5_rows if r["retrieved"]}
    oracle_by_key = {(r["claim_id"], r["doc_id"]): r for r in oracle_rows}

    matched: list[dict] = []
    for key in hit_keys:
        if key in oracle_by_key:
            matched.append(oracle_by_key[key])

    print()
    print("=" * 72)
    print("(3) Oracle (SB9.1) accuracy restricted to the k=5 hit subset")
    print("=" * 72)
    if not matched:
        print("  (no overlap; cannot compare)")
        return
    correct = sum(1 for r in matched if r["pred_label"] == r["gold_label"])
    acc = correct / len(matched)
    gold_dist = Counter(r["gold_label"] for r in matched)
    print(f"  hit subset size: {len(matched)} pairs")
    print(f"  oracle accuracy on hit subset:  {acc:.4f}")
    print(f"  oracle accuracy on full 340:    0.6912 (SB9.1 headline)")
    print(f"  gold-label dist on hit subset:  {dict(gold_dist)}")

    # Compare directly to SB9.2 k=5 hit-only accuracy
    k5_hit_rows = [r for r in k5_rows if r["retrieved"]]
    k5_hit_correct = sum(1 for r in k5_hit_rows if r["pred_label"] == r["gold_label"])
    k5_hit_acc = k5_hit_correct / len(k5_hit_rows) if k5_hit_rows else 0.0
    print()
    print(f"  -> SB9.2 hit-only accuracy at k=5:        {k5_hit_acc:.4f}")
    print(f"  -> SB9.1 oracle accuracy on same subset:  {acc:.4f}")
    delta = k5_hit_acc - acc
    sign = "+" if delta >= 0 else ""
    print(f"  -> delta (BM25 hit-only - oracle on subset): {sign}{delta:.4f}")
    if abs(delta) < 0.005:
        print("     interpretation: NLI quality is essentially identical on this subset;")
        print("     the BM25-retrieved abstract is functionally equivalent to oracle.")
    elif delta < 0:
        print("     interpretation: BM25 retrieves easier pairs on average; oracle does")
        print("     better on the same pairs because the gap isn't just 'which doc'.")
    else:
        print("     interpretation: BM25 surprisingly helps over oracle on this subset --")
        print("     check for confounds (e.g., wrong gold doc, retrieved doc more on-topic).")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle", type=Path, default=ORACLE_PATH)
    args = p.parse_args()

    oracle_rows = _load(args.oracle)
    k_to_rows = {k: _load(path) for k, path in K_PATHS.items()}

    _print_per_class_retrieval_recall(k_to_rows)
    _print_confusion_matrix(
        (r for r in k_to_rows[5] if r["retrieved"]),
        "(2) Confusion matrix on the k=5 hit subset (NLI failures given retrieval succeeded)",
    )
    _print_oracle_on_hit_subset(oracle_rows, k_to_rows[5])
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
