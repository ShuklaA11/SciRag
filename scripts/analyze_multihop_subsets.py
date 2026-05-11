"""Post-hoc breakdown of a multihop eval run by gate-decision subset.

Splits a `week7_multihop.jsonl` into atomic-gate-skipped vs LLM-decomposed
questions and reports recall@k per subset, plus a comparison against a
baseline run on the same questions (default: week6_rerank_k10).

Usage:
  PYTHONPATH=. python scripts/analyze_multihop_subsets.py \\
    --multihop eval/results/week7_multihop.jsonl \\
    --baseline eval/results/week6_rerank_k10.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from src.retrieval.decomposer import _looks_compound  # noqa


def _load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["question_id"]] = row
    return out


def _stats(rows: list[dict], baseline: dict[str, dict]) -> dict:
    fuzzy = [r["recall_at_k"] for r in rows if r.get("recall_at_k") is not None]
    strict = [r["recall_at_k_strict"] for r in rows if r.get("recall_at_k_strict") is not None]
    base_fuzzy: list[float] = []
    base_strict: list[float] = []
    for r in rows:
        b = baseline.get(r["question_id"])
        if not b:
            continue
        if b.get("recall_at_k") is not None:
            base_fuzzy.append(b["recall_at_k"])
        if b.get("recall_at_k_strict") is not None:
            base_strict.append(b["recall_at_k_strict"])
    return {
        "n": len(rows),
        "n_with_baseline": len(base_fuzzy),
        "fuzzy": round(mean(fuzzy), 4) if fuzzy else None,
        "strict": round(mean(strict), 4) if strict else None,
        "fuzzy_baseline": round(mean(base_fuzzy), 4) if base_fuzzy else None,
        "strict_baseline": round(mean(base_strict), 4) if base_strict else None,
        "fuzzy_delta": (
            round(mean(fuzzy) - mean(base_fuzzy), 4)
            if fuzzy and base_fuzzy else None
        ),
        "strict_delta": (
            round(mean(strict) - mean(base_strict), 4)
            if strict and base_strict else None
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--multihop", type=Path, required=True)
    p.add_argument(
        "--baseline", type=Path,
        default=Path("eval/results/week6_rerank_k10.jsonl"),
    )
    args = p.parse_args()

    mh_rows = list(_load(args.multihop).values())
    base_by_qid = _load(args.baseline)

    atomic: list[dict] = []
    compound: list[dict] = []
    for r in mh_rows:
        if _looks_compound(r["question"]):
            compound.append(r)
        else:
            atomic.append(r)

    report = {
        "multihop_run": str(args.multihop),
        "baseline_run": str(args.baseline),
        "overall": _stats(mh_rows, base_by_qid),
        "atomic_gate_skipped": _stats(atomic, base_by_qid),
        "compound_llm_decomposed": _stats(compound, base_by_qid),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
