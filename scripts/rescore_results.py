"""Re-score an existing per-question results JSONL with a new metric.

Reads `retrieved_chunk_ids` from each row, looks up chunk texts from a
flat-index `chunks.jsonl`, and recomputes recall@k under the current
`recall_at_k` implementation in fuzzy and strict modes. Writes a sibling
`<name>.rescored.jsonl` plus a summary printed to stdout.

No retrieval, no LLM, no FAISS. Read-only on the index dir.

Example:
  python scripts/rescore_results.py \\
    --results eval/results/week3_flat_baseline_full.jsonl \\
    --chunks  data/index/flat/chunks.jsonl \\
    --threshold 0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from src.evaluation.qasper_eval import (
    DEFAULT_MATCH_THRESHOLD,
    recall_at_k,
)


METRIC_VERSION = 2


def load_chunk_texts(chunks_path: Path) -> dict[str, str]:
    """Load {chunk_id: text} from a flat-index chunks.jsonl."""
    out: dict[str, str] = {}
    with chunks_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["chunk_id"]] = row["text"]
    return out


def rescore_file(
    results_path: Path,
    chunks_path: Path,
    threshold: float,
    extra_thresholds: list[float] | None = None,
) -> dict:
    chunk_texts = load_chunk_texts(chunks_path)
    out_path = results_path.with_suffix(".rescored.jsonl")
    extra_thresholds = extra_thresholds or []

    fuzzy_vals: list[float] = []
    strict_vals: list[float] = []
    extra_vals: dict[float, list[float]] = {t: [] for t in extra_thresholds}
    n_skipped_missing = 0
    n_no_evidence = 0
    n_total = 0

    with results_path.open() as f_in, out_path.open("w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            row = json.loads(line)

            chunk_ids = row.get("retrieved_chunk_ids") or []
            try:
                texts = [chunk_texts[cid] for cid in chunk_ids]
            except KeyError:
                # Chunk id not in this index; skip without crashing.
                n_skipped_missing += 1
                f_out.write(line + "\n")
                continue

            gold = row.get("gold_evidence") or []
            r_fuzzy = recall_at_k(texts, gold, strict=False, threshold=threshold)
            r_strict = recall_at_k(texts, gold, strict=True)

            row["recall_at_k_strict"] = r_strict
            row["recall_at_k_fuzzy"] = r_fuzzy
            row["recall_at_k"] = r_fuzzy
            row["match_mode"] = "fuzzy"
            row["match_threshold"] = threshold
            row["metric_version"] = METRIC_VERSION

            if r_fuzzy is None:
                n_no_evidence += 1
            else:
                fuzzy_vals.append(r_fuzzy)
            if r_strict is not None:
                strict_vals.append(r_strict)

            for t in extra_thresholds:
                v = recall_at_k(texts, gold, strict=False, threshold=t)
                if v is not None:
                    extra_vals[t].append(v)

            f_out.write(json.dumps(row) + "\n")

    summary = {
        "results_file": str(results_path),
        "chunks_file": str(chunks_path),
        "metric_version": METRIC_VERSION,
        "match_mode": "fuzzy",
        "match_threshold": threshold,
        "n_total": n_total,
        "n_with_evidence": len(fuzzy_vals),
        "n_no_evidence": n_no_evidence,
        "n_skipped_missing_chunks": n_skipped_missing,
        "mean_recall_at_k_fuzzy": mean(fuzzy_vals) if fuzzy_vals else None,
        "mean_recall_at_k_strict": mean(strict_vals) if strict_vals else None,
        "rescored_file": str(out_path),
    }
    if extra_thresholds:
        summary["sensitivity"] = {
            f"T={t}": (mean(v) if v else None) for t, v in extra_vals.items()
        }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--chunks", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    p.add_argument(
        "--extra-thresholds",
        type=str,
        default="",
        help="Comma-separated extra T values to report (sensitivity strip).",
    )
    args = p.parse_args()

    extras = (
        [float(x) for x in args.extra_thresholds.split(",") if x.strip()]
        if args.extra_thresholds
        else []
    )

    summary = rescore_file(args.results, args.chunks, args.threshold, extras)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
