"""Retrieval-only QASPER eval — no LLM, no answer F1.

Answers a single question: for a given embedder + flat index, what is the
mean recall@k of retrieved chunks against gold evidence on the QASPER dev
split? Skips the LLM entirely (~100x faster than the full baseline runner).

Per-question output mirrors the schema of run_qasper_baseline.py results
so scripts/rescore_results.py works on it unchanged.

Example:
  PYTHONPATH=. python scripts/retrieval_only_eval.py \\
    --embedder bge --index-dir data/index/flat_bge \\
    --run-name week4_bge_full_retrieval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.qasper_eval import (  # noqa: E402
    DEFAULT_MATCH_THRESHOLD,
    extract_gold_answers,
    extract_gold_evidence,
    recall_at_k,
)
from src.retrieval.flat_index import FlatIndex  # noqa: E402

DEFAULT_DEV = Path("data/datasets/qasper/dev.json")


def _load_dev(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    out = []
    for paper_id, paper in data.items():
        for q in paper.get("qas", []):
            out.append(
                {
                    "paper_id": paper_id,
                    "question_id": q["question_id"],
                    "question": q["question"],
                    "answers": q.get("answers", []),
                }
            )
    return out


def _in_corpus_arxiv_ids(index_dir: Path) -> set[str]:
    manifest = json.loads((index_dir / "manifest.json").read_text())
    return {aid for aid, m in manifest.items() if m.get("done")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedder", choices=["specter2", "bge"], default="bge")
    p.add_argument("--index-dir", type=Path, required=True)
    p.add_argument("--dev-path", type=Path, default=DEFAULT_DEV)
    p.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument(
        "--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD,
        help="Token-coverage threshold for fuzzy recall.",
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.run_name}.jsonl"
    summary_path = args.output_dir / f"{args.run_name}_summary.json"

    questions = _load_dev(args.dev_path)
    in_corpus = _in_corpus_arxiv_ids(args.index_dir)
    flat = FlatIndex(args.index_dir, embedder_name=args.embedder)

    fuzzy_vals: list[float] = []
    strict_vals: list[float] = []
    n_skipped_no_corpus = 0
    n_processed = 0

    t0 = time.time()
    with jsonl_path.open("w") as jf:
        for q in questions:
            if args.limit is not None and n_processed >= args.limit:
                break
            if q["paper_id"] not in in_corpus:
                n_skipped_no_corpus += 1
                continue

            gold_answers = extract_gold_answers(q["answers"])
            gold_evidence = extract_gold_evidence(q["answers"])
            retrieved = flat.search(q["question"], k=args.k, paper_ids={q["paper_id"]})
            texts = [r["text"] for r in retrieved]

            r_fuzzy = recall_at_k(texts, gold_evidence,
                                  strict=False, threshold=args.threshold)
            r_strict = recall_at_k(texts, gold_evidence, strict=True)

            row = {
                "question_id": q["question_id"],
                "paper_id": q["paper_id"],
                "question": q["question"],
                "gold_answers": gold_answers,
                "gold_evidence": gold_evidence,
                "retrieved_chunk_ids": [r["chunk_id"] for r in retrieved],
                "retrieved_arxiv_ids": [r["arxiv_id"] for r in retrieved],
                "recall_at_k": r_fuzzy,
                "recall_at_k_strict": r_strict,
                "match_mode": "fuzzy",
                "match_threshold": args.threshold,
                "metric_version": 2,
            }
            jf.write(json.dumps(row) + "\n")

            if r_fuzzy is not None:
                fuzzy_vals.append(r_fuzzy)
            if r_strict is not None:
                strict_vals.append(r_strict)
            n_processed += 1

            if n_processed % 50 == 0:
                m = mean(fuzzy_vals) if fuzzy_vals else 0.0
                print(f"  [{n_processed}] running R@{args.k}_fuzzy={m:.3f}")

    summary = {
        "run_name": args.run_name,
        "metric_version": 2,
        "match_mode": "fuzzy",
        "match_threshold": args.threshold,
        "k": args.k,
        "embedder": args.embedder,
        "index_dir": str(args.index_dir),
        "n_total_dev": len(questions),
        "n_processed": n_processed,
        "n_with_evidence": len(fuzzy_vals),
        "n_skipped_no_corpus": n_skipped_no_corpus,
        "mean_recall_at_k_fuzzy": mean(fuzzy_vals) if fuzzy_vals else None,
        "mean_recall_at_k_strict": mean(strict_vals) if strict_vals else None,
        "runtime_sec": round(time.time() - t0, 1),
        "results_file": str(jsonl_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[retrieval_only] summary -> {summary_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
