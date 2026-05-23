"""SciFact zero-shot NLI baseline (Week 9, SB9.1).

Loads the SciFact dev split, materialises (claim, cited_doc) pairs with
the full abstract as oracle premise, runs the zero-shot NLI classifier,
and writes ``eval/results/<run_name>.jsonl`` + ``<run_name>_summary.json``.

Oracle evidence isolates classifier quality from retrieval quality. The
PLAN.md Week 9 target is ~72% zero-shot label accuracy, +8-12pp after
SB9.3 fine-tuning. SB9.2 swaps the oracle premise for BM25-retrieved
sentences over the full corpus.

Usage:
    # smoke run on 20 pairs (~30s on M-series MPS once the model is cached)
    python scripts/run_scifact_baseline.py --limit 20 --run-name week9_scifact_smoke

    # full dev (340 pairs, a few minutes)
    python scripts/run_scifact_baseline.py --run-name week9_scifact_zeroshot
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.scifact_eval import (
    build_pairs,
    evaluate,
    load_claims,
    load_corpus,
    write_results,
)
from src.verification.nli_classifier import DEFAULT_MODEL, NLIClassifier

DEFAULT_CLAIMS = Path("data/datasets/scifact/claims_dev.json")
DEFAULT_CORPUS = Path("data/datasets/scifact/corpus.json")
DEFAULT_OUTPUT_DIR = Path("eval/results")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _progress(done: int, total: int) -> None:
    pct = 100 * done / total if total else 0.0
    print(f"  {done}/{total} ({pct:.1f}%)", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument(
        "--nei-threshold",
        type=float,
        default=0.5,
        help="If max(P(SUP), P(CON)) < threshold, label as NEI",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, evaluate only the first N pairs (smoke test)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device override (mps/cuda/cpu); auto-detected by default",
    )
    args = p.parse_args()

    print(f"[load] corpus={args.corpus}")
    corpus = load_corpus(args.corpus)
    print(f"[load] {len(corpus):,} corpus docs")

    print(f"[load] claims={args.claims}")
    claims = load_claims(args.claims)
    print(f"[load] {len(claims):,} claims")

    pairs, n_missing = build_pairs(claims, corpus)
    print(f"[pairs] built {len(pairs):,} (claim, doc) pairs; dropped {n_missing} missing docs")
    if args.limit:
        pairs = pairs[: args.limit]
        print(f"[pairs] --limit {args.limit} -> evaluating {len(pairs)}")

    print(f"[model] loading {args.model}")
    clf = NLIClassifier(
        model_name=args.model,
        device=args.device,
        nei_threshold=args.nei_threshold,
        batch_size=args.batch_size,
    )
    print(f"[model] device={clf.device}, label_map={clf.label_map}")

    summary = evaluate(
        pairs,
        clf.predict_batch,
        batch_size=args.batch_size,
        progress=_progress,
    )

    rows_path, summary_path = write_results(
        args.output_dir,
        args.run_name,
        summary,
        extra_summary_fields={
            "git_commit": _git_commit(),
            "model_name": args.model,
            "device": clf.device,
            "nei_threshold": args.nei_threshold,
            "batch_size": args.batch_size,
            "claims_path": str(args.claims),
            "corpus_path": str(args.corpus),
            "limit": args.limit,
            "n_missing_docs": n_missing,
        },
    )

    print()
    print(f"[done] accuracy={summary['accuracy']:.4f}  macro_f1={summary['macro_f1']:.4f}")
    print(f"[done] per_class_f1={summary['per_class_f1']}")
    print(f"[done] gold_dist={summary['gold_label_dist']}")
    print(f"[done] pred_dist={summary['pred_label_dist']}")
    print(f"[done] runtime={summary['runtime_sec']}s")
    print(f"[done] wrote {rows_path}")
    print(f"[done] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
