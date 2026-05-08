"""Train the Week 5 query router on QASPER section-type labels.

Reads ``data/router/{train,dev}.jsonl`` produced by
``scripts/build_router_training_data.py``, fits the configured model,
saves the artifact, and reports per-class P/R/F1 on dev.

Example:
  PYTHONPATH=. python scripts/train_router.py \\
    --model tfidf \\
    --train data/router/train.jsonl \\
    --dev data/router/dev.jsonl \\
    --out data/router/tfidf.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.metrics import classification_report, f1_score  # noqa: E402

from src.router.tfidf_classifier import SECTION_TYPES, TfidfRouter  # noqa: E402


def _load(path: Path, drop_empty: bool) -> tuple[list[str], list[list[str]]]:
    questions: list[str] = []
    labels: list[list[str]] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            sts = row.get("section_types") or []
            if drop_empty and not sts:
                continue
            questions.append(row["question"])
            labels.append(sts)
    return questions, labels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["tfidf"], default="tfidf")
    p.add_argument("--train", type=Path, default=Path("data/router/train.jsonl"))
    p.add_argument("--dev", type=Path, default=Path("data/router/dev.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/router/tfidf.joblib"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--top-n", type=int, default=2)
    args = p.parse_args()

    train_q, train_y = _load(args.train, drop_empty=True)
    dev_q, dev_y = _load(args.dev, drop_empty=True)
    print(f"[train_router] train={len(train_q)} (with-label only) "
          f"dev={len(dev_q)} (with-label only)")

    if args.model == "tfidf":
        router = TfidfRouter()
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(args.model)

    router.fit(train_q, train_y)
    router.save(args.out)
    print(f"[train_router] saved -> {args.out}")

    preds = router.predict(dev_q, threshold=args.threshold, top_n=args.top_n)
    classes = list(SECTION_TYPES)
    y_true = router.binarizer.transform(dev_y)
    y_pred = router.binarizer.transform([list(pp.labels) for pp in preds])

    print(f"\n[train_router] dev metrics (threshold={args.threshold}, "
          f"top_n={args.top_n})")
    print(classification_report(
        y_true, y_pred, target_names=classes, zero_division=0,
    ))

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    proba = router.predict_proba(dev_q)
    mean_prob_top1 = float(np.mean(np.max(proba, axis=1)))

    summary = {
        "model": args.model,
        "train_path": str(args.train),
        "dev_path": str(args.dev),
        "out": str(args.out),
        "threshold": args.threshold,
        "top_n": args.top_n,
        "n_train": len(train_q),
        "n_dev": len(dev_q),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        "mean_prob_top1": mean_prob_top1,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[train_router] macro_f1={macro_f1:.4f} "
          f"micro_f1={micro_f1:.4f} weighted_f1={weighted_f1:.4f}")
    print(f"[train_router] mean top-1 confidence on dev: {mean_prob_top1:.3f}")
    print(f"[train_router] summary -> {summary_path}")


if __name__ == "__main__":
    main()
