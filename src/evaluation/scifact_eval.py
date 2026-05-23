"""SciFact eval harness — abstract-level label accuracy.

Eval unit is a single ``(claim, cited_doc)`` pair. Each pair carries a
gold label in {SUPPORT, CONTRADICT, NEI} derived from the SciFact
annotation:

  * If ``claim.evidence[str(doc_id)]`` contains entries, the label is
    the entry label (SciFact annotators never produced within-doc label
    conflicts on dev/train as of this writing).
  * If the doc is in ``cited_doc_ids`` but absent from ``evidence``, the
    claim has no annotated entailment relation to that doc -> NEI.

This is the zero-shot baseline framing: oracle evidence (the cited
abstract) is fed straight to NLI, isolating classifier quality from
retrieval quality. SB9.2 will replace the oracle premise with a BM25
top-k sentence selection over the full corpus.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from src.verification.nli_classifier import CONTRADICT, NEI, SCIFACT_LABELS, SUPPORT


@dataclass(frozen=True)
class EvalPair:
    claim_id: int
    claim: str
    doc_id: int
    premise: str
    gold_label: str


def load_corpus(corpus_path: str | Path) -> dict[int, dict[str, Any]]:
    """Load SciFact corpus.json -> {doc_id: {title, abstract_sentences}}."""
    with open(corpus_path) as f:
        rows = json.load(f)
    return {
        int(row["doc_id"]): {
            "title": row.get("title", ""),
            "abstract": list(row.get("abstract", [])),
        }
        for row in rows
    }


def load_claims(claims_path: str | Path) -> list[dict[str, Any]]:
    with open(claims_path) as f:
        return json.load(f)


def gold_label_for(claim: dict[str, Any], doc_id: int) -> str:
    """Read SciFact gold label for one (claim, doc) pair."""
    entries = claim.get("evidence", {}).get(str(doc_id), [])
    if not entries:
        return NEI
    labels = {e["label"] for e in entries}
    if labels == {"SUPPORT"}:
        return SUPPORT
    if labels == {"CONTRADICT"}:
        return CONTRADICT
    # Mixed labels within one doc are unexpected on SciFact; treat as NEI
    # and let the caller see it in the eval log.
    return NEI


def build_pairs(
    claims: Iterable[dict[str, Any]],
    corpus: dict[int, dict[str, Any]],
) -> tuple[list[EvalPair], int]:
    """Materialise (claim, doc) pairs with premise text from the full abstract.

    Returns (pairs, n_missing_docs). Missing docs are dropped — they
    can't be scored without a premise.
    """
    pairs: list[EvalPair] = []
    n_missing = 0
    for claim in claims:
        for doc_id in claim["cited_doc_ids"]:
            doc = corpus.get(int(doc_id))
            if doc is None:
                n_missing += 1
                continue
            premise = " ".join(doc["abstract"]).strip()
            if not premise:
                # Title-only docs are degenerate; fall back to title.
                premise = doc["title"]
            pairs.append(
                EvalPair(
                    claim_id=int(claim["id"]),
                    claim=claim["claim"],
                    doc_id=int(doc_id),
                    premise=premise,
                    gold_label=gold_label_for(claim, int(doc_id)),
                )
            )
    return pairs, n_missing


def _per_class_f1(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Macro per-class F1 over SciFact's 3 labels (no sklearn dep)."""
    out: dict[str, float] = {}
    for label in SCIFACT_LABELS:
        tp = sum(1 for r in rows if r["pred_label"] == label and r["gold_label"] == label)
        fp = sum(1 for r in rows if r["pred_label"] == label and r["gold_label"] != label)
        fn = sum(1 for r in rows if r["pred_label"] != label and r["gold_label"] == label)
        if tp + fp == 0 or tp + fn == 0:
            out[label] = 0.0
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        out[label] = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return out


# A "predict_batch" callable: list[(claim, premise)] -> list of objects
# with .label / .support_prob / .contradict_prob / .nei_prob attributes.
PredictBatchFn = Callable[[list[tuple[str, str]]], list[Any]]


def evaluate(
    pairs: list[EvalPair],
    predict_batch: PredictBatchFn,
    *,
    batch_size: int = 32,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run NLI over all eval pairs and aggregate metrics.

    ``predict_batch`` is decoupled from ``NLIClassifier`` so the harness
    can be tested without loading transformers and so SB9.3 can plug in
    the fine-tuned model with no harness changes.
    """
    started = time.time()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        preds = predict_batch([(p.claim, p.premise) for p in batch])
        if len(preds) != len(batch):
            raise RuntimeError(
                f"predict_batch returned {len(preds)} preds for {len(batch)} pairs"
            )
        for pair, pred in zip(batch, preds):
            rows.append(
                {
                    "claim_id": pair.claim_id,
                    "doc_id": pair.doc_id,
                    "gold_label": pair.gold_label,
                    "pred_label": pred.label,
                    "support_prob": float(pred.support_prob),
                    "contradict_prob": float(pred.contradict_prob),
                    "nei_prob": float(pred.nei_prob),
                }
            )
        if progress is not None:
            progress(min(start + batch_size, len(pairs)), len(pairs))

    n = len(rows)
    n_correct = sum(1 for r in rows if r["pred_label"] == r["gold_label"])
    gold_dist = Counter(r["gold_label"] for r in rows)
    pred_dist = Counter(r["pred_label"] for r in rows)
    per_class_f1 = _per_class_f1(rows)
    macro_f1 = sum(per_class_f1.values()) / len(per_class_f1) if per_class_f1 else 0.0

    return {
        "n_pairs": n,
        "accuracy": n_correct / n if n else 0.0,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "gold_label_dist": dict(gold_dist),
        "pred_label_dist": dict(pred_dist),
        "runtime_sec": round(time.time() - started, 2),
        "rows": rows,
    }


def write_results(
    out_dir: str | Path,
    run_name: str,
    summary: dict[str, Any],
    *,
    extra_summary_fields: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Persist results: <run>.jsonl (one row per pair) + <run>_summary.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{run_name}.jsonl"
    summary_path = out_dir / f"{run_name}_summary.json"

    rows = summary["rows"]
    with open(rows_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary_lite = {k: v for k, v in summary.items() if k != "rows"}
    summary_lite["run_name"] = run_name
    summary_lite["results_file"] = str(rows_path)
    if extra_summary_fields:
        summary_lite.update(extra_summary_fields)
    with open(summary_path, "w") as f:
        json.dump(summary_lite, f, indent=2)

    return rows_path, summary_path
