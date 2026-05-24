"""SciFact end-to-end pipeline with BM25 evidence retrieval (Week 9, SB9.2).

Quantifies the cost of dropping SB9.1's oracle premise. For each claim:

  1. BM25-retrieve top-k docs from the full corpus (5,183 abstracts).
  2. For each gold ``(claim, cited_doc)`` pair (SB9.1's eval unit):
      * if ``cited_doc`` is in the top-k -> run NLI on the retrieved
        abstract and use that prediction;
      * else -> predict NEI ("we never saw the evidence, so we cannot
        say SUPPORT or CONTRADICT"). This is the honest abstention
        choice and matches how a deployed verifier would behave.

Two metrics are reported side by side:

  * **Retrieval recall@k** — fraction of gold cited_doc_ids that landed
    in the top-k. Cleanly diagnostic; isolates BM25 quality.
  * **End-to-end label accuracy** — same per-(claim, doc) accuracy as
    SB9.1, directly comparable to its 0.691 oracle anchor.

Usage:
    # smoke on 20 pairs at k=5
    python scripts/run_scifact_with_retrieval.py --limit 20 \
        --run-name week9_scifact_bm25_smoke

    # full dev at k=5
    python scripts/run_scifact_with_retrieval.py \
        --run-name week9_scifact_bm25_k5

    # sweep k -- writes one run per value
    python scripts/run_scifact_with_retrieval.py \
        --run-name week9_scifact_bm25 --k-sweep 1,3,5,10
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.scifact_eval import (
    build_pairs,
    load_claims,
    load_corpus,
    write_results,
    _per_class_f1,
)
from src.verification.evidence_retriever import BM25EvidenceRetriever
from src.verification.nli_classifier import (
    CONTRADICT,
    DEFAULT_MODEL,
    NEI,
    NLIClassifier,
    NLIPrediction,
    SUPPORT,
)

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


def _abstain_pred() -> NLIPrediction:
    """Prediction stand-in for 'gold doc not retrieved' — full NEI mass."""
    return NLIPrediction(label=NEI, support_prob=0.0, contradict_prob=0.0, nei_prob=1.0)


def _run_one_k(
    *,
    pairs,
    corpus: dict[int, dict[str, Any]],
    retriever: BM25EvidenceRetriever,
    clf: NLIClassifier,
    k: int,
    batch_size: int,
) -> dict[str, Any]:
    """Run the full pipeline for a single ``k`` value and aggregate metrics."""
    started = time.time()

    # 1. Per-claim BM25 retrieval. Dedup the claim text so we don't
    # re-run BM25 for every cited_doc on the same claim.
    unique_claims: dict[int, str] = {}
    for pair in pairs:
        unique_claims.setdefault(pair.claim_id, pair.claim)

    retrieve_started = time.time()
    retrieved: dict[int, list[int]] = {}
    for claim_id, claim_text in unique_claims.items():
        hits = retriever.retrieve(claim_text, k=k)
        retrieved[claim_id] = [h.doc_id for h in hits]
    retrieval_sec = time.time() - retrieve_started

    # 2. Retrieval-recall accounting (per gold pair).
    n_pairs = len(pairs)
    n_recall_hits = sum(
        1 for p in pairs if p.doc_id in retrieved[p.claim_id]
    )
    retrieval_recall_at_k = n_recall_hits / n_pairs if n_pairs else 0.0

    # 3. NLI only on hit pairs. Misses bypass the model and are scored
    # as NEI predictions (the abstain branch).
    hit_pairs = [p for p in pairs if p.doc_id in retrieved[p.claim_id]]
    miss_pairs = [p for p in pairs if p.doc_id not in retrieved[p.claim_id]]

    nli_started = time.time()
    hit_preds: list[NLIPrediction] = []
    if hit_pairs:
        for start in range(0, len(hit_pairs), batch_size):
            chunk = hit_pairs[start : start + batch_size]
            preds = clf.predict_batch([(p.claim, p.premise) for p in chunk])
            hit_preds.extend(preds)
    nli_sec = time.time() - nli_started

    # 4. Stitch rows back in pair order so the JSONL is reproducible.
    pred_by_pair_key: dict[tuple[int, int], NLIPrediction] = {
        (p.claim_id, p.doc_id): pred for p, pred in zip(hit_pairs, hit_preds)
    }
    abstain = _abstain_pred()

    rows: list[dict[str, Any]] = []
    for p in pairs:
        retrieved_ids = retrieved[p.claim_id]
        was_retrieved = p.doc_id in retrieved_ids
        pred = pred_by_pair_key[(p.claim_id, p.doc_id)] if was_retrieved else abstain
        rows.append(
            {
                "claim_id": p.claim_id,
                "doc_id": p.doc_id,
                "gold_label": p.gold_label,
                "pred_label": pred.label,
                "support_prob": float(pred.support_prob),
                "contradict_prob": float(pred.contradict_prob),
                "nei_prob": float(pred.nei_prob),
                "retrieved": was_retrieved,
                "retrieved_rank": retrieved_ids.index(p.doc_id) if was_retrieved else -1,
            }
        )

    n_correct = sum(1 for r in rows if r["pred_label"] == r["gold_label"])
    per_class_f1 = _per_class_f1(rows)
    macro_f1 = sum(per_class_f1.values()) / len(per_class_f1) if per_class_f1 else 0.0

    # Accuracy restricted to the hit subset — isolates NLI quality given
    # successful retrieval (apples-to-apples with SB9.1 oracle on the
    # subset of pairs where BM25 did its job).
    hit_rows = [r for r in rows if r["retrieved"]]
    hit_accuracy = (
        sum(1 for r in hit_rows if r["pred_label"] == r["gold_label"]) / len(hit_rows)
        if hit_rows
        else 0.0
    )

    return {
        "k": k,
        "n_pairs": n_pairs,
        "n_unique_claims": len(unique_claims),
        "retrieval_recall_at_k": retrieval_recall_at_k,
        "n_retrieval_hits": n_recall_hits,
        "n_retrieval_misses": n_pairs - n_recall_hits,
        "accuracy": n_correct / n_pairs if n_pairs else 0.0,
        "hit_accuracy": hit_accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "gold_label_dist": dict(Counter(r["gold_label"] for r in rows)),
        "pred_label_dist": dict(Counter(r["pred_label"] for r in rows)),
        "retrieval_sec": round(retrieval_sec, 2),
        "nli_sec": round(nli_sec, 2),
        "runtime_sec": round(time.time() - started, 2),
        "rows": rows,
    }


def _print_summary(name: str, s: dict[str, Any]) -> None:
    print(f"\n[done] {name}")
    print(f"  k={s['k']}  n_pairs={s['n_pairs']}  unique_claims={s['n_unique_claims']}")
    print(
        f"  retrieval_recall@{s['k']}={s['retrieval_recall_at_k']:.4f} "
        f"(hits={s['n_retrieval_hits']}, misses={s['n_retrieval_misses']})"
    )
    print(
        f"  end-to-end accuracy={s['accuracy']:.4f}  "
        f"hit-only accuracy={s['hit_accuracy']:.4f}  macro_f1={s['macro_f1']:.4f}"
    )
    print(f"  per_class_f1={s['per_class_f1']}")
    print(f"  pred_dist={s['pred_label_dist']}")
    print(f"  timings: retrieve={s['retrieval_sec']}s nli={s['nli_sec']}s total={s['runtime_sec']}s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--nei-threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--k", type=int, default=5)
    p.add_argument(
        "--k-sweep",
        type=str,
        default=None,
        help="Comma-separated list of k values; emits one run per value (suffix _kN).",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    ks: list[int]
    if args.k_sweep:
        ks = [int(x) for x in args.k_sweep.split(",") if x.strip()]
    else:
        ks = [args.k]

    print(f"[load] corpus={args.corpus}")
    corpus = load_corpus(args.corpus)
    print(f"[load] {len(corpus):,} corpus docs")

    print(f"[load] claims={args.claims}")
    claims = load_claims(args.claims)
    print(f"[load] {len(claims):,} claims")

    pairs, n_missing = build_pairs(claims, corpus)
    print(f"[pairs] {len(pairs):,} (claim, doc) pairs; dropped {n_missing} missing docs")
    if args.limit:
        pairs = pairs[: args.limit]
        print(f"[pairs] --limit {args.limit} -> {len(pairs)} pairs")

    print("[index] building BM25 over abstracts...")
    bm25_started = time.time()
    retriever = BM25EvidenceRetriever(corpus)
    print(f"[index] indexed {len(retriever)} docs in {time.time() - bm25_started:.1f}s")

    print(f"[model] loading {args.model}")
    clf = NLIClassifier(
        model_name=args.model,
        device=args.device,
        nei_threshold=args.nei_threshold,
        batch_size=args.batch_size,
    )
    print(f"[model] device={clf.device}")

    extra_fields = {
        "git_commit": _git_commit(),
        "model_name": args.model,
        "device": clf.device,
        "nei_threshold": args.nei_threshold,
        "batch_size": args.batch_size,
        "claims_path": str(args.claims),
        "corpus_path": str(args.corpus),
        "limit": args.limit,
        "n_missing_docs": n_missing,
        "n_corpus_docs": len(retriever),
        "retriever": "bm25_okapi_title_plus_abstract",
    }

    for k in ks:
        run_name = args.run_name if len(ks) == 1 else f"{args.run_name}_k{k}"
        print(f"\n[run] {run_name} (k={k})")
        summary = _run_one_k(
            pairs=pairs,
            corpus=corpus,
            retriever=retriever,
            clf=clf,
            k=k,
            batch_size=args.batch_size,
        )
        rows_path, summary_path = write_results(
            args.output_dir,
            run_name,
            summary,
            extra_summary_fields=extra_fields,
        )
        _print_summary(run_name, summary)
        print(f"  wrote {rows_path}")
        print(f"  wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
