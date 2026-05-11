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
    token_set,
)
from src.retrieval.flat_index import FlatIndex  # noqa: E402
from src.router.tfidf_classifier import TfidfRouter  # noqa: E402

# CrossEncoderReranker is imported lazily inside main() to avoid loading
# sentence-transformers and torch when --rerank is not used.

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
    p.add_argument(
        "--mode", choices=["flat", "section-oracle", "section-router"],
        default="flat",
        help="flat: no section restriction. section-oracle: per-question, "
             "restrict retrieval to the section_types that contain a gold "
             "evidence sentence (best-case routing upper bound). "
             "section-router: use a trained classifier to predict section "
             "types from the question text; falls back to flat when the "
             "predicted set is empty or contains 'other'.",
    )
    p.add_argument(
        "--match-threshold-oracle", type=float, default=DEFAULT_MATCH_THRESHOLD,
        help="Token-coverage threshold for oracle 'this chunk contains gold' check.",
    )
    p.add_argument(
        "--router-path", type=Path, default=Path("data/router/tfidf.joblib"),
        help="Path to a saved TfidfRouter (only used when mode=section-router).",
    )
    p.add_argument(
        "--router-threshold", type=float, default=0.5,
        help="Per-class probability threshold for router inclusion.",
    )
    p.add_argument(
        "--router-top-n", type=int, default=2,
        help="Top-N classes by probability to always include (union with threshold).",
    )
    p.add_argument(
        "--router-other-fallback", action="store_true", default=True,
        help="If 'other' is in the predicted set, drop section restriction.",
    )
    p.add_argument(
        "--rerank", action="store_true",
        help="Apply cross-encoder reranking to retrieved candidates.",
    )
    p.add_argument(
        "--rerank-model", type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model name (sentence-transformers).",
    )
    p.add_argument(
        "--retrieve-k", type=int, default=20,
        help="Number of bi-encoder candidates to fetch before reranking. "
             "Ignored when --rerank is off.",
    )
    p.add_argument(
        "--expand-citations", action="store_true",
        help="Expand retrieval to in-corpus 1-hop citation neighbors.",
    )
    p.add_argument(
        "--multihop", action="store_true",
        help="Run multi-hop retrieval: LLM decomposes the question into "
             "atomic sub-Qs, each retrieves+reranks, results merged and "
             "re-ranked vs original. Requires --rerank.",
    )
    p.add_argument(
        "--multihop-llm-provider", type=str, default=None,
        help="Override SCIRAG_LLM_PROVIDER for the decomposer only.",
    )
    p.add_argument(
        "--multihop-dedup-threshold", type=float, default=0.85,
        help="BGE cosine threshold for dropping near-duplicate sub-Qs.",
    )
    p.add_argument(
        "--multihop-max-subqs", type=int, default=4,
        help="Cap on sub-questions emitted by the decomposer.",
    )
    p.add_argument(
        "--citation-graph",
        type=Path, default=Path("data/citation_graph/graph.pickle"),
        help="Path to the citation-graph pickle.",
    )
    p.add_argument(
        "--citation-directions",
        choices=["out", "in", "both"], default="both",
        help="Edge direction for 1-hop expansion.",
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.run_name}.jsonl"
    summary_path = args.output_dir / f"{args.run_name}_summary.json"

    questions = _load_dev(args.dev_path)
    in_corpus = _in_corpus_arxiv_ids(args.index_dir)
    flat = FlatIndex(args.index_dir, embedder_name=args.embedder)

    # Per-paper chunk lookup, built for any mode that needs oracle labels.
    chunks_by_paper: dict[str, list[dict]] = {}
    if args.mode in ("section-oracle", "section-router"):
        for c in flat.chunks:
            chunks_by_paper.setdefault(c["arxiv_id"], []).append(c)

    router: TfidfRouter | None = None
    router_predictions: dict[str, dict] = {}
    if args.mode == "section-router":
        router = TfidfRouter.load(args.router_path)
        eligible = [q for q in questions if q["paper_id"] in in_corpus]
        if args.limit is not None:
            eligible = eligible[: args.limit]
        preds = router.predict(
            [q["question"] for q in eligible],
            threshold=args.router_threshold,
            top_n=args.router_top_n,
        )
        for q, pp in zip(eligible, preds):
            router_predictions[q["question_id"]] = {
                "labels": list(pp.labels),
                "probabilities": pp.probabilities,
            }
        print(f"[retrieval_only] router loaded from {args.router_path}; "
              f"predicted on {len(router_predictions)} questions")

    reranker = None
    if args.rerank:
        from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(model_name=args.rerank_model)
        print(f"[retrieval_only] reranker loaded: {args.rerank_model} "
              f"on {reranker.device}")

    multihop = None
    decompose_latency_ms: list[float] = []
    sub_question_counts: list[int] = []
    if args.multihop:
        if not args.rerank:
            raise SystemExit("--multihop requires --rerank")
        from src.llm.client import get_client
        from src.pipeline.bge_embedder import BGEEmbedder
        from src.retrieval.decomposer import QueryDecomposer
        from src.retrieval.multihop import MultiHopRetriever

        decomposer_embedder = (
            flat.embedder if args.embedder == "bge" else BGEEmbedder()
        )
        decomposer = QueryDecomposer(
            llm=get_client(args.multihop_llm_provider),
            embedder=decomposer_embedder,
            dedup_threshold=args.multihop_dedup_threshold,
            max_sub_questions=args.multihop_max_subqs,
        )
        multihop = MultiHopRetriever(
            decomposer=decomposer,
            flat_index=flat,
            reranker=reranker,
            retrieve_k=args.retrieve_k,
            top_k=args.k,
        )
        print(f"[retrieval_only] multihop enabled "
              f"(max_subqs={args.multihop_max_subqs}, "
              f"dedup={args.multihop_dedup_threshold})")

    expander = None
    if args.expand_citations:
        from src.retrieval.citation_expander import CitationExpander
        expander = CitationExpander(graph_path=args.citation_graph)
        print(f"[retrieval_only] citation graph loaded: "
              f"{len(expander.graph.nodes)} nodes, "
              f"{len(expander.in_corpus)} in-corpus")
    expansion_neighbor_counts: list[int] = []
    n_questions_with_neighbors = 0

    retrieve_latency_ms: list[float] = []
    rerank_latency_ms: list[float] = []
    fuzzy_vals: list[float] = []
    strict_vals: list[float] = []
    n_skipped_no_corpus = 0
    n_processed = 0
    n_oracle_routed = 0
    n_oracle_fallback = 0
    oracle_type_dist: dict[str, int] = {}
    n_router_routed = 0
    n_router_fallback_empty = 0
    n_router_fallback_other = 0
    iou_vals: list[float] = []
    router_type_dist: dict[str, int] = {}

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

            oracle_types: list[str] = []
            section_filter: set[str] | None = None
            router_labels: list[str] = []
            router_probs: dict[str, float] = {}
            router_status: str = ""

            if args.mode in ("section-oracle", "section-router"):
                paper_chunks = chunks_by_paper.get(q["paper_id"], [])
                found: set[str] = set()
                for sent in gold_evidence:
                    gold_tokens = token_set(sent)
                    if not gold_tokens:
                        continue
                    for c in paper_chunks:
                        ct = token_set(c["text"])
                        if not ct:
                            continue
                        overlap = len(gold_tokens & ct) / len(gold_tokens)
                        if overlap >= args.match_threshold_oracle:
                            st = c.get("section_type")
                            if st:
                                found.add(st)
                if found:
                    oracle_types = sorted(found)

            if args.mode == "section-oracle":
                if oracle_types:
                    section_filter = set(oracle_types)
                    n_oracle_routed += 1
                    for st in oracle_types:
                        oracle_type_dist[st] = oracle_type_dist.get(st, 0) + 1
                else:
                    n_oracle_fallback += 1

            if args.mode == "section-router":
                pred = router_predictions.get(q["question_id"])
                if pred is not None:
                    router_labels = list(pred["labels"])
                    router_probs = pred["probabilities"]
                if not router_labels:
                    section_filter = None
                    router_status = "fallback_empty"
                    n_router_fallback_empty += 1
                elif args.router_other_fallback and "other" in router_labels:
                    section_filter = None
                    router_status = "fallback_other"
                    n_router_fallback_other += 1
                else:
                    section_filter = set(router_labels)
                    router_status = "routed"
                    n_router_routed += 1
                    for st in router_labels:
                        router_type_dist[st] = router_type_dist.get(st, 0) + 1
                if oracle_types:
                    pred_set = set(router_labels)
                    gold_set = set(oracle_types)
                    union = pred_set | gold_set
                    if union:
                        iou_vals.append(
                            len(pred_set & gold_set) / len(union)
                        )

            t_q0 = time.time()
            fetch_k = args.retrieve_k if args.rerank else args.k
            paper_ids = {q["paper_id"]}
            expansion_neighbors: list[str] = []
            if expander is not None:
                nbrs = expander.neighbors(
                    q["paper_id"],
                    in_corpus_only=True,
                    directions=args.citation_directions,
                )
                expansion_neighbors = sorted(nbrs)
                paper_ids |= nbrs
            expansion_neighbor_counts.append(len(expansion_neighbors))
            if expansion_neighbors:
                n_questions_with_neighbors += 1

            sub_questions: list[str] = []
            ce_scores: list[float] = []
            if multihop is not None:
                retrieved, mh_meta = multihop.retrieve(
                    q["question"], paper_ids=paper_ids,
                    section_types=section_filter,
                )
                sub_questions = mh_meta["sub_questions"]
                decompose_latency_ms.append(mh_meta["decompose_ms"])
                sub_question_counts.append(mh_meta["n_sub_questions"])
                t_retrieve = mh_meta["retrieve_ms"] / 1000.0
                t_rerank = mh_meta["rerank_ms"] / 1000.0
            else:
                retrieved = flat.search(
                    q["question"], k=fetch_k,
                    paper_ids=paper_ids,
                    section_types=section_filter,
                )
                t_retrieve = time.time() - t_q0
                t_rerank = 0.0
                if args.rerank and retrieved:
                    t_r0 = time.time()
                    ranked = reranker.rerank(
                        q["question"], retrieved, top_k=args.k,
                    )
                    t_rerank = time.time() - t_r0
                    retrieved = [r.chunk for r in ranked]
                    ce_scores = [r.ce_score for r in ranked]
                else:
                    retrieved = retrieved[: args.k]

            texts = [r["text"] for r in retrieved]
            retrieve_latency_ms.append(round(t_retrieve * 1000, 2))
            rerank_latency_ms.append(round(t_rerank * 1000, 2))

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
                "retrieved_section_types": [r.get("section_type") for r in retrieved],
                "recall_at_k": r_fuzzy,
                "recall_at_k_strict": r_strict,
                "match_mode": "fuzzy",
                "match_threshold": args.threshold,
                "metric_version": 2,
                "mode": args.mode,
                "oracle_section_types": oracle_types,
                "router_section_types": router_labels,
                "router_probabilities": router_probs,
                "router_status": router_status,
                "rerank": args.rerank,
                "ce_scores": ce_scores,
                "retrieve_latency_ms": retrieve_latency_ms[-1],
                "rerank_latency_ms": rerank_latency_ms[-1],
                "expand_citations": args.expand_citations,
                "expansion_neighbors": expansion_neighbors,
                "expansion_neighbor_count": len(expansion_neighbors),
                "multihop": args.multihop,
                "sub_questions": sub_questions,
                "n_sub_questions": len(sub_questions),
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
        "mode": args.mode,
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
        "rerank": args.rerank,
    }
    if args.expand_citations:
        summary["citation_graph"] = str(args.citation_graph)
        summary["citation_directions"] = args.citation_directions
        summary["n_questions_with_neighbors"] = n_questions_with_neighbors
        summary["mean_neighbors_per_question"] = (
            sum(expansion_neighbor_counts) / len(expansion_neighbor_counts)
            if expansion_neighbor_counts else 0.0
        )
        summary["max_neighbors_per_question"] = (
            max(expansion_neighbor_counts) if expansion_neighbor_counts else 0
        )
    if args.rerank:
        def _q(xs: list[float], p: float) -> float:
            if not xs:
                return 0.0
            xs_sorted = sorted(xs)
            i = max(0, min(len(xs_sorted) - 1, int(round(p * (len(xs_sorted) - 1)))))
            return xs_sorted[i]
        summary["rerank_model"] = args.rerank_model
        summary["retrieve_k"] = args.retrieve_k
        summary["latency_ms"] = {
            "retrieve_p50": round(_q(retrieve_latency_ms, 0.5), 2),
            "retrieve_p95": round(_q(retrieve_latency_ms, 0.95), 2),
            "rerank_p50": round(_q(rerank_latency_ms, 0.5), 2),
            "rerank_p95": round(_q(rerank_latency_ms, 0.95), 2),
            "total_p50": round(_q(
                [r + c for r, c in zip(retrieve_latency_ms, rerank_latency_ms)], 0.5
            ), 2),
            "total_p95": round(_q(
                [r + c for r, c in zip(retrieve_latency_ms, rerank_latency_ms)], 0.95
            ), 2),
        }
    if args.multihop:
        summary["multihop"] = True
        summary["multihop_max_subqs"] = args.multihop_max_subqs
        summary["multihop_dedup_threshold"] = args.multihop_dedup_threshold
        summary["mean_sub_questions"] = (
            sum(sub_question_counts) / len(sub_question_counts)
            if sub_question_counts else 0.0
        )
        summary["sub_question_count_distribution"] = {
            n: sub_question_counts.count(n)
            for n in sorted(set(sub_question_counts))
        }
        # _q is defined inside the rerank branch above, which --multihop
        # requires; safe to reuse here.
        summary["latency_ms"]["decompose_p50"] = round(
            _q(decompose_latency_ms, 0.5), 2,
        )
        summary["latency_ms"]["decompose_p95"] = round(
            _q(decompose_latency_ms, 0.95), 2,
        )
    if args.mode == "section-oracle":
        summary["match_threshold_oracle"] = args.match_threshold_oracle
        summary["n_oracle_routed"] = n_oracle_routed
        summary["n_oracle_fallback_no_match"] = n_oracle_fallback
        summary["oracle_section_types_distribution"] = dict(
            sorted(oracle_type_dist.items(), key=lambda kv: -kv[1])
        )
    if args.mode == "section-router":
        summary["router_path"] = str(args.router_path)
        summary["router_threshold"] = args.router_threshold
        summary["router_top_n"] = args.router_top_n
        summary["router_other_fallback"] = args.router_other_fallback
        summary["n_router_routed"] = n_router_routed
        summary["n_router_fallback_empty"] = n_router_fallback_empty
        summary["n_router_fallback_other"] = n_router_fallback_other
        summary["mean_iou_router_vs_oracle"] = (
            mean(iou_vals) if iou_vals else None
        )
        summary["router_section_types_distribution"] = dict(
            sorted(router_type_dist.items(), key=lambda kv: -kv[1])
        )
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[retrieval_only] summary -> {summary_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
