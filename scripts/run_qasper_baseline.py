"""QASPER dev-set baseline runner.

Loads QASPER dev, runs flat retrieval + LLM generation for each question,
streams per-question results to a JSONL sidecar, and writes a final
summary JSON with mean recall@k, mean answer F1, and skip counts.

Resumable: on re-run, already-evaluated question_ids are loaded from the
JSONL and skipped.

Memory dance: stop Grobid before running (needs Ollama in RAM).
    docker compose stop grobid
    python scripts/run_qasper_baseline.py --limit 10   # smoke
    python scripts/run_qasper_baseline.py              # full dev
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.qasper_eval import (
    aggregate_results,
    evaluate_question,
    extract_gold_answers,
    extract_gold_evidence,
)
from src.llm.client import get_client
from src.retrieval.flat_index import FlatIndex

DEFAULT_DEV = Path("data/datasets/qasper/dev.json")
DEFAULT_INDEX = Path("data/index/flat")
DEFAULT_OUTPUT_DIR = Path("eval/results")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


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


def _load_done(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done = set()
    for line in jsonl_path.read_text().splitlines():
        if line.strip():
            try:
                done.add(json.loads(line)["question_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _in_corpus_arxiv_ids(index_dir: Path) -> set[str]:
    manifest = json.loads((index_dir / "manifest.json").read_text())
    return {aid for aid, m in manifest.items() if m.get("done")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N questions (smoke).")
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--dev-path", type=Path, default=DEFAULT_DEV)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--llm-provider", type=str, default=None)
    ap.add_argument("--llm-model", type=str, default=None,
                    help="Override LLM model name (e.g. llama3.1:8b).")
    ap.add_argument("--rebuild", action="store_true",
                    help="Wipe existing output and start over.")
    ap.add_argument("--run-name", type=str, default="week3_flat_baseline")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.run_name}.jsonl"
    summary_path = args.output_dir / f"{args.run_name}_summary.json"

    if args.rebuild:
        if jsonl_path.exists():
            jsonl_path.unlink()
        if summary_path.exists():
            summary_path.unlink()

    print(f"[qasper_baseline] loading dev split: {args.dev_path}")
    questions = _load_dev(args.dev_path)
    print(f"[qasper_baseline] total questions in dev: {len(questions)}")

    in_corpus = _in_corpus_arxiv_ids(args.index_dir)
    print(f"[qasper_baseline] papers in flat index: {len(in_corpus)}")

    done = _load_done(jsonl_path)
    if done:
        print(f"[qasper_baseline] resuming — {len(done)} questions already evaluated")

    flat_index = FlatIndex(args.index_dir)
    llm_kwargs = {}
    if args.llm_model:
        llm_kwargs["model"] = args.llm_model
    llm_client = get_client(args.llm_provider)
    if args.llm_model and hasattr(llm_client, "model"):
        llm_client.model = args.llm_model

    n_total = len(questions)
    n_skipped_no_corpus = 0
    n_done_before = len(done)
    evaluated: list[dict] = []
    processed = 0

    t0 = time.time()

    with jsonl_path.open("a") as jf:
        for q in questions:
            if args.limit is not None and processed >= args.limit:
                break
            qid = q["question_id"]
            if qid in done:
                continue
            if q["paper_id"] not in in_corpus:
                n_skipped_no_corpus += 1
                continue

            gold_answers = extract_gold_answers(q["answers"])
            gold_evidence = extract_gold_evidence(q["answers"])

            result = evaluate_question(
                question=q["question"],
                question_id=qid,
                paper_id=q["paper_id"],
                gold_answers=gold_answers,
                gold_evidence=gold_evidence,
                flat_index=flat_index,
                llm_client=llm_client,
                k=args.k,
            )
            jf.write(json.dumps(result) + "\n")
            jf.flush()
            evaluated.append(result)
            processed += 1

            if processed % 10 == 0:
                agg = aggregate_results(evaluated)
                r = agg["mean_recall_at_k"]
                f1 = agg["mean_answer_f1"]
                r_str = f"{r:.3f}" if r is not None else "—"
                f1_str = f"{f1:.3f}" if f1 is not None else "—"
                print(f"  [{processed}] running R@{args.k}={r_str} F1={f1_str}")

    # Re-load everything we've written to compute the final summary
    # (includes prior resumed rows).
    all_results: list[dict] = []
    for line in jsonl_path.read_text().splitlines():
        if line.strip():
            all_results.append(json.loads(line))
    agg = aggregate_results(all_results)

    summary = {
        "run_name": args.run_name,
        "git_commit": _git_commit(),
        "n_total_dev_questions": n_total,
        "n_evaluated": agg["n_evaluated"],
        "n_with_evidence": agg["n_with_evidence"],
        "n_skipped_no_corpus_this_run": n_skipped_no_corpus,
        "n_done_before_this_run": n_done_before,
        "k": args.k,
        "mean_recall_at_k": agg["mean_recall_at_k"],
        "mean_answer_f1": agg["mean_answer_f1"],
        "runtime_sec_this_run": round(time.time() - t0, 1),
        "llm_provider": args.llm_provider or "ollama",
        "llm_model": getattr(llm_client, "model", None),
        "index_dir": str(args.index_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[qasper_baseline] summary -> {summary_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
