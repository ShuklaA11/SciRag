"""Build router training data from QASPER train + dev splits.

For each question, run the same oracle logic as
``scripts/retrieval_only_eval.py --mode section-oracle``: find chunks of
the question's paper whose token-overlap with any gold-evidence sentence
is >= threshold, then collect their ``section_type`` values.

Output: ``data/router/{train,dev}.jsonl`` with one record per question
that has at least one gold-evidence sentence and whose paper is present
in the section-aware FAISS index. Records that produce an empty section
set are still emitted (with ``section_types: []``) so downstream code can
decide how to treat them; counts are reported separately.

Example:
  PYTHONPATH=. python scripts/build_router_training_data.py \\
    --index-dir data/index/flat_bge_sectioned \\
    --output-dir data/router
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.qasper_eval import (  # noqa: E402
    DEFAULT_MATCH_THRESHOLD,
    extract_gold_evidence,
    token_set,
)
from src.retrieval.flat_index import FlatIndex  # noqa: E402

DEFAULT_TRAIN = Path("data/datasets/qasper/train.json")
DEFAULT_DEV = Path("data/datasets/qasper/dev.json")


def _load_split(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    out: list[dict] = []
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


def _oracle_section_types(
    paper_chunks: list[dict],
    gold_evidence: list[str],
    threshold: float,
) -> set[str]:
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
            if overlap >= threshold:
                st = c.get("section_type")
                if st:
                    found.add(st)
    return found


def _build_split(
    split_name: str,
    questions: list[dict],
    chunks_by_paper: dict[str, list[dict]],
    in_corpus: set[str],
    threshold: float,
    out_path: Path,
) -> dict:
    n_total = len(questions)
    n_skipped_no_corpus = 0
    n_skipped_no_evidence = 0
    n_emitted = 0
    n_empty_label = 0
    type_dist: Counter[str] = Counter()
    label_count_dist: Counter[int] = Counter()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for q in questions:
            if q["paper_id"] not in in_corpus:
                n_skipped_no_corpus += 1
                continue
            gold_evidence = extract_gold_evidence(q["answers"])
            if not gold_evidence:
                n_skipped_no_evidence += 1
                continue
            paper_chunks = chunks_by_paper.get(q["paper_id"], [])
            section_types = sorted(
                _oracle_section_types(paper_chunks, gold_evidence, threshold)
            )
            if not section_types:
                n_empty_label += 1
            else:
                for st in section_types:
                    type_dist[st] += 1
            label_count_dist[len(section_types)] += 1
            row = {
                "question_id": q["question_id"],
                "paper_id": q["paper_id"],
                "question": q["question"],
                "section_types": section_types,
            }
            f.write(json.dumps(row) + "\n")
            n_emitted += 1

    return {
        "split": split_name,
        "out_path": str(out_path),
        "n_total": n_total,
        "n_skipped_no_corpus": n_skipped_no_corpus,
        "n_skipped_no_evidence": n_skipped_no_evidence,
        "n_emitted": n_emitted,
        "n_empty_label": n_empty_label,
        "n_with_label": n_emitted - n_empty_label,
        "section_type_distribution": dict(
            sorted(type_dist.items(), key=lambda kv: -kv[1])
        ),
        "label_count_distribution": dict(sorted(label_count_dist.items())),
        "mean_labels_per_question": (
            sum(k * v for k, v in label_count_dist.items()) / n_emitted
            if n_emitted
            else 0.0
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-dir", type=Path, required=True)
    p.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--dev-path", type=Path, default=DEFAULT_DEV)
    p.add_argument("--output-dir", type=Path, default=Path("data/router"))
    p.add_argument("--embedder", choices=["specter2", "bge"], default="bge")
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help="Token-coverage threshold for oracle 'this chunk contains gold' check.",
    )
    args = p.parse_args()

    in_corpus = _in_corpus_arxiv_ids(args.index_dir)
    print(f"[router-data] section index covers {len(in_corpus)} papers")

    flat = FlatIndex(args.index_dir, embedder_name=args.embedder)
    chunks_by_paper: dict[str, list[dict]] = {}
    for c in flat.chunks:
        chunks_by_paper.setdefault(c["arxiv_id"], []).append(c)
    print(f"[router-data] loaded {len(flat.chunks)} chunks "
          f"across {len(chunks_by_paper)} papers")

    summaries = []
    for split_name, in_path in [
        ("train", args.train_path),
        ("dev", args.dev_path),
    ]:
        questions = _load_split(in_path)
        out_path = args.output_dir / f"{split_name}.jsonl"
        print(f"\n[router-data] building {split_name} from {in_path} "
              f"({len(questions)} questions)")
        summary = _build_split(
            split_name=split_name,
            questions=questions,
            chunks_by_paper=chunks_by_paper,
            in_corpus=in_corpus,
            threshold=args.threshold,
            out_path=out_path,
        )
        summaries.append(summary)
        for k, v in summary.items():
            print(f"  {k}: {v}")

    summary_path = args.output_dir / "build_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "index_dir": str(args.index_dir),
                "threshold": args.threshold,
                "splits": summaries,
            },
            indent=2,
        )
    )
    print(f"\n[router-data] summary -> {summary_path}")


if __name__ == "__main__":
    main()
