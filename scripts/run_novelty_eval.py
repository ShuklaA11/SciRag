"""Temporal-novelty eval — the real run (v3 Phase C, SB-C5 run).

Wires real components into the tested harness (``src.evaluation.novelty_eval``):

    corpus   = abstracts of QASPER papers with year <= cutoff (BM25 evidence pool)
    claim    = each paper's TITLE (one claim/paper → LLM-free evaluate_claims)
    engine   = IdeaEvaluator(BM25→Evidence adapter, zero-shot DeBERTa NLI)
    metric   = NOVEL-bucket rate for in-corpus (<=Y) vs held-out (Y+1), and the gap

Prediction: in-corpus titles retrieve their own abstract in the corpus → SUPPORT
→ low novelty; held-out (Y+1) titles are absent → NEI → higher novelty, so
novelty_gap > 0. This is a novelty *proxy* (see the harness docstring), reported
as a directional signal.

Usage:
    python scripts/run_novelty_eval.py [--cutoff 2018] [--k 5] [--limit N]
Needs the QASPER dev split on disk + the zero-shot NLI model (auto-downloaded).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.novelty_eval import arxiv_year, novelty_rates, temporal_split
from src.hub import current_git_commit
from src.ideas import Evidence, IdeaEvaluator
from src.verification.evidence_retriever import BM25EvidenceRetriever, _doc_text
from src.verification.nli_classifier import DEFAULT_MODEL, NLIClassifier

QASPER_DEV = Path("data/datasets/qasper/dev.json")
OUT_DIR = Path("eval/results")


class _BM25EvidenceAdapter:
    """Adapts BM25EvidenceRetriever (doc_id/score) to the engine's retriever
    protocol (Evidence carrying the text NLI reads)."""

    def __init__(self, corpus: dict[int, dict[str, Any]]) -> None:
        self._bm25 = BM25EvidenceRetriever(corpus)
        self._corpus = corpus

    def retrieve(self, query: str, k: int) -> list[Evidence]:
        return [
            Evidence(ref=d.doc_id, text=_doc_text(self._corpus[d.doc_id]), score=d.score)
            for d in self._bm25.retrieve(query, k=k)
        ]


class _UnusedDecomposer:
    def decompose(self, idea: str) -> list[str]:  # pragma: no cover - never called
        raise RuntimeError("novelty eval uses evaluate_claims; decomposer is unused")


def _load_papers(limit: int | None) -> list[tuple[str, str, str, int]]:
    """Return (paper_id, title, abstract, year) for dated dev papers."""
    papers = json.loads(QASPER_DEV.read_text())
    rows = []
    for pid, p in papers.items():
        title = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()
        if not title or not abstract:
            continue
        rows.append((pid, title, abstract, arxiv_year(pid)))
    rows.sort(key=lambda r: r[0])
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=2018, help="cutoff year Y")
    ap.add_argument("--k", type=int, default=5, help="evidence docs per claim")
    ap.add_argument("--limit", type=int, default=None, help="cap papers (smoke runs)")
    args = ap.parse_args()

    rows = _load_papers(args.limit)
    in_ids, held_ids = temporal_split((r[0] for r in rows), args.cutoff)
    in_set, held_set = set(in_ids), set(held_ids)

    # Corpus = abstracts of in-corpus (<=Y) papers; scifact-shaped for BM25.
    corpus_rows = [r for r in rows if r[0] in in_set]
    corpus = {i: {"title": t, "abstract": [a]} for i, (_pid, t, a, _y) in enumerate(corpus_rows)}
    in_claims = [t for _pid, t, _a, _y in corpus_rows]
    held_claims = [t for pid, t, _a, _y in rows if pid in held_set]

    print(f"papers={len(rows)}  corpus(<= {args.cutoff})={len(corpus)}  "
          f"in-corpus claims={len(in_claims)}  held-out({args.cutoff + 1})={len(held_claims)}")
    print(f"loading NLI model {DEFAULT_MODEL} ...", flush=True)

    evaluator = IdeaEvaluator(
        _UnusedDecomposer(),
        _BM25EvidenceAdapter(corpus),
        NLIClassifier(),
        k=args.k,
        model=DEFAULT_MODEL,
    )

    print("scoring in-corpus and held-out claims ...", flush=True)
    metrics = novelty_rates(evaluator, in_claims, held_claims, cutoff_year=args.cutoff)

    result = {
        "run_name": "novelty_temporal_qasper",
        "cutoff_year": args.cutoff,
        "k": args.k,
        "nli_model": DEFAULT_MODEL,
        "claim_unit": "paper_title",
        "corpus": "qasper_dev_abstracts_le_cutoff",
        "git_commit": current_git_commit(),
        "metrics": metrics,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "novelty_temporal_qasper.json"
    out_path.write_text(json.dumps(result, indent=2))

    ic, ho = metrics["in_corpus"], metrics["held_out"]
    print(f"\nin-corpus  novel_rate = {ic['novel_rate']:.3f}  (n={ic['n']}, {ic['buckets']})")
    print(f"held-out   novel_rate = {ho['novel_rate']:.3f}  (n={ho['n']}, {ho['buckets']})")
    print(f"novelty_gap = {metrics['novelty_gap']:+.3f}  →  wrote {out_path}")


if __name__ == "__main__":
    main()
