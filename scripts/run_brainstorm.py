"""Agentic brainstorm loop — real run (v3 Phase D, SB-D4).

Wires real components into the fake-tested brainstorm engine (src.brainstorm):

    seed idea → ClaimDecomposer (Ollama)         → seed directions
    per direction: FlatIndex(bge) retrieve
                   → CitationExpander 1-hop        → re-retrieve in neighborhood
                   → zero-shot DeBERTa NLI         → NOVEL/ENTAILED/CONTRADICTED
    NOVEL directions = gaps → DirectionProposer (Ollama) → next directions
    Frontier dedups (BGE cosine 0.85); loop capped at --max-iters.

Citation expansion lives here in the retriever adapter (not the loop) — its job
is cross-paper *discovery*, which is expansion's intended win (cf. config-E,
where it hurt in-paper recall). Qualitative demo, not a benchmark.

Usage: python scripts/run_brainstorm.py --seed "<idea>" [--max-iters 2] [--k 5]
Needs: flat_bge_tagged index, citation graph, NLI model, Ollama running.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.brainstorm import BrainstormLoop, DirectionProposer, Frontier
from src.hub import current_git_commit
from src.ideas import ClaimDecomposer, Evidence, IdeaEvaluator
from src.llm.client import OllamaProvider
from src.retrieval.citation_expander import CitationExpander
from src.retrieval.flat_index import FlatIndex
from src.verification.nli_classifier import DEFAULT_MODEL, NLIClassifier

INDEX_DIR = Path("data/index/flat_bge_tagged")
OUT_DIR = Path("eval/results")


class _CitationExpandingRetriever:
    """Engine retriever protocol over FlatIndex, widened by 1-hop citations.

    retrieve → collect hit papers → expand to in-corpus neighbors →
    re-retrieve within that neighborhood (always ⊇ the original papers)."""

    def __init__(self, index: FlatIndex, expander: CitationExpander) -> None:
        self.index = index
        self.expander = expander

    def retrieve(self, query: str, k: int) -> list[Evidence]:
        initial = self.index.search(query, k=k)
        papers = {c["arxiv_id"] for c in initial}
        if not papers:
            return []
        expanded = set(papers)
        for pid in papers:
            expanded |= self.expander.expanded_paper_ids(pid)
        final = self.index.search(query, k=k, paper_ids=expanded)
        return [Evidence(ref=c["arxiv_id"], text=c["text"], score=c["score"]) for c in final]


def _verdict_row(v) -> dict:
    return {
        "direction": v.claim,
        "bucket": v.bucket,
        "best_support": round(v.best_support, 3),
        "best_contradict": round(v.best_contradict, 3),
        "top_evidence": v.top_evidence.ref if v.top_evidence else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="contrastive pretraining improves low-resource NER")
    ap.add_argument("--max-iters", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    print("loading index / citation graph / NLI / Ollama ...", flush=True)
    index = FlatIndex(INDEX_DIR, embedder_name="bge")
    retriever = _CitationExpandingRetriever(index, CitationExpander())
    llm = OllamaProvider()

    evaluator = IdeaEvaluator(
        ClaimDecomposer(llm), retriever, NLIClassifier(), k=args.k, model=DEFAULT_MODEL
    )
    proposer = DirectionProposer(llm)
    loop = BrainstormLoop(
        evaluator,
        proposer,
        lambda: Frontier(index.embedder),
        max_iters=args.max_iters,
        batch_size=args.batch_size,
    )

    seed_directions = ClaimDecomposer(llm).decompose(args.seed)
    print(f"seed='{args.seed}'  seed_directions={len(seed_directions)}", flush=True)
    print("running brainstorm loop ...", flush=True)
    report = loop.run(args.seed, seed_directions)

    result = {
        "run_name": "brainstorm_demo",
        "seed": report.seed,
        "iterations": report.iterations,
        "n_assessed": report.n_assessed,
        "max_iters": report.max_iters,
        "batch_size": report.batch_size,
        "k": args.k,
        "nli_model": DEFAULT_MODEL,
        "git_commit": current_git_commit(),
        "discovered_directions": [_verdict_row(v) for v in report.directions],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "brainstorm_demo.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\niterations={report.iterations}  assessed={report.n_assessed}  "
          f"discovered gaps={len(report.directions)}")
    for row in result["discovered_directions"]:
        print(f"  [{row['bucket']}] {row['direction']}  (ev={row['top_evidence']})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
