"""SciRAG end-to-end demo — narrated walkthrough (v2 W13).

Tells the whole story in five stages: the benchmarked retrieval spine, claim
verification, a LIVE idea-novelty evaluation, the temporal-novelty result, and
pointers to the interactive surfaces. Reads frozen results for the benchmarked
stages; runs the real engine for the live stage.

Usage:
    python scripts/demo.py                       # full, incl. live eval (needs Ollama + models)
    python scripts/demo.py --idea "<your idea>"  # customise the live stage
    python scripts/demo.py --no-live             # skip model loading (fast dry run)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MANIFEST = Path("eval/baseline_v2.json")
NOVELTY = Path("eval/results/novelty_temporal_qasper.json")
BUCKET_ICON = {"ENTAILED": "🟢", "CONTRADICTED": "🔴", "NOVEL": "🟡"}


def _rule(title: str) -> None:
    print(f"\n{'─' * 70}\n  {title}\n{'─' * 70}")


def _stage_retrieval() -> None:
    _rule("1 · Benchmarked retrieval spine  (QASPER dev, fuzzy recall@5, n=925)")
    cells = json.loads(MANIFEST.read_text())["retrieval_cells"]
    labels = {
        "A_flat_baseline": "A  flat baseline",
        "B_section_router": "B  + section routing",
        "C_citation_expand_rerank": "C  + citation expand + rerank",
        "D_multihop": "D  + multi-hop",
        "E_full_system": "E  full system (B+C+D)",
    }
    for key, label in labels.items():
        r = cells[key]["mean_recall_at_k_fuzzy"]
        print(f"    {label:<34} {r:.3f}")
    print("\n  Finding: E (0.692) < C (0.771) and D (0.772) — stacking HURTS.")
    print("  Citation expansion dilutes in-paper recall@5; measured, not assumed.")


def _stage_verification() -> None:
    _rule("2 · Claim verification  (SciFact, zero-shot DeBERTa-v3 → SUPPORT/CONTRADICT/NEI)")
    v = json.loads(MANIFEST.read_text())["verification"]["scifact_zeroshot"]
    print(f"    accuracy {v['accuracy']:.3f}   macro-F1 {v['macro_f1']:.3f}   (n={v['n_pairs']} pairs)")
    print("  Fine-tune deferred (CUDA-only; NaNs on MPS) — documented, not hidden.")


def _stage_idea_eval(idea: str, live: bool) -> None:
    _rule("3 · Idea-novelty evaluation  (the differentiator — LIVE)")
    print(f"    idea: “{idea}”\n")
    if not live:
        print("    [--no-live] skipping model load. This stage decomposes the idea into")
        print("    atomic claims, retrieves corpus evidence, runs NLI, and buckets each")
        print("    claim ENTAILED / CONTRADICTED / NOVEL — per-claim, never a scalar.")
        return

    print("    loading FlatIndex(bge) + NLI + Ollama decomposer ...", flush=True)
    from src.ideas import ClaimDecomposer, Evidence, IdeaEvaluator
    from src.llm.client import OllamaProvider
    from src.retrieval.flat_index import FlatIndex
    from src.verification.nli_classifier import DEFAULT_MODEL, NLIClassifier

    index = FlatIndex(Path("data/index/flat_bge_tagged"), embedder_name="bge")

    class _Retriever:
        def retrieve(self, query: str, k: int) -> list[Evidence]:
            return [Evidence(c["arxiv_id"], c["text"], c["score"]) for c in index.search(query, k=k)]

    evaluator = IdeaEvaluator(
        ClaimDecomposer(OllamaProvider()), _Retriever(), NLIClassifier(), k=5, model=DEFAULT_MODEL
    )
    report = evaluator.evaluate(idea)
    print(f"\n    {len(report.verdicts)} atomic claim(s):\n")
    for v in report.verdicts:
        icon = BUCKET_ICON.get(v.bucket, "")
        ev = v.top_evidence.ref if v.top_evidence else "—"
        print(f"    {icon} [{v.bucket}] {v.claim}")
        print(f"        support={v.best_support:.2f} contradict={v.best_contradict:.2f} evidence={ev}")


def _stage_novelty() -> None:
    _rule("4 · Temporal-novelty validation  (the wedge)")
    m = json.loads(NOVELTY.read_text())["metrics"]
    ic, ho = m["in_corpus"], m["held_out"]
    print(f"    cutoff {m['cutoff_year']}:")
    print(f"      in-corpus (<= Y)  NOVEL rate {ic['novel_rate']:.3f}  (n={ic['n']})")
    print(f"      held-out  (Y+1)   NOVEL rate {ho['novel_rate']:.3f}  (n={ho['n']})")
    print(f"      novelty_gap = +{m['novelty_gap']:.3f}  — held-out papers score more novel.")
    print("  Directional novelty proxy, validated by an arXiv-year holdout.")


def _stage_interactive() -> None:
    _rule("5 · Interactive surfaces")
    print("    Hub UI    :  streamlit run app/streamlit_app.py")
    print("                 (create project → Evaluate idea → color-coded verdicts, persisted)")
    print("    Brainstorm:  python scripts/run_brainstorm.py --seed \"<idea>\"")
    print("                 (agentic loop: retrieve → assess → gaps → propose → dedup)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", default="contrastive pretraining improves low-resource NER and reduces annotation cost")
    ap.add_argument("--no-live", action="store_true", help="skip the model-loading live stage")
    args = ap.parse_args()

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║   SciRAG — benchmarked scientific RAG + auditable idea evaluator  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    _stage_retrieval()
    _stage_verification()
    _stage_idea_eval(args.idea, live=not args.no_live)
    _stage_novelty()
    _stage_interactive()
    print()


if __name__ == "__main__":
    main()
