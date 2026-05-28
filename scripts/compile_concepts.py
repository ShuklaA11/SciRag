"""Compile concept articles for the SciRAG wiki.

Reads wiki/papers/*.md, ranks concept candidates via
src/wiki/concept_extractor, then asks the LLM to synthesize an article
per concept via src/wiki/concept_compiler. Output lands in
wiki/concepts/{slug}.md.

Two modes:

1) Candidate-list dump (no LLM):
     python scripts/compile_concepts.py --list-candidates --top-n 50
   Prints the ranked candidate list to stdout; pipe to a file or eyeball
   to pick the 15-20 that get articles.

2) Compile from a curated list:
     python scripts/compile_concepts.py --concepts-file concepts.txt
   Where concepts.txt has one concept name per line (matching the
   normalized form from the candidate list).

Per the SB10.3 quality gate: after the first 3 articles, halt and let
the operator review. Continue with --skip-gate or by re-running with
just the remaining concepts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.client import get_client
from src.wiki.concept_compiler import compile_concept
from src.wiki.concept_extractor import (
    ConceptEvidence,
    load_summaries_dir,
    rank_concepts,
)

DEFAULT_PAPERS_DIR = Path("wiki/papers")
DEFAULT_CONCEPTS_DIR = Path("wiki/concepts")
RUN_LOG = ".run_log.json"
QUALITY_GATE_AFTER = 3  # per SB10.3 plan


def _slugify(concept: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", concept.lower()).strip("_")
    return s or "untitled"


def _load_concept_list(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _build_evidence_index(
    papers_dir: Path,
    *,
    top_n: int,
    min_paper_count: int,
) -> dict[str, list[ConceptEvidence]]:
    summaries = load_summaries_dir(papers_dir)
    ranked = rank_concepts(summaries, top_n=top_n, min_paper_count=min_paper_count)
    return {concept: evidence for concept, _count, evidence in ranked}


def _cmd_list_candidates(args: argparse.Namespace) -> None:
    summaries = load_summaries_dir(args.papers_dir)
    ranked = rank_concepts(summaries, top_n=args.top_n, min_paper_count=args.min_paper_count)
    print(f"# {len(summaries)} summaries, {len(ranked)} candidates "
          f"(top_n={args.top_n}, min_paper_count={args.min_paper_count})")
    print("# count  concept")
    for concept, count, _evs in ranked:
        print(f"{count:>7}  {concept}")


def _cmd_compile(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    concept_names = _load_concept_list(args.concepts_file)
    if not concept_names:
        print("[compile_concepts] empty concept list; aborting")
        sys.exit(1)

    evidence_index = _build_evidence_index(
        args.papers_dir,
        top_n=max(args.top_n, len(concept_names) * 5),
        min_paper_count=args.min_paper_count,
    )

    client = get_client(args.llm_provider)
    if args.llm_model and hasattr(client, "model"):
        client.model = args.llm_model
    model_name = getattr(client, "model", args.llm_provider or "unknown")

    print(f"[compile_concepts] {len(concept_names)} concepts -> {args.output_dir}")
    print(f"[compile_concepts] model: {model_name}")

    run_log: list[dict] = []
    n_ok = n_parse_error = n_empty = n_skipped = n_missing = 0
    new_compiled = 0
    t0 = time.time()

    for i, concept in enumerate(concept_names, start=1):
        slug = _slugify(concept)
        md_path = args.output_dir / f"{slug}.md"
        if md_path.exists() and not args.rebuild:
            print(f"  [skip] {concept}")
            n_skipped += 1
            run_log.append({"concept": concept, "slug": slug, "status": "skipped"})
            continue

        evidence = evidence_index.get(concept, [])
        if not evidence:
            print(f"  [miss] {concept}  (not in candidate ranking; check spelling)")
            n_missing += 1
            run_log.append({"concept": concept, "slug": slug, "status": "missing_evidence"})
            continue

        print(f"  [..] {concept}  ({len(evidence)} papers)", flush=True)
        result = compile_concept(concept, evidence, client, model_name=model_name)
        md_path.write_text(result.markdown)

        if result.status == "ok":
            n_ok += 1
        elif result.status == "parse_error":
            n_parse_error += 1
        elif result.status == "empty_evidence":
            n_empty += 1
        new_compiled += 1

        print(f"  [{result.status}] {concept}  {result.latency_ms}ms")
        run_log.append({
            "concept": concept,
            "slug": slug,
            "status": result.status,
            "latency_ms": result.latency_ms,
            "n_evidence": len(evidence),
            "raw_output_preview": result.raw_output,
        })

        if new_compiled == QUALITY_GATE_AFTER and not args.skip_gate:
            elapsed = time.time() - t0
            print(f"\n[compile_concepts] quality gate: compiled {QUALITY_GATE_AFTER} concepts.")
            print(f"  review wiki/concepts/*.md before proceeding.")
            print(f"  re-run with --skip-gate or with the remaining concepts to continue.")
            print(f"  elapsed: {elapsed:.1f}s")
            _write_run_log(args.output_dir, run_log, model_name, elapsed,
                           n_ok, n_parse_error, n_empty, n_skipped, n_missing)
            sys.exit(0)

    elapsed = time.time() - t0
    _write_run_log(args.output_dir, run_log, model_name, elapsed,
                   n_ok, n_parse_error, n_empty, n_skipped, n_missing)

    print(f"\n[compile_concepts] done in {elapsed:.1f}s")
    print(f"  ok:          {n_ok}")
    print(f"  parse_error: {n_parse_error}")
    print(f"  empty:       {n_empty}")
    print(f"  skipped:     {n_skipped}")
    print(f"  missing:     {n_missing}")


def _write_run_log(
    output_dir: Path,
    run_log: list[dict],
    model_name: str,
    elapsed: float,
    n_ok: int,
    n_parse_error: int,
    n_empty: int,
    n_skipped: int,
    n_missing: int,
) -> None:
    summary = {
        "model": model_name,
        "elapsed_sec": round(elapsed, 1),
        "n_ok": n_ok,
        "n_parse_error": n_parse_error,
        "n_empty": n_empty,
        "n_skipped": n_skipped,
        "n_missing": n_missing,
        "entries": run_log,
    }
    (output_dir / RUN_LOG).write_text(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_CONCEPTS_DIR)
    ap.add_argument("--top-n", type=int, default=50,
                    help="How many candidates to rank/show.")
    ap.add_argument("--min-paper-count", type=int, default=2,
                    help="Drop concepts mentioned in fewer than N papers.")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-candidates", action="store_true",
                      help="Dump the ranked candidate list and exit.")
    mode.add_argument("--concepts-file", type=Path,
                      help="One concept name per line; compile each via LLM.")

    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--skip-gate", action="store_true",
                    help="Skip the quality-gate halt after the first 3 concepts.")
    ap.add_argument("--llm-provider", type=str, default=None)
    ap.add_argument("--llm-model", type=str, default=None)
    args = ap.parse_args()

    if args.list_candidates:
        _cmd_list_candidates(args)
    else:
        _cmd_compile(args)


if __name__ == "__main__":
    main()
