"""Compile paper summaries at scale for the SciRAG wiki.

Generalizes ``scripts/compile_first_10.py``. Selects a paper set
(``canonical``, ``qasper``, or ``all``), runs the LLM summarizer,
writes one markdown per paper to ``wiki/papers/``, and appends a
trend row to ``wiki/papers/.compile_history.csv``.

Existing summaries are skipped unless ``--rebuild`` is passed; this
makes the script safe to re-run for resumption after a crash or for
incremental top-up as new TEIs arrive.

Memory dance (same as compile_first_10.py): stop Grobid before
running, since Ollama + Grobid don't both fit in 16GB RAM.
    docker compose stop grobid
    ollama run llama3.1:8b
    python scripts/compile_papers.py --paper-set qasper --limit 50

For the Week 10 N=55 run:
    python scripts/compile_papers.py --paper-set all --limit 50
(produces 5 canonical + 50 alphabetical QASPER = 55 summaries)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.client import get_client
from src.wiki.incremental import (
    decide,
    flag_stale_concepts,
    load_state,
    save_state,
    update_state,
)
from src.wiki.summarizer import summarize_paper

CANONICAL = [
    "attention_is_all_you_need",
    "bert",
    "elmo",
    "gpt2",
    "scibert",
]
GROBID_ROOT = Path("data/grobid_output")
QASPER_DIR_NAME = "qasper"
DEFAULT_OUTPUT = Path("wiki/papers")
DEFAULT_CONCEPTS = Path("wiki/concepts")
DEFAULT_WIKI_ROOT = Path("wiki")
HISTORY_CSV = "compile_history.csv"
RUN_LOG = ".run_log.json"

SAMPLE_SEED = 42  # pre-registered in PLAN.md decision #5


def _pick_papers(
    grobid_root: Path,
    paper_set: str,
    limit: int | None,
) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if paper_set in ("canonical", "all"):
        for stem in CANONICAL:
            p = grobid_root / f"{stem}.xml"
            if p.exists():
                out.append((stem, p))
            else:
                print(f"  [warn] canonical paper missing: {p}")

    if paper_set in ("qasper", "all"):
        qasper_xmls = sorted((grobid_root / QASPER_DIR_NAME).glob("*.xml"))
        if limit is not None:
            qasper_xmls = qasper_xmls[:limit]
        for p in qasper_xmls:
            out.append((p.stem, p))

    return out


def _append_history(
    output_dir: Path,
    *,
    model: str,
    paper_set: str,
    n_papers: int,
    n_ok: int,
    n_parse_error: int,
    n_empty: int,
    n_skipped: int,
    elapsed_sec: float,
) -> None:
    history_path = output_dir / HISTORY_CSV
    write_header = not history_path.exists()
    with history_path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow([
                "timestamp",
                "model",
                "paper_set",
                "n_papers",
                "n_ok",
                "n_parse_error",
                "n_empty_tei",
                "n_skipped",
                "elapsed_sec",
            ])
        w.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model,
            paper_set,
            n_papers,
            n_ok,
            n_parse_error,
            n_empty,
            n_skipped,
            f"{elapsed_sec:.1f}",
        ])


def _sample_for_review(
    output_dir: Path,
    *,
    new_stems: list[str],
    k: int,
    seed: int,
) -> list[str]:
    """Pick k random stems from new_stems with a fixed seed; print them."""
    if not new_stems:
        return []
    k = min(k, len(new_stems))
    rng = random.Random(seed)
    sample = sorted(rng.sample(new_stems, k))
    print(f"\n[compile_papers] sampled {k}/{len(new_stems)} for review (seed={seed}):")
    for s in sample:
        print(f"  - wiki/papers/{s}.md")
    return sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--paper-set",
        choices=["canonical", "qasper", "all"],
        default="all",
        help="Which papers to compile.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of QASPER papers (alphabetical). "
        "Canonical 5 are always included for --paper-set=all.",
    )
    ap.add_argument("--grobid-root", type=Path, default=GROBID_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--rebuild", action="store_true", help="Overwrite existing .md files.")
    ap.add_argument("--wiki-root", type=Path, default=DEFAULT_WIKI_ROOT,
                    help="Directory holding .state.json for incremental compilation.")
    ap.add_argument("--concepts-dir", type=Path, default=DEFAULT_CONCEPTS,
                    help="Concepts directory to flag stale articles in when a "
                    "source paper's TEI hash changes.")
    ap.add_argument("--llm-provider", type=str, default=None)
    ap.add_argument("--llm-model", type=str, default=None)
    ap.add_argument(
        "--sample-review",
        type=int,
        default=10,
        help="Random-sample N newly compiled papers for manual review (seed=42). "
        "Set to 0 to skip.",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    papers = _pick_papers(args.grobid_root, args.paper_set, args.limit)
    if not papers:
        print("[compile_papers] no papers selected; aborting")
        sys.exit(1)

    client = get_client(args.llm_provider)
    if args.llm_model and hasattr(client, "model"):
        client.model = args.llm_model
    model_name = getattr(client, "model", args.llm_provider or "unknown")

    print(f"[compile_papers] paper_set={args.paper_set} limit={args.limit}")
    print(f"[compile_papers] {len(papers)} papers -> {args.output_dir}")
    print(f"[compile_papers] model: {model_name}")

    state = load_state(args.wiki_root)
    run_log: list[dict] = []
    new_stems: list[str] = []
    changed_ids: set[str] = set()
    n_ok = n_parse_error = n_empty = n_skipped = 0

    t0 = time.time()
    for stem, xml_path in papers:
        md_path = args.output_dir / f"{stem}.md"
        tei = xml_path.read_text()
        decision = decide(
            stem, tei, state,
            md_path_exists=md_path.exists(),
            force_rebuild=args.rebuild,
        )
        if not decision.needs_compile:
            if decision.reason == "bootstrap":
                state = update_state(state, stem, tei)
            print(f"  [skip:{decision.reason}] {stem}")
            n_skipped += 1
            run_log.append({"stem": stem, "status": "skipped", "reason": decision.reason})
            continue

        print(f"  [..:{decision.reason}] {stem}", flush=True)
        result = summarize_paper(tei, arxiv_id=stem, llm_client=client, model_name=model_name)
        md_path.write_text(result["markdown"])

        status = result["status"]
        if status == "ok":
            n_ok += 1
            state = update_state(state, stem, tei)
            if decision.reason == "changed":
                changed_ids.add(stem)
        elif status == "parse_error":
            n_parse_error += 1
        elif status == "empty_tei":
            n_empty += 1
        new_stems.append(stem)

        print(f"  [{status}] {stem}  {result['latency_ms']}ms")
        run_log.append({
            "stem": stem,
            "status": status,
            "reason": decision.reason,
            "latency_ms": result["latency_ms"],
            "raw_output_preview": result.get("raw_output"),
        })

    save_state(args.wiki_root, state)
    stale_markers = flag_stale_concepts(args.concepts_dir, changed_ids)
    if stale_markers:
        print(f"\n[compile_papers] flagged {len(stale_markers)} concept(s) as stale:")
        for m in stale_markers:
            print(f"  - {m}")
        print("  re-run scripts/compile_concepts.py --rebuild on those slugs to refresh.")

    elapsed = time.time() - t0
    summary = {
        "model": model_name,
        "paper_set": args.paper_set,
        "limit": args.limit,
        "elapsed_sec": round(elapsed, 1),
        "n_papers": len(papers),
        "n_ok": n_ok,
        "n_parse_error": n_parse_error,
        "n_empty_tei": n_empty,
        "n_skipped": n_skipped,
        "entries": run_log,
    }
    (args.output_dir / RUN_LOG).write_text(json.dumps(summary, indent=2))

    _append_history(
        args.output_dir,
        model=model_name,
        paper_set=args.paper_set,
        n_papers=len(papers),
        n_ok=n_ok,
        n_parse_error=n_parse_error,
        n_empty=n_empty,
        n_skipped=n_skipped,
        elapsed_sec=elapsed,
    )

    if args.sample_review > 0:
        _sample_for_review(
            args.output_dir,
            new_stems=new_stems,
            k=args.sample_review,
            seed=SAMPLE_SEED,
        )

    print(f"\n[compile_papers] done in {elapsed:.1f}s")
    print(f"  ok:          {n_ok}")
    print(f"  parse_error: {n_parse_error}")
    print(f"  empty_tei:   {n_empty}")
    print(f"  skipped:     {n_skipped}")


if __name__ == "__main__":
    main()
