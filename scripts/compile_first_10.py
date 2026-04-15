"""Compile the first-10 paper summaries for Week 3 manual quality review.

Paper set (decided in the Sub-task D plan):
  - 5 canonical: attention_is_all_you_need, bert, elmo, gpt2, scibert
  - 5 QASPER: first 5 alphabetical from data/grobid_output/qasper/

Output goes to wiki/papers/{stem}.md plus a .run_log.json with per-paper
timing and status. Resumable: skips papers whose .md already exists
unless --rebuild is passed.

Memory dance: stop Grobid before running (needs Ollama in RAM).
    docker compose stop grobid
    ollama run llama3.1:8b   # warm model, then /bye
    python scripts/compile_first_10.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.client import get_client
from src.wiki.summarizer import summarize_paper

CANONICAL = [
    "attention_is_all_you_need",
    "bert",
    "elmo",
    "gpt2",
    "scibert",
]
GROBID_ROOT = Path("data/grobid_output")
QASPER_DIR = GROBID_ROOT / "qasper"
DEFAULT_OUTPUT = Path("wiki/papers")


def _pick_papers(grobid_root: Path) -> list[tuple[str, Path]]:
    """Return [(stem, xml_path), ...] for 5 canonical + 5 qasper."""
    out: list[tuple[str, Path]] = []
    for stem in CANONICAL:
        p = grobid_root / f"{stem}.xml"
        if p.exists():
            out.append((stem, p))
        else:
            print(f"  [warn] canonical paper missing: {p}")

    qasper_xmls = sorted((grobid_root / "qasper").glob("*.xml"))[:5]
    for p in qasper_xmls:
        out.append((p.stem, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grobid-root", type=Path, default=GROBID_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--rebuild", action="store_true",
                    help="Overwrite existing .md files.")
    ap.add_argument("--llm-provider", type=str, default=None)
    ap.add_argument("--llm-model", type=str, default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    papers = _pick_papers(args.grobid_root)
    if not papers:
        print("[compile_first_10] no papers selected; aborting")
        sys.exit(1)

    client = get_client(args.llm_provider)
    if args.llm_model and hasattr(client, "model"):
        client.model = args.llm_model
    model_name = getattr(client, "model", args.llm_provider or "unknown")

    print(f"[compile_first_10] {len(papers)} papers -> {args.output_dir}")
    print(f"[compile_first_10] model: {model_name}")

    run_log: list[dict] = []
    n_ok = 0
    n_parse_error = 0
    n_empty = 0
    n_skipped = 0

    t0 = time.time()
    for stem, xml_path in papers:
        md_path = args.output_dir / f"{stem}.md"
        if md_path.exists() and not args.rebuild:
            print(f"  [skip] {stem} (already compiled)")
            n_skipped += 1
            run_log.append({"stem": stem, "status": "skipped"})
            continue

        print(f"  [..] {stem}", flush=True)
        tei = xml_path.read_text()
        result = summarize_paper(tei, arxiv_id=stem, llm_client=client, model_name=model_name)
        md_path.write_text(result["markdown"])

        status = result["status"]
        if status == "ok":
            n_ok += 1
        elif status == "parse_error":
            n_parse_error += 1
        elif status == "empty_tei":
            n_empty += 1

        print(f"  [{status}] {stem}  {result['latency_ms']}ms")
        run_log.append({
            "stem": stem,
            "status": status,
            "latency_ms": result["latency_ms"],
            "raw_output_preview": result.get("raw_output"),
        })

    elapsed = time.time() - t0
    summary = {
        "model": model_name,
        "elapsed_sec": round(elapsed, 1),
        "n_papers": len(papers),
        "n_ok": n_ok,
        "n_parse_error": n_parse_error,
        "n_empty_tei": n_empty,
        "n_skipped": n_skipped,
        "entries": run_log,
    }
    (args.output_dir / ".run_log.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[compile_first_10] done in {elapsed:.1f}s")
    print(f"  ok:          {n_ok}")
    print(f"  parse_error: {n_parse_error}")
    print(f"  empty_tei:   {n_empty}")
    print(f"  skipped:     {n_skipped}")
    print(f"\nNext: review wiki/papers/*.md and fill out wiki/papers/REVIEW_NOTES.md")


if __name__ == "__main__":
    main()
