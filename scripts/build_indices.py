"""Rebuild all wiki indices from scratch.

Idempotent: walks wiki/papers/ + wiki/concepts/ and overwrites the
five files in wiki/indices/. Safe to re-run after every paper or
concept compile.

    python scripts/build_indices.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wiki.indices import build_all

DEFAULT_PAPERS = Path("wiki/papers")
DEFAULT_CONCEPTS = Path("wiki/concepts")
DEFAULT_OUT = Path("wiki/indices")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS)
    ap.add_argument("--concepts-dir", type=Path, default=DEFAULT_CONCEPTS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    t0 = time.time()
    counts = build_all(args.papers_dir, args.concepts_dir, args.output_dir)
    elapsed = time.time() - t0

    print(f"[build_indices] wrote 5 indices to {args.output_dir} in {elapsed:.2f}s")
    print(f"  papers:   {counts['n_papers']}")
    print(f"  concepts: {counts['n_concepts']}")
    print(f"  edges:    {counts['n_edges']}")


if __name__ == "__main__":
    main()
