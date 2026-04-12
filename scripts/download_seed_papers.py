"""Download a seed corpus of NLP papers from arXiv, sourced from QASPER IDs.

QASPER's dict keys are arXiv IDs, so the seed corpus is guaranteed to overlap
with the eval set.

Usage:
    python scripts/download_seed_papers.py             # download 50 papers
    python scripts/download_seed_papers.py --count 100 # custom count
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
QASPER_TRAIN = ROOT / "data" / "datasets" / "qasper" / "train.json"
PAPERS_DIR = ROOT / "raw" / "papers"
MANIFEST_PATH = PAPERS_DIR / "seed_manifest.json"


def load_qasper_arxiv_ids() -> list[str]:
    """QASPER paper dicts are keyed by arXiv ID."""
    data = json.loads(QASPER_TRAIN.read_text())
    return list(data.keys())


def already_downloaded() -> set[str]:
    """Return set of arxiv IDs (or filenames) already in raw/papers/."""
    if not PAPERS_DIR.exists():
        return set()
    return {p.stem for p in PAPERS_DIR.glob("*.pdf")}


def download_arxiv_pdf(arxiv_id: str, out_path: Path) -> bool:
    """Download a PDF from arxiv.org. Returns True on success."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            return False
        out_path.write_bytes(resp.content)
        return True
    except (requests.RequestException, OSError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50, help="Target paper count")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between requests")
    args = parser.parse_args()

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    arxiv_ids = load_qasper_arxiv_ids()
    existing = already_downloaded()
    print(f"QASPER train has {len(arxiv_ids)} papers")
    print(f"Already in raw/papers/: {len(existing)} files")
    print(f"Target: {args.count} total")
    print()

    needed = args.count - len(existing)
    if needed <= 0:
        print(f"Already have {len(existing)} papers — nothing to download")
        return

    downloaded = []
    failed = []
    candidates = [aid for aid in arxiv_ids if aid not in existing]

    for arxiv_id in candidates:
        if len(downloaded) >= needed:
            break

        out_path = PAPERS_DIR / f"{arxiv_id}.pdf"
        print(f"  [{len(downloaded) + 1}/{needed}] {arxiv_id} ... ", end="", flush=True)

        if download_arxiv_pdf(arxiv_id, out_path):
            print("OK")
            downloaded.append(arxiv_id)
        else:
            print("FAIL")
            failed.append(arxiv_id)

        time.sleep(args.delay)  # be polite to arxiv

    # Write manifest
    manifest = {
        "total_in_corpus": len(already_downloaded()),
        "downloaded_this_run": downloaded,
        "failed_this_run": failed,
        "source": "qasper train.json arXiv IDs",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print()
    print(f"Downloaded: {len(downloaded)}")
    print(f"Failed: {len(failed)}")
    print(f"Total corpus: {len(already_downloaded())} papers in {PAPERS_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")

    if failed and len(downloaded) + len(existing) < args.count:
        print(f"\nWarning: only {len(downloaded) + len(existing)}/{args.count} papers — re-run to retry failures")
        sys.exit(1)


if __name__ == "__main__":
    main()
