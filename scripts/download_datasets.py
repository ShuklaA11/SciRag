"""Download QASPER and SciFact datasets into data/datasets/.

QASPER: downloaded from S3 (HF datasets lib dropped script-based loaders in v4).
SciFact: loaded via HuggingFace datasets (has native Parquet support).

Usage:
    python scripts/download_datasets.py          # download both
    python scripts/download_datasets.py qasper    # download only QASPER
    python scripts/download_datasets.py scifact   # download only SciFact
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"

# QASPER v0.3 hosted on S3 (same URLs the HF dataset script uses)
_QASPER_URLS = {
    "train_dev": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
}
_QASPER_FILES = {
    "train": "qasper-train-v0.3.json",
    "dev": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}


def download_qasper() -> None:
    out_dir = DATA_DIR / "qasper"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = out_dir / "manifest.json"
    if manifest.exists():
        print(f"[qasper] Already downloaded — skipping (delete {manifest} to re-download)")
        return

    counts = {}
    for archive_name, url in _QASPER_URLS.items():
        print(f"[qasper] Downloading {archive_name} archive...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            for split, filename in _QASPER_FILES.items():
                members = [m for m in tar.getmembers() if m.name.endswith(filename)]
                if not members:
                    continue
                f = tar.extractfile(members[0])
                if f is None:
                    continue
                data = json.load(f)
                out_path = out_dir / f"{split}.json"
                out_path.write_text(json.dumps(data, indent=2))
                counts[split] = len(data)
                print(f"  {split}: {counts[split]} papers -> {out_path.name}")

    manifest.write_text(json.dumps(counts, indent=2))
    print(f"[qasper] Done. Total: {sum(counts.values())} papers")


_SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def download_scifact() -> None:
    out_dir = DATA_DIR / "scifact"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = out_dir / "manifest.json"
    if manifest.exists():
        print(f"[scifact] Already downloaded — skipping (delete {manifest} to re-download)")
        return

    print("[scifact] Downloading from S3...")
    resp = requests.get(_SCIFACT_URL, timeout=120)
    resp.raise_for_status()

    counts = {}
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".jsonl"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            lines = [json.loads(line) for line in f.read().decode().strip().split("\n")]
            name = Path(member.name).stem  # e.g. "claims_train", "corpus"
            out_path = out_dir / f"{name}.json"
            out_path.write_text(json.dumps(lines, indent=2))
            counts[name] = len(lines)
            print(f"  {name}: {counts[name]} rows -> {out_path.name}")

    manifest.write_text(json.dumps(counts, indent=2))
    print(f"[scifact] Done. Total: {sum(counts.values())} rows")


def main() -> None:
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["qasper", "scifact"]

    for target in targets:
        if target == "qasper":
            download_qasper()
        elif target == "scifact":
            download_scifact()
        else:
            print(f"Unknown dataset: {target}. Choose from: qasper, scifact")
            sys.exit(1)


if __name__ == "__main__":
    main()
