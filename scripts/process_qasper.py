"""Download QASPER papers from arXiv and process them through Grobid.

Pipelined: downloads paper N+1 in a background thread while Grobid processes
paper N. Idempotent: re-runs skip papers already in the manifest. Writes the
manifest every 10 papers so crashes don't lose progress.

Usage:
    python scripts/process_qasper.py                  # train + dev (1,169 papers)
    python scripts/process_qasper.py --splits train   # train only (888 papers)
    python scripts/process_qasper.py --limit 50       # first 50 papers (smoke test)
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import requests

from src.pipeline.grobid_client import extract_sections, extract_title, process_pdf

ROOT = Path(__file__).resolve().parent.parent
QASPER_DIR = ROOT / "data" / "datasets" / "qasper"
PDFS_DIR = ROOT / "raw" / "papers" / "qasper"
XML_DIR = ROOT / "data" / "grobid_output" / "qasper"
MANIFEST_PATH = XML_DIR / "manifest.json"

ARXIV_HEADERS = {"User-Agent": "SciRAG-research/0.1 (https://github.com/ShuklaA11/SciRag)"}
ARXIV_DELAY = 3.0


def load_arxiv_ids(splits: list[str]) -> list[str]:
    ids = []
    for split in splits:
        path = QASPER_DIR / f"{split}.json"
        data = json.loads(path.read_text())
        ids.extend(data.keys())
    seen = set()
    unique = []
    for aid in ids:
        if aid not in seen:
            seen.add(aid)
            unique.append(aid)
    return unique


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(MANIFEST_PATH)


def download_pdf(arxiv_id: str, out_path: Path) -> dict:
    """Download a PDF from arxiv. Returns status dict."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    backoff = 5.0
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=ARXIV_HEADERS, timeout=60, allow_redirects=True)
            if resp.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                return {"download": "fail", "error": "not_a_pdf"}
            out_path.write_bytes(resp.content)
            return {"download": "ok", "size": len(resp.content)}
        except requests.HTTPError as e:
            return {"download": "fail", "error": f"http_{resp.status_code}"}
        except requests.RequestException as e:
            if attempt == 2:
                return {"download": "fail", "error": type(e).__name__}
            time.sleep(backoff)
            backoff *= 2
    return {"download": "fail", "error": "retries_exhausted"}


def downloader_worker(
    arxiv_ids: list[str],
    pdf_queue: "queue.Queue",
    manifest: dict,
    stop_event: threading.Event,
) -> None:
    """Producer: downloads PDFs and pushes (arxiv_id, pdf_path, status) tuples."""
    for arxiv_id in arxiv_ids:
        if stop_event.is_set():
            break
        if arxiv_id in manifest and manifest[arxiv_id].get("download") == "ok":
            pdf_path = PDFS_DIR / f"{arxiv_id}.pdf"
            if pdf_path.exists():
                pdf_queue.put((arxiv_id, pdf_path, manifest[arxiv_id]))
                continue

        pdf_path = PDFS_DIR / f"{arxiv_id}.pdf"
        status = download_pdf(arxiv_id, pdf_path)
        pdf_queue.put((arxiv_id, pdf_path, status))
        time.sleep(ARXIV_DELAY)

    pdf_queue.put(None)  # sentinel


def process_one(arxiv_id: str, pdf_path: Path) -> dict:
    """Run Grobid on a PDF and extract sections."""
    try:
        tei_xml = process_pdf(pdf_path)
        xml_path = XML_DIR / f"{arxiv_id}.xml"
        xml_path.write_text(tei_xml)
        sections = extract_sections(tei_xml)
        title = extract_title(tei_xml)
        return {
            "grobid": "ok",
            "title": title[:200],
            "num_sections": len(sections),
            "xml_size": len(tei_xml),
        }
    except Exception as e:
        return {"grobid": "fail", "error": f"{type(e).__name__}: {str(e)[:200]}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "dev"], choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Process only first N papers (smoke test)")
    args = parser.parse_args()

    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    XML_DIR.mkdir(parents=True, exist_ok=True)

    arxiv_ids = load_arxiv_ids(args.splits)
    if args.limit:
        arxiv_ids = arxiv_ids[: args.limit]

    manifest = load_manifest()
    print(f"QASPER splits: {args.splits}")
    print(f"Total papers: {len(arxiv_ids)}")
    print(f"Already in manifest: {len(manifest)}")
    print(f"Output: PDFs -> {PDFS_DIR.relative_to(ROOT)}, XML -> {XML_DIR.relative_to(ROOT)}")
    print()

    pdf_queue: queue.Queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    downloader = threading.Thread(
        target=downloader_worker,
        args=(arxiv_ids, pdf_queue, manifest, stop_event),
        daemon=True,
    )
    downloader.start()

    counts = {"download_ok": 0, "download_fail": 0, "grobid_ok": 0, "grobid_fail": 0, "skipped": 0}
    processed = 0
    start_time = time.time()

    try:
        while True:
            item = pdf_queue.get()
            if item is None:
                break
            arxiv_id, pdf_path, dl_status = item
            processed += 1

            already_done = (
                arxiv_id in manifest
                and manifest[arxiv_id].get("download") == "ok"
                and manifest[arxiv_id].get("grobid") == "ok"
            )
            if already_done:
                counts["skipped"] += 1
                if processed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"  [{processed}/{len(arxiv_ids)}] skipped (cached)  rate={rate:.2f}/s")
                continue

            if dl_status.get("download") != "ok":
                counts["download_fail"] += 1
                manifest[arxiv_id] = dl_status
                print(f"  [{processed}/{len(arxiv_ids)}] {arxiv_id} DOWNLOAD FAIL: {dl_status.get('error')}")
            else:
                counts["download_ok"] += 1
                grobid_status = process_one(arxiv_id, pdf_path)
                merged = {**dl_status, **grobid_status}
                manifest[arxiv_id] = merged
                if grobid_status["grobid"] == "ok":
                    counts["grobid_ok"] += 1
                    if processed % 25 == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        eta_min = (len(arxiv_ids) - processed) / rate / 60 if rate > 0 else 0
                        print(
                            f"  [{processed}/{len(arxiv_ids)}] {arxiv_id} ok "
                            f"({grobid_status['num_sections']} sections)  rate={rate:.2f}/s  eta={eta_min:.0f}min"
                        )
                else:
                    counts["grobid_fail"] += 1
                    print(f"  [{processed}/{len(arxiv_ids)}] {arxiv_id} GROBID FAIL: {grobid_status.get('error')}")

            if processed % 10 == 0:
                save_manifest(manifest)

    except KeyboardInterrupt:
        print("\nInterrupted — saving manifest")
        stop_event.set()
    finally:
        save_manifest(manifest)

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Processed: {processed} papers in {elapsed / 60:.1f} min")
    print(f"  download_ok:   {counts['download_ok']}")
    print(f"  download_fail: {counts['download_fail']}")
    print(f"  grobid_ok:     {counts['grobid_ok']}")
    print(f"  grobid_fail:   {counts['grobid_fail']}")
    print(f"  skipped:       {counts['skipped']}")
    coverage = counts["grobid_ok"] / max(processed, 1) * 100
    print(f"Coverage (grobid_ok / processed): {coverage:.1f}%")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
