"""Build the flat FAISS index over the QASPER corpus.

Loops every TEI XML in data/grobid_output/qasper/, chunks each paper with
the SPECTER2 tokenizer, embeds the chunks, and appends them to a FAISS
IndexFlatIP at data/index/flat/index.faiss. Chunk metadata is written to
chunks.jsonl one row per FAISS vector (same order). A manifest.json
tracks per-paper completion so re-runs are resumable.

Crash safety:
  - jsonl line is appended first, then the FAISS row, then manifest.
    Worst case on crash: jsonl has a few extra trailing lines we trim
    on resume to match index.ntotal.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.chunker import chunk_paper
from src.pipeline.embedder import EMBED_DIM, Specter2Embedder

TEI_DIR = Path("data/grobid_output/qasper")
DEFAULT_OUTPUT = Path("data/index/flat")


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)


def _open_or_create_index(index_path: Path) -> faiss.Index:
    if index_path.exists():
        return faiss.read_index(str(index_path))
    return faiss.IndexFlatIP(EMBED_DIM)


def _trim_jsonl_to(rows: int, jsonl_path: Path) -> None:
    """Truncate chunks.jsonl to the first `rows` lines (alignment recovery)."""
    if not jsonl_path.exists():
        return
    lines = jsonl_path.read_text().splitlines()
    if len(lines) > rows:
        jsonl_path.write_text("\n".join(lines[:rows]) + ("\n" if rows else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N TEI files (smoke test).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--rebuild", action="store_true",
                    help="Wipe output dir and start from scratch.")
    ap.add_argument("--tei-dir", type=Path, default=TEI_DIR)
    args = ap.parse_args()

    out_dir: Path = args.output
    if args.rebuild and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.faiss"
    jsonl_path = out_dir / "chunks.jsonl"
    manifest_path = out_dir / "manifest.json"

    manifest = _load_manifest(manifest_path)
    index = _open_or_create_index(index_path)
    _trim_jsonl_to(index.ntotal, jsonl_path)

    embedder = Specter2Embedder()
    tokenizer = embedder.tokenizer

    xml_files = sorted(args.tei_dir.glob("*.xml"))
    if args.limit is not None:
        xml_files = xml_files[: args.limit]

    print(f"[build_flat_index] {len(xml_files)} TEI files queued, "
          f"{sum(1 for x in xml_files if x.stem in manifest and manifest[x.stem].get('done'))} already done")
    print(f"[build_flat_index] index dim={EMBED_DIM} starting rows={index.ntotal} device={embedder.device}")

    t0 = time.time()
    new_papers = 0
    new_chunks = 0
    skipped_empty = 0

    with jsonl_path.open("a") as jf:
        for i, xml_path in enumerate(xml_files, 1):
            arxiv_id = xml_path.stem
            if manifest.get(arxiv_id, {}).get("done"):
                continue

            tei = xml_path.read_text()
            chunks = chunk_paper(tei, tokenizer, chunk_size=512, overlap=64)

            if not chunks:
                manifest[arxiv_id] = {"num_chunks": 0, "done": True}
                skipped_empty += 1
                if i % 50 == 0:
                    _save_manifest(manifest_path, manifest)
                continue

            texts = [c["text"] for c in chunks]
            vecs = embedder.encode(texts, batch_size=args.batch_size)

            for c, v in zip(chunks, vecs):
                row = {
                    "chunk_id": f"{arxiv_id}::{c['chunk_idx']}",
                    "arxiv_id": arxiv_id,
                    "chunk_idx": c["chunk_idx"],
                    "text": c["text"],
                    "token_count": c["token_count"],
                }
                jf.write(json.dumps(row) + "\n")
            jf.flush()

            index.add(vecs.astype(np.float32))
            faiss.write_index(index, str(index_path))

            manifest[arxiv_id] = {"num_chunks": len(chunks), "done": True}
            new_papers += 1
            new_chunks += len(chunks)

            if i % 25 == 0 or i == len(xml_files):
                _save_manifest(manifest_path, manifest)
                elapsed = time.time() - t0
                rate = new_chunks / elapsed if elapsed > 0 else 0
                print(f"  [{i}/{len(xml_files)}] {arxiv_id} +{len(chunks)} chunks "
                      f"(total rows={index.ntotal}, {rate:.1f} chunks/s)")

    _save_manifest(manifest_path, manifest)
    elapsed = time.time() - t0
    print(f"\n[build_flat_index] done in {elapsed:.1f}s")
    print(f"  papers added this run: {new_papers}")
    print(f"  chunks added this run: {new_chunks}")
    print(f"  empty papers skipped:  {skipped_empty}")
    print(f"  total index rows:      {index.ntotal}")
    print(f"  index file size:       {index_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
