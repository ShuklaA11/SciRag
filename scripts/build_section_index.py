"""Build a section-aware FAISS index over the QASPER corpus.

Mirrors scripts/build_flat_index.py but uses src.pipeline.section_chunker
so each chunk carries section_type and section_head fields. Output dir
defaults to data/index/flat_bge_sectioned/ (parallel to flat_bge/), so the
flat baseline stays reproducible.

Crash safety mirrors build_flat_index.py: jsonl row appended first, then
FAISS row added, then manifest. Resume trims jsonl to index.ntotal.
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

from src.pipeline.bge_embedder import BGEEmbedder
from src.pipeline.embedder import EMBED_DIM, Specter2Embedder
from src.pipeline.section_chunker import chunk_paper_by_section

TEI_DIR = Path("data/grobid_output/qasper")
DEFAULT_OUTPUT_BY_EMBEDDER = {
    "specter2": Path("data/index/flat_sectioned"),
    "bge": Path("data/index/flat_bge_sectioned"),
}


def _build_embedder(name: str):
    if name == "specter2":
        return Specter2Embedder()
    if name == "bge":
        return BGEEmbedder()
    raise ValueError(f"unknown embedder: {name}")


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
    if not jsonl_path.exists():
        return
    lines = jsonl_path.read_text().splitlines()
    if len(lines) > rows:
        jsonl_path.write_text("\n".join(lines[:rows]) + ("\n" if rows else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--embedder", choices=["specter2", "bge"], default="bge")
    ap.add_argument("--output", type=Path, default=None,
                    help="Index output dir. Defaults to data/index/flat_sectioned "
                         "or data/index/flat_bge_sectioned based on --embedder.")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--tei-dir", type=Path, default=TEI_DIR)
    args = ap.parse_args()

    out_dir: Path = args.output or DEFAULT_OUTPUT_BY_EMBEDDER[args.embedder]
    if args.rebuild and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.faiss"
    jsonl_path = out_dir / "chunks.jsonl"
    manifest_path = out_dir / "manifest.json"

    manifest = _load_manifest(manifest_path)
    index = _open_or_create_index(index_path)
    _trim_jsonl_to(index.ntotal, jsonl_path)

    embedder = _build_embedder(args.embedder)
    tokenizer = embedder.tokenizer

    xml_files = sorted(args.tei_dir.glob("*.xml"))
    if args.limit is not None:
        xml_files = xml_files[: args.limit]

    n_done = sum(1 for x in xml_files
                 if x.stem in manifest and manifest[x.stem].get("done"))
    print(f"[build_section_index] {len(xml_files)} TEI files queued, "
          f"{n_done} already done")
    print(f"[build_section_index] index dim={EMBED_DIM} "
          f"starting rows={index.ntotal} device={embedder.device}")

    t0 = time.time()
    new_papers = 0
    new_chunks = 0
    skipped_empty = 0
    section_type_counts: dict[str, int] = {}

    with jsonl_path.open("a") as jf:
        for i, xml_path in enumerate(xml_files, 1):
            arxiv_id = xml_path.stem
            if manifest.get(arxiv_id, {}).get("done"):
                continue

            tei = xml_path.read_text()
            chunks = chunk_paper_by_section(
                tei, tokenizer, chunk_size=512, overlap=64,
            )

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
                    "section_type": c["section_type"],
                    "section_head": c["section_head"],
                }
                jf.write(json.dumps(row) + "\n")
                section_type_counts[c["section_type"]] = (
                    section_type_counts.get(c["section_type"], 0) + 1
                )
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
                print(f"  [{i}/{len(xml_files)}] {arxiv_id} "
                      f"+{len(chunks)} chunks "
                      f"(total rows={index.ntotal}, {rate:.1f} chunks/s)")

    _save_manifest(manifest_path, manifest)
    elapsed = time.time() - t0
    print(f"\n[build_section_index] done in {elapsed:.1f}s")
    print(f"  papers added this run: {new_papers}")
    print(f"  chunks added this run: {new_chunks}")
    print(f"  empty papers skipped:  {skipped_empty}")
    print(f"  total index rows:      {index.ntotal}")
    print(f"  index file size:       {index_path.stat().st_size / 1e6:.1f} MB")
    if section_type_counts:
        print("  chunks by section_type (this run):")
        for st in sorted(section_type_counts, key=lambda k: -section_type_counts[k]):
            print(f"    {st:14s} {section_type_counts[st]}")


if __name__ == "__main__":
    main()
