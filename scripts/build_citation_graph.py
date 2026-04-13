"""Build a NetworkX citation DiGraph over the SciRAG corpus via Semantic Scholar.

Corpus = union of:
  - raw/papers/seed_manifest.json[downloaded_this_run]
  - data/grobid_output/qasper/manifest.json (entries where grobid == "ok")

For each corpus paper, fetch S2 references (SQLite-cached). Edges go from
citer -> cited. A node's `in_corpus` attribute is True iff the arXiv ID is in
the corpus set.

Output: data/citation_graph/graph.pickle
Report: papers fetched, cache hit rate, in-corpus edges, density, out-degree.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.s2_client import S2Client  # noqa: E402

SEED_MANIFEST = ROOT / "raw" / "papers" / "seed_manifest.json"
QASPER_MANIFEST = ROOT / "data" / "grobid_output" / "qasper" / "manifest.json"
GRAPH_OUT = ROOT / "data" / "citation_graph" / "graph.pickle"
ENV_PATH = ROOT / ".env"


def load_env(path: Path) -> None:
    """Minimal .env loader — only sets vars that aren't already in the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_corpus_ids() -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    if SEED_MANIFEST.exists():
        seed = json.loads(SEED_MANIFEST.read_text())
        for aid in seed.get("downloaded_this_run", []):
            if aid not in seen:
                seen.add(aid)
                ids.append(aid)

    if QASPER_MANIFEST.exists():
        qasper = json.loads(QASPER_MANIFEST.read_text())
        for aid, status in qasper.items():
            if status.get("grobid") == "ok" and aid not in seen:
                seen.add(aid)
                ids.append(aid)

    return ids


def extract_arxiv_id(reference: dict) -> str | None:
    """Pull the arXiv ID out of an S2 reference record, if present."""
    ext = reference.get("externalIds") or {}
    return ext.get("ArXiv")


def main() -> None:
    load_env(ENV_PATH)
    if not os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        print("WARNING: SEMANTIC_SCHOLAR_API_KEY not set — using unauthenticated rate limit (1 req/s)")

    corpus_ids = load_corpus_ids()
    corpus_set = set(corpus_ids)
    print(f"Corpus size: {len(corpus_ids)} papers")
    print(f"  seed manifest: {SEED_MANIFEST.exists()}")
    print(f"  qasper manifest: {QASPER_MANIFEST.exists()}")
    print()

    client = S2Client()
    graph = nx.DiGraph()

    # Pre-add every corpus paper as a node, regardless of S2 lookup outcome
    for aid in corpus_ids:
        graph.add_node(aid, in_corpus=True)

    s2_found = 0
    s2_missing = 0
    s2_errored = 0
    total_refs_seen = 0
    in_corpus_edges = 0
    start = time.time()

    for i, arxiv_id in enumerate(corpus_ids, start=1):
        try:
            paper = client.get_paper(arxiv_id)
        except Exception as e:
            s2_errored += 1
            print(f"  [{i}/{len(corpus_ids)}] {arxiv_id} ERROR: {type(e).__name__}: {e}")
            paper = None
        if paper is None:
            s2_missing += 1
            if i % 25 == 0:
                _log_progress(i, len(corpus_ids), start, client)
            continue
        s2_found += 1
        refs = paper.get("references") or []
        for ref in refs:
            ref_arxiv = extract_arxiv_id(ref)
            if ref_arxiv is None:
                continue
            total_refs_seen += 1
            if ref_arxiv not in graph:
                graph.add_node(ref_arxiv, in_corpus=(ref_arxiv in corpus_set))
            graph.add_edge(arxiv_id, ref_arxiv)
            if ref_arxiv in corpus_set:
                in_corpus_edges += 1

        if i % 25 == 0:
            _log_progress(i, len(corpus_ids), start, client)

    elapsed = time.time() - start

    GRAPH_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_OUT, "wb") as f:
        pickle.dump(graph, f)

    total_http = client.http_calls
    total_lookups = s2_found + s2_missing
    cache_hit_rate = client.cache_hits / max(total_lookups, 1) * 100
    out_degrees = [d for _, d in graph.out_degree() if d > 0]
    avg_out = sum(out_degrees) / len(out_degrees) if out_degrees else 0.0

    print()
    print("=" * 60)
    print(f"Elapsed: {elapsed / 60:.1f} min")
    print(f"Corpus papers looked up:   {total_lookups}")
    print(f"  S2 found:                {s2_found}")
    print(f"  S2 missing (404 or err): {s2_missing}")
    print(f"  S2 errored (retried-out): {s2_errored}")
    print(f"HTTP calls:                {total_http}")
    print(f"Cache hits:                {client.cache_hits}  ({cache_hit_rate:.1f}%)")
    print(f"Total arXiv references:    {total_refs_seen}")
    print(f"  in-corpus edges:         {in_corpus_edges}")
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"  density:                 {nx.density(graph):.6f}")
    print(f"  avg out-degree (non-zero): {avg_out:.2f}")
    print(f"Saved: {GRAPH_OUT.relative_to(ROOT)}")


def _log_progress(i: int, total: int, start: float, client: S2Client) -> None:
    elapsed = time.time() - start
    rate = i / elapsed if elapsed > 0 else 0
    eta = (total - i) / rate / 60 if rate > 0 else 0
    print(
        f"  [{i}/{total}] http={client.http_calls} cache={client.cache_hits} "
        f"rate={rate:.1f}/s eta={eta:.1f}min"
    )


if __name__ == "__main__":
    main()
