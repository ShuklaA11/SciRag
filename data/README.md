# SciRAG Data Inventory

Single source of truth for every data artifact under `data/` and `raw/`. If
you need to know where something came from, when it was built, or how to
regenerate it, look here first.

> **Snapshot:** numbers in this document reflect commit `2c2264a`
> (Week 2, end of Task 2). Re-run the regenerate commands below if you need
> fresh numbers.

---

## Layout

```
SciRag/
├── raw/papers/                       # source PDFs (gitignored except seed_manifest.json)
│   ├── *.pdf                         # 50 seed papers
│   ├── seed_manifest.json            # tracked
│   └── qasper/*.pdf                  # 1,166 QASPER PDFs (gitignored)
└── data/
    ├── datasets/                     # raw HF dataset dumps (gitignored)
    │   ├── qasper/
    │   └── scifact/
    ├── grobid_output/qasper/         # TEI XML cache (gitignored)
    │   ├── *.xml
    │   └── manifest.json
    ├── cache/
    │   └── s2_cache.db               # SQLite Semantic Scholar response cache (gitignored)
    └── citation_graph/
        └── graph.pickle              # NetworkX DiGraph (gitignored)
```

`data/README.md` (this file) is the only thing under `data/` that lives in
git. The JSON manifests under `raw/papers/` are also tracked.

---

## Artifacts

| # | Artifact | Path | Source | Size | Items | Regenerate with |
|---|---|---|---|---|---|---|
| 1 | QASPER dataset | `data/datasets/qasper/` | HF `allenai/qasper` | 59 MB | 1,585 papers / 5,049 Q&A | `python scripts/download_datasets.py` |
| 2 | SciFact dataset | `data/datasets/scifact/` | HF `allenai/scifact` | 11 MB | 1,409 claims | `python scripts/download_datasets.py` |
| 3 | Seed corpus PDFs | `raw/papers/*.pdf` | arXiv | 41 MB | 50 papers (45 from QASPER train + 5 canonical) | `python scripts/download_seed_papers.py` |
| 4 | QASPER PDFs | `raw/papers/qasper/*.pdf` | arXiv (via QASPER train+dev IDs) | 962 MB | 1,166 PDFs | `python scripts/process_qasper.py` |
| 5 | QASPER TEI XML | `data/grobid_output/qasper/*.xml` | Grobid `lfoppiano/grobid:0.8.1` | 79 MB | 1,166 XMLs | `python scripts/process_qasper.py` |
| 6 | S2 response cache | `data/cache/s2_cache.db` | Semantic Scholar Graph API | 12 MB | 1,166 papers cached | `python scripts/build_citation_graph.py` |
| 7 | Citation graph | `data/citation_graph/graph.pickle` | Derived from S2 cache | 548 KB | 6,300 nodes / 15,642 edges | `python scripts/build_citation_graph.py` |

The seed corpus (#3) is a deliberate subset for pipeline development: 45
papers sampled from QASPER train, plus 5 canonical NLP papers
(Attention-Is-All-You-Need, BERT, ELMo, GPT-2, SciBERT). The full QASPER
processing pipeline (#4, #5) is the production data path.

---

## Coverage numbers (Week 2)

End-of-week metrics, frozen for the writeup.

### Grobid PDF parsing (script: `process_qasper.py`)

| Metric | Value |
|---|---|
| Papers attempted | 1,169 (QASPER train+dev) |
| download_ok + grobid_ok | 1,166 |
| Coverage | **99.7%** |
| download_fail | 3 |
| grobid_fail | 0 |

### Semantic Scholar lookup (script: `build_citation_graph.py`)

| Metric | Value |
|---|---|
| Corpus papers looked up | 1,166 |
| Found in S2 | 1,159 |
| Coverage | **99.4%** |
| Missing (404) | 7 |
| Retry-exhausted errors | 0 |

### Citation graph

| Metric | Value |
|---|---|
| Total arXiv references collected | 15,642 |
| In-corpus edges | 565 |
| In-corpus edge ratio | **3.6%** |
| Graph nodes | 6,300 |
| Graph edges | 15,642 |
| Density | 3.94e-4 |
| Avg out-degree (non-zero) | 13.94 |

The 3.6% in-corpus ratio is sparse but expected: QASPER is a sliced NLP
subset, not a closed citation universe. This is the baseline number that
Week 6's citation-graph retrieval ablation will improve on.

---

## Manifest schemas

### `raw/papers/seed_manifest.json`

```jsonc
{
  "total_in_corpus": 50,
  "downloaded_this_run": ["1909.00694", "2003.07723", ...],   // arXiv IDs
  "failed_this_run": [],
  "source": "QASPER train + canonical NLP papers"
}
```

### `data/grobid_output/qasper/manifest.json`

One entry per arXiv ID, written incrementally during processing:

```jsonc
{
  "1903.10676": {
    "download": "ok",                  // "ok" | "fail"
    "size": 487341,                    // bytes (when download=ok)
    "error": "http_404",               // (when download=fail)
    "grobid": "ok",                    // "ok" | "fail" (when download=ok)
    "title": "SciBERT: A Pretrained ...",
    "num_sections": 9,
    "xml_size": 41827
  }
}
```

### `data/cache/s2_cache.db` (SQLite)

```sql
CREATE TABLE papers (
    paper_id      TEXT PRIMARY KEY,    -- arXiv ID (NOT S2 paperId)
    response_json TEXT NOT NULL,       -- raw S2 response, or {"_s2_not_found": true}
    fetched_at    INTEGER NOT NULL     -- unix seconds
);
```

The `_s2_not_found` sentinel caches 404s so repeated lookups don't re-hit
the API.

### `data/citation_graph/graph.pickle`

`networkx.DiGraph` with:
- **Nodes**: arXiv IDs. Attribute `in_corpus: bool` is `True` iff the ID
  is in the SciRAG corpus (seed + QASPER train+dev).
- **Edges**: `citer → cited` (one edge per arXiv-resolvable reference).

Load with:

```python
import pickle
with open("data/citation_graph/graph.pickle", "rb") as f:
    graph = pickle.load(f)
```

---

## Reproducibility

**Deterministic** (same input → same output, indefinitely):
- arXiv IDs and PDF contents (arXiv preprints are immutable once posted)
- QASPER and SciFact dataset dumps (versioned on Hugging Face)
- Grobid output, given the pinned image `lfoppiano/grobid:0.8.1`

**Non-deterministic** (output drifts over time):
- Semantic Scholar reference lists. S2 enriches metadata over time, so a
  paper fetched today may have more references than the same paper fetched
  six months ago. The cache mitigates this but doesn't eliminate it — TTL
  refresh is a future enhancement.

**Environment pins:**
- Python: 3.12.4
- NetworkX: 3.3+ (see `pyproject.toml`)
- Grobid: `lfoppiano/grobid:0.8.1` (see `docker-compose.yml`)

---

## Gitignored vs tracked

| Tracked in git | Gitignored |
|---|---|
| `raw/papers/seed_manifest.json` | All `*.pdf` files (`raw/papers/**/*.pdf`) |
| `data/README.md` (this file) | `data/datasets/` |
|  | `data/grobid_output/` |
|  | `data/cache/*.db` |
|  | `data/citation_graph/*.pickle` |
|  | `data/qdrant/` |

The full data footprint (~1.1 GB) is reproducible from the scripts above
and is not worth committing. The seed manifest is small and acts as the
canonical record of the seed corpus contents.

---

## Future enhancements (not in scope for Week 2)

- Add `created_at` / `tool_version` fields to manifests on next regenerate
- TTL refresh logic for S2 cache (re-fetch entries older than N days)
- DVC integration if/when a collaborator needs to pull artifacts
- File-level hashes for tamper detection (not needed for solo dev)
