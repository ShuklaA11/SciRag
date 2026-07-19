# SciRAG

**An LLM-compiled scientific knowledge base with section-aware retrieval, citation-graph expansion, and claim verification.**

SciRAG ingests scientific papers (PDFs), parses them with Grobid into structured sections, embeds them with SPECTER2, builds a citation graph via Semantic Scholar, and uses an LLM to compile a living Obsidian wiki of paper summaries and synthesized concept articles. Every retrieval and verification component is benchmarked individually against QASPER and SciFact. Domain: NLP/ML.

---

## Why this exists

Standard RAG fails on scientific papers in four ways:

1. **Dense jargon** breaks surface-similarity embeddings.
2. **Citations** create cross-paper dependencies that single-chunk retrieval misses.
3. **Tables, equations, and figures** carry meaning that text-only pipelines discard.
4. **Research questions are synthesis-oriented**, not factoid lookups.

SciRAG addresses each as a benchmarked component, then uses those components as the compilation engine for a maintained knowledge base.

## Architecture

```
Papers (PDF / arXiv)
    │
    ▼
Grobid ──► TEI XML (sections, refs, figures)
    │
    ▼
Section-aware chunking ──► SPECTER2 / BGE ──► FAISS (flat index)
    │
    ▼
Semantic Scholar ──► NetworkX citation graph
    │
    ▼
LLM compiler (pluggable) ──► Obsidian wiki (papers/, concepts/, indices/)
    │
    ▼
Query engine: classifier ──► section-routed retrieval + citation expansion
              ──► LangGraph multi-hop ──► NLI claim verification ──► wiki linter
```

## Tech stack

| Layer | Tool | Notes |
|---|---|---|
| PDF parsing | Grobid (Docker) | `lfoppiano/grobid:0.8.1` |
| Embeddings | SPECTER2 / BGE | scientific-paper-tuned / generic retrieval |
| Vector store | FAISS (`IndexFlatIP`) | flat index; Qdrant retained but legacy |
| LLM (verification + compilation) | Llama-3.1-8B Q4_K_M via Ollama | pluggable |
| LLM swap | `SCIRAG_LLM_PROVIDER` env var | `ollama` \| `anthropic` \| `openai` |
| Citation graph | Semantic Scholar API | SQLite-cached |
| Orchestration | LangGraph | multi-hop state machine |
| SQL warehouse | DuckDB | eval results to text-to-SQL (`src/sqllab/`) |
| Frontend | Obsidian + Streamlit | Obsidian vault viewer + Streamlit hub UI |

## Datasets

| Dataset | Purpose | Size |
|---|---|---|
| **QASPER** (1,585 NLP papers, 5,049 Q&A) | Primary retrieval benchmark | 59 MB |
| **SciFact** (1,409 claims with sentence-level evidence) | Claim verification | 11 MB |
| **Seed corpus** (50 NLP papers from QASPER train) | Pipeline development | 41 MB |

---

## Prerequisites

Built and tested on **Apple Silicon (M1 Pro, 16 GB RAM)** running macOS. Should work on any system with Docker, Python 3.11+, and ~16 GB RAM.

You will need:

- **Docker Desktop** (running)
- **Python 3.11+** (managed via `python3 -m venv` - no conda needed)
- **Ollama** ([install instructions](https://ollama.com/download))
- **~5 GB disk** for models, ~70 MB for datasets, ~50 MB for the seed corpus
- **(Optional) Semantic Scholar API key** - raises rate limit from 1 to 100 req/sec. Request one [here](https://www.semanticscholar.org/product/api).

### Memory constraint (important on 16 GB machines)

Grobid (JVM, ~4 GB) and Ollama with Llama-3.1-8B (~6 GB) cannot run concurrently on a 16 GB machine without swapping. The pipeline uses them in sequence:

```bash
# When you need Grobid:
docker compose start grobid
ollama stop llama3.1:8b

# When you need Ollama:
docker compose stop grobid
# Ollama loads the model on first request
```

Qdrant (~200 MB) is legacy and idle (FAISS is the active store), but it's light enough to leave running.

---

## Setup

### 1. Clone and create the venv

```bash
git clone <repo-url> SciRag
cd SciRag
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Pull the LLM model

```bash
ollama pull llama3.1:8b   # ~5 GB, Q4_K_M quantization
```

### 3. Start the services

```bash
docker compose up -d           # starts Grobid (8070) and Qdrant (6333)
```

Wait ~30 s for Grobid to be healthy:

```bash
curl http://localhost:8070/api/isalive   # should return "true"
```

### 4. Configure API keys (optional)

Create a `.env` file in the project root:

```
SEMANTIC_SCHOLAR_API_KEY=your_key_here
SCIRAG_LLM_PROVIDER=ollama   # or "anthropic" / "openai" once those are wired up
```

### 5. Download datasets and seed corpus

```bash
python scripts/download_datasets.py        # QASPER + SciFact (~70 MB)
python scripts/download_seed_papers.py     # 50 NLP papers from arXiv (~40 MB)
```

### 6. Verify with the smoke test

```bash
python scripts/grobid_smoke_test.py
```

You should see all PDFs in `raw/papers/` processed, with section counts printed for each. TEI XML is saved to `data/grobid_output/`.

### 7. Run the test suite

```bash
pytest tests/ -v
```

The suite passes; some tests skip when Grobid or Ollama are not running. The
text-to-SQL guard tests (`tests/test_sqllab_guard.py`) run standalone with no models.

---

## Project structure

```
SciRag/
├── raw/papers/              # source PDFs (gitignored)
├── wiki/                    # compiled Obsidian vault
│   ├── papers/              # auto-generated paper summaries
│   ├── concepts/            # synthesized concept articles
│   ├── indices/             # INDEX, GLOSSARY, TIMELINE, GRAPH, QUESTIONS
│   └── outputs/             # query results, generated figures
├── data/
│   ├── datasets/            # QASPER, SciFact (gitignored)
│   ├── index/               # FAISS flat index (gitignored)
│   ├── qdrant/              # legacy vector store (gitignored)
│   ├── citation_graph/      # NetworkX pickles (gitignored)
│   ├── cache/               # Semantic Scholar SQLite cache
│   ├── eval.duckdb          # eval-results warehouse (gitignored)
│   └── grobid_output/       # TEI XML cache (gitignored)
├── src/
│   ├── llm/                 # pluggable LLM client (Ollama / Anthropic / OpenAI)
│   ├── pipeline/            # Grobid client, chunking, embedding
│   ├── retrieval/           # FAISS search, citation expansion, multi-hop
│   ├── verification/        # NLI claim verification
│   ├── ideas/               # idea → claims → novelty-verdict engine
│   ├── brainstorm/          # agentic novelty-gap loop
│   ├── crew/                # LangGraph multi-agent supervisor
│   ├── domain/              # domain profiles (NLP/ML, biomedical)
│   ├── hub/                 # SQLite project/evaluation persistence
│   ├── router/              # section-routing classifiers
│   ├── sqllab/              # DuckDB warehouse + text-to-SQL over eval results
│   ├── evaluation/          # QASPER + SciFact eval harnesses
│   └── wiki/                # wiki compiler, linter, search
├── scripts/                 # one-shot CLI scripts (download, smoke tests, eval DB)
├── eval/                    # baselines, ablations, results, sql_gold.jsonl
├── tests/                   # pytest suites
├── docker-compose.yml       # Grobid + Qdrant
└── pyproject.toml           # dependencies and build config
```

## Running things

| Command | What it does |
|---|---|
| `python scripts/download_datasets.py` | Fetch QASPER + SciFact into `data/datasets/` |
| `python scripts/download_seed_papers.py` | Fetch 50 NLP papers into `raw/papers/` |
| `python scripts/grobid_smoke_test.py` | Process every PDF in `raw/papers/` through Grobid |
| `pytest tests/ -v` | Run the full test suite |
| `python scripts/build_eval_db.py` | Build the DuckDB eval warehouse from `eval/results/` |
| `python scripts/ask_sql.py "..."` | Ask the eval warehouse a question in natural language |
| `python scripts/eval_text_to_sql.py` | Benchmark text-to-SQL execution accuracy |
| `docker compose up -d` | Start Grobid + Qdrant |
| `docker compose stop grobid` | Free RAM for Ollama |

## Swapping the LLM provider

All LLM calls go through `src/llm/client.py`. Switch providers via env var - no code changes:

```bash
export SCIRAG_LLM_PROVIDER=ollama        # default: local Llama-3.1-8B Q4
export SCIRAG_LLM_PROVIDER=anthropic     # Claude (requires `pip install scirag[anthropic]`)
export SCIRAG_LLM_PROVIDER=openai        # GPT-4o (requires `pip install scirag[openai]`)
```

The Anthropic and OpenAI providers are stubs that raise `NotImplementedError` until wired up.

---

## Text-to-SQL over eval results

Retrieval is over unstructured papers. The benchmark runs those experiments produce,
however, are structured, so `src/sqllab/` materializes every `eval/results/` summary
and per-record `.jsonl` into a DuckDB warehouse and queries it in natural language.
This answers the aggregate questions vector search can't: *which config won?*,
*rerank vs. no rerank?*, *latency by embedder?*

```bash
python scripts/build_eval_db.py                                   # build data/eval.duckdb
python scripts/ask_sql.py "which retrieval run had the best fuzzy recall?"
python scripts/ask_sql.py --summary "how many runs used reranking?"
python scripts/eval_text_to_sql.py                                # execution-match benchmark
```

The NL→SQL layer is **schema-grounded** (an auto-generated data dictionary is injected
into the prompt), enforces a **read-only single-`SELECT` guard**, and runs one
`EXPLAIN`-driven **repair** on failure. Accuracy is measured by execution match against
`eval/sql_gold.jsonl`: **13/16 (81.2%)** with Llama-3.1-8B via Ollama. The harness is
provider-agnostic (`SCIRAG_LLM_PROVIDER`), so additional providers can be benchmarked head-to-head once wired up.

---

## Further reading

| Doc | What's inside |
|---|---|
| [`docs/WRITEUP.md`](docs/WRITEUP.md) | Full technical writeup: benchmarked methods layer, domain-adaptive hub, and the auditable novelty evaluator |
| [`DEMO.md`](DEMO.md) | ~3-minute end-to-end walkthrough of the system |
| [`eval/results/README.md`](eval/results/README.md) | How the per-component eval runs are structured (`.jsonl` rows + `_summary.json` aggregates) |
| [`data/README.md`](data/README.md) | Inventory of every data artifact under `data/` and `raw/` |
| [`app/streamlit_app.py`](app/streamlit_app.py) | Streamlit hub UI entrypoint |
| [`wiki/indices/`](wiki/indices/) | Auto-generated INDEX, GLOSSARY, TIMELINE, and QUESTIONS over the compiled wiki |
