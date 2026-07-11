# SciRAG — Demo Guide

A ~3-minute walkthrough of the system: the benchmarked retrieval spine, claim
verification, the live idea-novelty evaluator (the differentiator), the
temporal-novelty result, and the interactive surfaces.

## Fastest path — the narrated CLI demo

```bash
python scripts/demo.py                 # full run, incl. a LIVE idea evaluation
python scripts/demo.py --no-live       # fast dry run (no model load)
python scripts/demo.py --idea "your research idea here"
```

The full run needs Ollama running and pulls the zero-shot NLI model on first use;
the benchmarked stages read frozen results and are instant.

## 3-minute storyboard

**0:00 — Framing (15s).** "SciRAG is an open, *benchmarked* methods layer for
scientific-paper RAG — every component is measured as an ablation — plus an
auditable idea-novelty evaluator that closed products don't publish numbers for."

**0:15 — Benchmarked spine (40s).** Run `demo.py`, stage 1. Walk the A–E ablation
(QASPER, recall@5). Land the honest finding: **the full system (E=0.692) is
*worse* than its parts (C=0.771, D=0.772)** — citation expansion dilutes in-paper
recall@5. "Measuring beat guessing; more components isn't monotonically better."

**0:55 — Verification (20s).** Stage 2. Zero-shot NLI claim verification on
SciFact (0.691 acc). Note the fine-tune is deferred (CUDA-only) and *documented*,
not hidden — honesty is the point.

**1:15 — Live idea evaluation (60s) — the differentiator.** Stage 3, live.
An idea is decomposed into atomic claims, each retrieves corpus evidence, NLI
buckets it **🟢 ENTAILED / 🔴 CONTRADICTED / 🟡 NOVEL** — per-claim, never a
gameable scalar. Then show the same thing interactively:

```bash
streamlit run app/streamlit_app.py
```

Create a project → **Evaluate idea** → color-coded verdicts, persisted to SQLite.

**2:15 — Temporal-novelty result (30s).** Stage 4. The wedge: an arXiv-year
holdout. In-corpus papers score **0.388** NOVEL, held-out (next-year) papers
**0.605** — a **+0.218 gap** in the pre-registered direction. "This is the number
closed products don't publish."

**2:45 — Agentic loop (15s).** The brainstorm loop discovers literature gaps:

```bash
python scripts/run_brainstorm.py --seed "diffusion models for low-resource text generation"
```

Retrieve → assess novelty → surface NOVEL gaps → LLM proposes next directions →
cosine-dedup → repeat (capped).

## What to emphasize

- **Benchmarked, not vibes** — every retrieval claim traces to `eval/baseline_v2.json`,
  regression-guarded by `tests/test_baseline_v2_manifest.py`.
- **Honest negatives** — the config-E finding and the deferred fine-tune are
  features of the writeup, not omissions.
- **Auditable novelty** — per-claim entailment against a corpus, validated by a
  temporal holdout; a directional proxy, framed as such.

Full numbers and positioning: `docs/WRITEUP.md`.
