# SciRAG v2 — Implementation Plan

Condensed from `SciRAG_v2_Project_Plan.pdf` (21pp). Source of truth for week-by-week execution. Update as decisions land.

## Locked Decisions
- ✅ **Domain**: NLP/ML (QASPER-aligned, cleaner eval story, lower setup cost)
- ✅ **LLM**: Ollama local (Llama-3.1-8B Q4) for both verification *and* wiki compilation. Pluggable — see `src/llm/client.py` abstraction.

## Open Decisions (resolve before Week 1)
- [ ] **Schedule**: 14 weeks @ 10–15h/wk, or 18 weeks if in class simultaneously
- [ ] **Semantic Scholar API key** (optional but raises rate limit 1 → 100 req/sec)
- [ ] **Obsidian** installed, vault initialized at `SciRag/wiki/`
- [ ] Any prior SciRAG v1 code to reuse?

## The Problem (1-paragraph)
Standard RAG fails on scientific papers in 4 ways: dense jargon breaks surface-similarity embeddings; citations create cross-paper dependencies single-chunk retrieval misses; tables/equations/figures carry meaning text pipelines discard; research questions are synthesis-oriented not factoid. SciRAG v2 addresses each as a benchmarked component, then uses those components as the compilation engine for a living LLM-maintained knowledge base.

## Architecture
```
Papers (PDF/arXiv)
  → Grobid → TEI XML (sections, refs, figures)
  → Section-aware chunking → SPECTER2 → Qdrant (per-section collections)
  → Semantic Scholar → NetworkX citation graph
  → LLM compiler (via pluggable client) → Obsidian wiki (papers/, concepts/, indices/)
  → Query engine: classifier → section-routed retrieval + citation expansion
                → LangGraph multi-hop → NLI claim verification → wiki linter
```

## LLM Abstraction (pluggable)
All LLM calls go through `src/llm/client.py`. Provider selected by env var:
```
SCIRAG_LLM_PROVIDER=ollama       # default
SCIRAG_LLM_PROVIDER=anthropic    # swap to Claude Sonnet
SCIRAG_LLM_PROVIDER=openai       # swap to GPT-4o
```
Interface: `generate(system, user, max_tokens, temperature, response_format) -> str`. Providers implement the same signature. Adding Claude later = one new file + env var, no callsite changes.

**Quality tradeoff to eyes-open on**: Llama-3.1-8B is strong for claim verification (NLI-style) but weaker than Sonnet for concept article synthesis (Week 10). If compilation quality is visibly poor on manual review of first 10 papers, switch the compile step to a stronger local model (Qwen2.5-14B-Instruct Q4 ~9GB, fits on 16GB if Grobid is down) or flip the provider env var to Anthropic just for compilation.

## Components & Targets

| # | Component | Metric | Target |
|---|---|---|---|
| 1 | Section-aware chunking | recall@5 vs flat | **+15–25%** |
| 2 | Citation-graph expansion | answer F1 | **+5–12%** |
| 3 | Query classifier (DistilBERT, 4-class) | accuracy | 82–88% |
| 4 | Multi-hop reasoning (LangGraph) | coverage vs single-hop | **+20–35%** |
| 5 | Claim verification (NLI, DeBERTa fine-tuned on SciFact) | SciFact label acc | **75–80%** (vs ~72% baseline) |
| 6 | Wiki compilation | papers w/ summaries | **>95%** |
| 7 | Wiki link integrity | resolving wikilinks | **100%** |

## Ablation Table (Week 12 deliverable)
| Config | Description |
|---|---|
| A | Flat chunking, no routing, no expansion (pure baseline) |
| B | A + section chunking + routing |
| C | B + citation expansion |
| D | B + multi-hop reasoning |
| E | Full system (B + C + D + verification) |

## Datasets
- **QASPER** (1,585 NLP papers, 5,049 Q&A) — primary retrieval benchmark
- **SciFact** (1,409 claims w/ sentence-level evidence) — claim verification
- **SciQ** (13,679 science Q&A) — secondary factoid routing data
- **Domain corpus** — 50–100 curated papers (target depends on domain decision above)
- **S2ORC 100K subset** — optional; prefer lazy S2 API calls (simpler, incremental cache)

## Timeline

### Phase 1 — Foundation (Weeks 1–3)
- **Week 1**: Docker (Grobid, Qdrant, Ollama), S2 API key, download QASPER + SciFact, curate 50 seed papers, init Obsidian vault. *Goal: Grobid processing 50 test papers.*
- **Week 2**: Full QASPER PDF → Grobid. Build citation graph via S2 → SQLite cache. Log coverage rates. Version datasets (DVC or JSON manifests).
- **Week 3** ⚠️ **do not compress**: Flat-chunking baseline (SPECTER2 + FAISS). Record baseline recall@5 and F1. First Claude-compiled paper summaries for 10 papers. *Goal: baseline numbers locked, first wiki content in Obsidian.*

### Phase 2 — Retrieval Components (Weeks 4–8)
- **Week 4** — Section-aware chunking: Grobid XML → 6 canonical section types → per-section Qdrant collections + query router.
- **Week 5** — Section chunking polish: train DistilBERT query classifier on QASPER, confidence-gated fallback. **Ablation A vs B**. Freeze.
- **Week 6** — Citation-graph retrieval: 1-hop expansion, MiniLM-L6-v2 cross-encoder re-ranking, latency < 500ms. **Ablation B vs C**.
- **Week 7** — Multi-hop reasoning: LangGraph state machine, query decomposition (2–4 sub-questions, cosine-dedup > 0.85), context compression.
- **Week 8** — Multi-hop polish: inline `[S1][S2]` attribution, coverage metric, **ablation B vs D**, error analysis on 30 failures. Freeze.

### Phase 3 — Verification + Compilation (Weeks 9–11)
- **Week 9** — Claim verification: LLM claim extraction (3–8 atomic claims per answer), regex numerical detection, BM25 evidence retrieval, NLI fine-tune on SciFact (3–5 epochs, +8–12pp over zero-shot).
- **Week 10** — Wiki compilation engine: paper summarizer, concept article synthesizer, auto-generated indices (INDEX, GLOSSARY, TIMELINE, GRAPH, QUESTIONS), incremental compilation loop. *Goal: 50+ paper summaries, 15–20 concept articles.* **Manual-review first 10 summaries for quality**; if Llama-8B output is weak, upgrade local model or flip provider env var.
- **Week 11** — Wiki linting + search: claim consistency, link integrity, staleness/gap/duplicate detection. Custom BM25 + SPECTER2 search engine (web UI + CLI).

### Phase 4 — Integration + Polish (Weeks 12–14)
- **Week 12** ⚠️ **do not compress**: Full SciFact eval, Streamlit verification UI, full ablation table A–E, wiki quality metrics, per-component error analysis.
- **Week 13**: Integration testing, latency measurement, 3-min demo video (paper → Grobid → wiki → Q&A with verification), README, clean repo.
- **Week 14**: 2-page technical writeup, resume bullets with real numbers, 60-sec verbal summary, Marp slides. Optional: arXiv workshop submission.

## Deliverables Checklist
- [ ] GitHub repo, clean structure, README
- [ ] QASPER baseline numbers (recall@5, F1)
- [ ] Ablation table A–E with per-component deltas
- [ ] SciFact claim verification accuracy
- [ ] Compiled Obsidian wiki (50+ paper summaries, 15–20 concept articles)
- [ ] Wiki lint report (all checks passing)
- [ ] Streamlit verification UI with claim color-coding
- [ ] Wiki search engine (web UI + CLI)
- [ ] 3-minute demo video
- [ ] 2-page technical writeup
- [ ] Resume bullets with real numbers
- [ ] Practiced 60-sec verbal summary

## Risk Register
| Risk | Mitigation |
|---|---|
| Scope creep from wiki features | Retrieval benchmarks are core. Cut wiki linting + search first if behind. |
| Grobid extraction failures | Track/report rate. Learned SPECTER2 fallback. Curated corpus = high success. |
| LLM compilation quality inconsistent | Fixed templates + few-shot. Manual review of first 10 papers. Use Claude Sonnet. |
| Citation graph sparsity | Report in-corpus coverage honestly. Frame as motivation for larger corpus. |
| "Obsidian feels non-technical" | Obsidian is just the viewer. Engineering is the pipeline. Swappable. |
| 14 weeks too ambitious with coursework | Extend to 18. Protect Weeks 3 and 12. |

## Resume Framing (What to Write)
- "Developed section-aware chunking with query routing over QASPER, improving retrieval recall@5 by X% vs flat-chunking baseline"
- "Built citation-graph neighborhood expansion using Semantic Scholar API, improving cross-paper answer F1 by X% on multi-reference QASPER questions"
- "Implemented scientific claim verification via NLI-based entailment (fine-tuned DeBERTa on SciFact), achieving X% label accuracy vs Y% zero-shot baseline"
- "Designed iterative multi-hop retrieval with LangGraph, increasing synthesis query coverage from X% to Y% on QASPER abstractive questions"
- "Built an LLM-compiled scientific knowledge base (N papers, M synthesized concept articles) with automated claim verification and wiki linting"
