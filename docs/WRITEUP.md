# SciRAG — Benchmarked, Domain-Adaptive Scientific RAG with an Auditable Novelty Evaluator

**One line:** An open, benchmarked methods layer for scientific-paper RAG — section-aware retrieval, citation-graph expansion, multi-hop, and NLI claim verification, each measured as an ablation — extended with a domain-adaptive hub and an auditable idea-novelty evaluator validated by a temporal holdout.

All numbers below are re-aggregated from stored per-query results and regression-guarded by `tests/test_baseline_v2_manifest.py` (canonical embedder: BGE; retrieval benchmark: QASPER dev, n=925 questions with gold evidence).

---

## The problem

Standard RAG degrades on scientific papers in four ways: dense jargon breaks surface-similarity embeddings; citations create cross-paper dependencies single-chunk retrieval misses; research questions are synthesis-oriented, not factoid; and generated claims need verification against evidence, not just fluent restatement. SciRAG treats each as a **separately benchmarked component** rather than a monolithic pipeline, then reuses those components as the engine for a knowledge base and a research-ideation tool.

## System

```
Papers → Grobid → section-aware chunking → BGE/SPECTER2 embeddings → FAISS
       → Semantic Scholar → NetworkX citation graph
Query  → TF-IDF section router → retrieval (+ 1-hop citation expansion, cross-encoder rerank)
       → multi-hop decomposition (cosine-dedup > 0.85) → NLI claim verification (DeBERTa)
       → LLM-compiled wiki (57 paper summaries, concept articles, auto indices)
v3     → domain profiles (NLP/ML, biomedical) → SQLite hub → idea-novelty evaluator
```

The v3 layer is decoupled: all pipeline behavior that is domain-specific (section taxonomy, embedder, verifier, data sources) is captured in a `DomainProfile`, so retargeting to a new field swaps a profile rather than editing code. Persistence and logic live in a pure-Python core (`src/hub`, SQLite); the Streamlit view is a thin renderer.

## Benchmarked results

### Retrieval ablation (QASPER dev, fuzzy recall@5, n=925)

| Config | Description | recall@5 |
|--------|-------------|----------|
| A | Flat chunking, no routing, no expansion (baseline) | 0.738 |
| B | A + section chunking + TF-IDF router | 0.736 |
| B* | Oracle section routing (headroom ceiling) | 0.866 |
| C | B + 1-hop citation expansion + cross-encoder rerank | **0.771** |
| D | B + multi-hop decomposition | **0.772** |
| E | Full system (B + C + D stacked) | 0.692 |

**Headline finding — stacking is not monotonic.** The "full system" (E = 0.692) is *worse* than either component alone (C = 0.771, D = 0.772). Citation expansion is the culprit: on QASPER, gold evidence is almost always in-paper, so pulling 1-hop citation-neighbor chunks dilutes precision@5 and pushes gold below rank 5. Multi-hop adds nothing here (mean sub-questions 1.05 — QASPER questions are mostly atomic). **Takeaway: more components ≠ better; expansion's win is cross-paper F1, its intended purpose, not in-paper recall — so the honest "full system" for QASPER retrieval leaves expansion off.**

A second honest note: section routing (B ≈ A) is roughly neutral at the shipped confidence threshold, while the **oracle ceiling (0.866)** shows the headroom is in *routing accuracy*, not the section-aware architecture — the classifier, not the design, is the bottleneck.

### Claim verification (SciFact, n=340 pairs)

Zero-shot DeBERTa-v3 (MNLI/FEVER/ANLI) → SciFact labels: **accuracy 0.691, macro-F1 0.676.** The fine-tune (pre-registered targets: contradiction-recall ≥0.70, end-to-end@k5 ≥0.72) is **deferred, hardware-bound** — it NaNs on Apple MPS and needs CUDA. The training scaffolding ships and is tested; the deferral is documented rather than hidden.

### Idea-novelty evaluation (the differentiator)

The idea evaluator decomposes an idea into atomic claims, retrieves corpus evidence per claim, runs NLI, and buckets each claim **ENTAILED** (not novel) / **CONTRADICTED** (contrarian) / **NOVEL** (untested) — reported per-claim, never as a single (gameable) scalar. Retrieval + NLI compute the verdict; the LLM is a narrator only.

**Temporal-novelty validation** (QASPER, arXiv-year holdout, cutoff 2018, zero-shot NLI): contributions already in the corpus (≤2018, n=129) vs the next year (2019, n=114):

| Group | NOVEL rate | ENTAILED |
|-------|-----------|----------|
| In-corpus (≤2018) | 0.388 | 45% |
| Held-out (2019) | 0.605 | 20% |

**Novelty gap = +0.218**, in the pre-registered direction: next-year papers score more novel, and entailment drops 45%→20% (in-corpus papers are 2.2× more likely to be recognized). This is a directional novelty **proxy**, not an oracle — the claim unit is a paper title, so a shared ~40% NEI-noise floor sits under both groups; the *gap* is the signal, reported honestly as such. It is a number closed products in this space do not publish.

## Positioning

SciRAG is the **open, benchmarked, domain-adaptive** version of the tooling that agentic-science products (e.g. Claude Science) productize as closed systems — with measured ablations and a temporal-novelty eval that closed products don't disclose. The edge is transparency and auditability, not breadth.

## Honest limitations

- **NLI fine-tune deferred** (CUDA-only); shipped number is zero-shot 0.691.
- **Novelty is a directional proxy**, not ground-truth novelty (title-as-claim noise).
- **Wiki concept articles under target** (7 vs 15–20 planned); paper summaries met target (57).
- **Agentic brainstorm loop is a qualitative demo**, not a benchmark; exploration depth is bounded by the local 8B proposer.

---

## Resume bullets (real numbers)

- Built a **benchmarked scientific-RAG ablation** over QASPER (n=925): section-aware chunking, citation-graph expansion, cross-encoder reranking, and multi-hop decomposition — and **measured that naively stacking components *hurts*** (full system 0.692 recall@5 vs 0.772 best single component), tracing it to citation-expansion diluting in-paper recall.
- Implemented **scientific claim verification** via NLI entailment (DeBERTa-v3 → SciFact), **0.691 accuracy / 0.676 macro-F1 zero-shot**, with a documented, regression-guarded evaluation manifest.
- Designed an **auditable idea-novelty evaluator** (claim decomposition → evidence retrieval → NLI → entailed/contradicted/novel buckets) and **validated it with a temporal holdout: +0.218 novelty gap** between in-corpus and held-out-year papers (entailment 45%→20%).
- Built a **domain-adaptive architecture** (`DomainProfile`) validated across NLP/ML and biomedical, with a persistence-first hub (SQLite core + thin Streamlit view) and an agentic gap-discovery loop.
