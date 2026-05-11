# Evaluation results

Per-component eval runs land here. Each run produces a `<name>.jsonl` (per-question rows) and `<name>_summary.json` (aggregates).

## Locked baselines

### `week7_multihop` — query decomposition + multi-hop retrieval (NEGATIVE on headline, micro-positive on subset)

| | |
|---|---|
| Index | `data/index/flat_bge_tagged` (same as W6) |
| Retriever | BGE flat + cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`) over multi-hop sub-questions |
| Decomposer | Llama-3.1-8B Q4 via Ollama, JSON output, 4-shot prompt |
| Gate | Regex pre-filter: skip LLM unless compound marker present (`and`/`or`/`both`/`,`+wh-word/`;`/multiple `?`) |
| Eval set | QASPER dev, 1,002 in-corpus questions |
| **Mean recall@5 fuzzy** | **0.7723** (vs W6 `0.7712`, **+0.11pt** — null) |
| Mean recall@5 strict | 0.2203 (vs W6 `0.2205`, −0.03pt) |
| Wall time | 41 min |
| Gate hit rate | 46/1002 questions (4.6%) actually invoked the LLM; 956 atomic-skipped |
| Sub-Q distribution | `{1: 956, 2: 39, 3: 6, 4: 1}` |
| Decompose latency | p50 `0.01ms` (gate skip), p95 `5.2s` (LLM path) |

**Subset breakdown** (via `scripts/analyze_multihop_subsets.py`):

| Subset | n | Fuzzy | Fuzzy Δ vs W6 | Strict | Strict Δ vs W6 |
|---|---:|---:|---:|---:|---:|
| Atomic (gate skip) | 927 | 0.7729 | 0.00 | 0.2128 | 0.00 |
| Compound (LLM fired) | 75 | 0.7656 | **+1.41pt** | 0.3095 | −0.35pt |

**Interpretation.** The architecture works where it fires — +1.4pt fuzzy on the 7.5% compound-marker subset — but QASPER is overwhelmingly within-paper extractive QA, so the headline number doesn't move. Pre-registered success criterion (≥+2pt fuzzy overall, no >1pt strict regression) was **not met**; result is logged as **negative** but with a clean micro-positive on the targeted subset. Atomic-subset deltas are exactly 0.00 on both metrics, confirming the gate is a perfect no-op (no regression risk introduced for the 92.5% bypass path).

**Failure mode notes for future runs.** Early smoke (pre-gate, full LLM) showed Llama-3.1-8B (i) leaking the system prompt into outputs, (ii) hallucinating outside-paper context (e.g. Reddit trivia), and (iii) force-decomposing atomic yes/no questions. The heuristic gate eliminates (iii) on 92.5% of questions; the few-shot prompt mitigates (i) and (ii) on the remainder. If multi-hop is revisited on a less-atomic dataset (HotpotQA, QASPER-abstractive subset), consider flipping `--multihop-llm-provider anthropic` for higher-quality decomposition.

---


### `week3_flat_baseline` — flat-chunking RAG anchor

| | |
|---|---|
| Commit | `e30b48b` |
| Index | `data/index/flat` (14,496 chunks over 1,166 QASPER papers) |
| Retriever | SPECTER2 + FAISS IndexFlatIP, paper-scoped top-5 |
| Reader | Llama-3.1-8B Q4_K_M via Ollama (`num_ctx=4096`) |
| Eval set | QASPER dev, 1,002 in-corpus questions (3 skipped) |
| **Mean recall@5** | **0.182** |
| **Mean answer F1** | **0.287** |
| Recall denominator | 925 questions with non-empty highlighted evidence |
| Wall time | 5h 0min |

This is the deliberately-weak flat-chunking anchor that every Phase 2 component (section-aware chunking, citation expansion, multi-hop) measures deltas from.

## Caveats — read before quoting these numbers externally

1. **Paper-scoped retrieval.** Each question retrieves only from its own paper's chunks, not the full 1,166-paper corpus. This is correct for QASPER (within-paper QA) but is more permissive than published "open-corpus" flat baselines, so F1 is mildly inflated relative to those.

2. **BIBREF tokens stripped upstream.** `grobid_client.extract_sections` drops `<ref type="bibr">` elements, so chunks contain zero `BIBREFn` tokens. QASPER gold evidence often anchors on these citations as substring matches, capping substring-match recall by an estimated 3–5 percentage points. Fixable; not fixed yet.

3. **Substring-match recall is brittle.** The standard QASPER recall metric (gold sentence ⊆ retrieved chunk after whitespace normalization) misses semantically-correct retrievals when sentence boundaries differ. F1 is the more robust signal.

4. **Small quantized reader.** Llama-3.1-8B Q4 is far below QASPER paper SOTA setups (LED, GPT-class). Direct comparison to published numbers (e.g. Dasigi 2021 F1 ~30–35%) is not apples-to-apples — they read the full paper, no retrieval bottleneck.

## Reproducing

```bash
# 1. Rebuild the flat index (~10 min)
python scripts/build_flat_index.py --rebuild

# 2. Run the full eval (~5 hours; resumable via JSONL)
python scripts/run_qasper_baseline.py --run-name week3_flat_baseline --rebuild
```
