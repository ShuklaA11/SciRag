# Evaluation results

Per-component eval runs land here. Each run produces a `<name>.jsonl` (per-question rows) and `<name>_summary.json` (aggregates).

## Locked baselines

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
