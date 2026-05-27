# Evaluation results

Per-component eval runs land here. Each run produces a `<name>.jsonl` (per-question rows) and `<name>_summary.json` (aggregates).

## Locked baselines

### `week9_scifact_finetune` — SciFact NLI fine-tune (DEFERRED, infrastructure-bound)

| | |
|---|---|
| Target metric | CONTRADICT recall on k=5 hit subset: 0.574 -> >= 0.70 (pre-registered in SB9.2) |
| Secondary    | End-to-end accuracy at k=5: 0.659 -> >= 0.72 |
| Plan         | Fine-tune `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` on SciFact train (~900 pairs, 4 epochs, AdamW, stratified 10% val split) |
| Status       | **Not run.** Scaffolding shipped in `scripts/train_nli.py` + `tests/test_train_nli.py`; final fine-tune deferred to GPU. |

**Why deferred.** Two independent failure modes on M1 Pro local hardware:

1. **MPS numerical instability.** DeBERTa-v2 disentangled attention produces `grad_norm=nan` within ~20 training steps on Apple Silicon MPS, regardless of learning rate (tested 5e-6, 1e-5, 2e-5) or gradient clipping (`max_grad_norm=1.0`). The forward pass is stable — eval/inference on MPS works (this is what the SB9.1 zero-shot run does). Only backprop blows up.

2. **CPU is too slow.** A single backward step on DeBERTa-v3-base at seq_len=512 measures ~4 minutes on CPU. A full 4-epoch run on 900 train pairs (batch=4 → ~900 steps) extrapolates to **~60 hours** — incompatible with iterative experimentation.

Both findings are reproducible from the diagnostics in `scripts/train_nli.py` (forward pass returns sane loss ~0.88; backward + AdamW step hangs at ~4 min/step on CPU and NaNs out at step ~20 on MPS).

**What ships anyway.**
* `scripts/train_nli.py` — Trainer scaffolding (CLI, dataset builder, stratified split, `compute_metrics` closure, label-map inversion). Tested via `tests/test_train_nli.py` (7 unit tests; heavy end-to-end gated behind `SCIRAG_RUN_HEAVY=1`).
* The data prep pipeline is hardware-independent: stratified train/val split keeps all 3 classes in val (SUPPORT 50:1 majority would otherwise dominate at random sampling), label IDs are mapped from SciFact strings via the same `_build_label_map` used by SB9.1 inference, so the trained checkpoint loads through `NLIClassifier(model_name=<path>)` with zero caller changes.
* Pre-registered success metrics are locked here so a future GPU run cannot retroactively pick a friendlier target.

**Honest framing.** The Week 9 deliverable is the *retrieval-as-bottleneck* attribution from SB9.1 + SB9.2 (oracle 0.691 → BM25 k=5 0.659, with -3.2pp fully explained by miss-as-NEI). The fine-tune was meant to lift the NLI ceiling on the hit subset; that lift is the natural continuation of this work on appropriate hardware. PLAN.md "Phase 3 may need cloud compute" anticipated this exact case.

---


### `week9_scifact_bm25` — SciFact end-to-end: BM25 retrieval + zero-shot NLI (k-sweep)

| | |
|---|---|
| Eval set | SciFact dev, 340 `(claim, cited_doc)` pairs from 300 claims |
| Index | BM25 (rank-bm25 Okapi) over title + abstract, all 5,183 corpus docs |
| Tokeniser | Lowercase + alnum-word split (no stemming, no stopword removal) |
| NLI | Same model + threshold as SB9.1 oracle (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, NEI threshold 0.5) |
| Miss policy | If gold cited_doc ∉ top-k retrieved → predict NEI (honest abstention) |
| Index build | 0.7s on M1 Pro |
| Retrieval time | ~6s for all 300 claims at any k |
| NLI time | 20-35s on MPS, scaling with hits per claim |

**k-sweep headline numbers**

| k | retrieval recall@k | end-to-end acc | hit-only acc | macro F1 |
|---:|---:|---:|---:|---:|
| 1  | 0.462 | 0.609 | 0.643 | 0.564 |
| 3  | 0.629 | 0.635 | 0.640 | 0.600 |
| 5  | **0.694** | **0.659** | **0.652** | **0.628** |
| 10 | 0.756 | 0.671 | 0.658 | 0.643 |
| oracle (SB9.1) | 1.000 | 0.691 | 0.691 | 0.676 |

**Per-class retrieval recall** (BM25 is uneven across gold classes)

| gold | recall@5 | recall@10 |
|---|---:|---:|
| SUPPORT (138 pairs) | 0.877 | 0.920 |
| CONTRADICT (71 pairs) | 0.761 | 0.817 |
| **NEI (131 pairs)** | **0.466** | **0.550** |

NEI pairs are claims cited to docs that were *not* annotated as evidence — the cited doc is only tangentially related, so BM25 with claim-query naturally doesn't find it. On the pairs that matter for actual verification (SUPPORT + CONTRADICT), recall@5 is **0.838** and recall@10 is **0.885**.

**Failure attribution (apples-to-apples).** Oracle (SB9.1) accuracy restricted to the k=5 hit subset is **0.6525 — identical to BM25 hit-only accuracy (0.6525, delta = 0.0000)**. When BM25 finds the gold doc, the retrieved abstract IS the cited abstract, and NLI behaves identically. *All* of SB9.2's accuracy drop vs SB9.1 oracle comes from retrieval-miss → NEI substitution, none from "BM25 surfaced a worse doc."

**Confusion matrix on k=5 hit subset** (NLI failures given retrieval succeeded)

| gold \\ pred | SUPPORT | CONTRADICT | NEI |
|---|---:|---:|---:|
| SUPPORT (121)   | 64.5% | 10.7% | **24.8%** |
| CONTRADICT (54) | **20.4%** | 57.4% | 22.2% |
| NEI (61)        | 16.4% | 9.8%  | 73.8% |

Two dominant model failure modes: SUPPORT→NEI (under-confident entailment, 25% leak) and CONTRADICT→SUPPORT (sign reversal, 20% leak). The latter is the deeper bug — vanilla MNLI/FEVER training does not teach the negation cues used in scientific abstracts. **SB9.3 fine-tuning targets CONTRADICT recall specifically** (currently 0.574 hit-only) as its pre-registered success metric.

**Reproducing**

```bash
python scripts/run_scifact_with_retrieval.py --run-name week9_scifact_bm25 --k-sweep 1,3,5,10
python scripts/analyze_scifact_subsets.py
```

Subset analysis (`scripts/analyze_scifact_subsets.py`) recomputes the per-class retrieval recall, confusion matrix, and oracle-on-hit-subset comparison directly from the JSONLs — no model load needed.

---


### `week9_scifact_zeroshot` — SciFact claim verification, zero-shot NLI w/ oracle evidence

| | |
|---|---|
| Commit | `519ce92` |
| Eval set | SciFact dev, 340 `(claim, cited_doc)` pairs from 300 claims |
| Premise | Full abstract of the cited doc (oracle — no retrieval) |
| Hypothesis | Claim text |
| Model | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (~750MB), MPS |
| Label map | HF `id2label` -> SciFact: ENTAIL→SUPPORT, NEUTRAL→NEI, CONTRADICT→CONTRADICT |
| NEI gate | If `max(P(SUP), P(CON)) < 0.5` -> NEI (untuned default) |
| **Label accuracy** | **0.691** |
| Macro F1 | 0.676 |
| Per-class F1 | SUPPORT 0.690 / CONTRADICT 0.603 / NEI 0.734 |
| Gold dist | SUPPORT 138 / CONTRADICT 71 / NEI 131 |
| Pred dist | SUPPORT 117 / CONTRADICT 65 / NEI 158 |
| Wall time | 43s on M1 Pro MPS |

**Interpretation.** Lands slightly below the PLAN.md ~72% zero-shot anchor — expected for a FEVER-ANLI checkpoint that has never seen scientific abstracts. Per-class F1 is balanced (no collapse onto a single class); CONTRADICT is hardest (0.60), which is typical for scientific NLI where refuting a claim requires fine-grained quantitative reasoning. NEI is mildly over-predicted (158 vs 131 gold), suggesting the 0.5 threshold is slightly too conservative; **threshold tuning is deliberately deferred to SB9.3** so we don't cherry-pick on dev. This is the anchor every Week 9 follow-up (SB9.2 BM25 retrieval, SB9.3 SciFact fine-tune) measures deltas from.

**Caveats — read before quoting externally**

1. **Oracle premise.** Each pair uses the full cited abstract as the premise. Real-world claim verification has to first *find* the abstract; SB9.2 measures the BM25-retrieval gap on the full 5,183-doc corpus.
2. **Threshold not tuned.** 0.5 is the unprincipled default. SB9.3 will tune on train.
3. **3-class accuracy on 3-class gold.** Standard SciFact has more nuanced abstract-level + sentence-level F1 (Wadden 2020). Our metric is simpler and not directly comparable to leaderboard numbers; it is comparable across our own Week 9 sub-tasks.
4. **Single deterministic forward pass.** No seed variance to report.

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
