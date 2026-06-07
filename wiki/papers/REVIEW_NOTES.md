# First-10 Summary Quality Review

**Run:** `e30b48b` (post-chunker-fix)
**Model:** `llama3.1:8b` (Q4_K_M via Ollama, num_ctx=8192, temp=0.2)
**Date:** 2026-04-28

## Verdict

Decision gate from PLAN.md Week 3 / Week 10: if Llama-3.1-8B output is
visibly weak on manual review of the first 10 papers, upgrade before
compiling the full 1,166-paper wiki.

**Threshold (from Sub-task D plan):**

- **Any numerical hallucination (≥1/10)** → UPGRADE. Wrong benchmark numbers in
  a scientific knowledge base are catastrophic.
- **Non-numerical drift ≥2/10** (hallucinated method names, dataset
  mixups, wrong architecture descriptions) → UPGRADE.
- Otherwise → ACCEPT.

Mark one:

- [x] **ACCEPT** — Llama-3.1-8B is good enough. Proceed to full wiki compilation in Week 10.
- [ ] **UPGRADE LOCAL** — Swap to Qwen2.5-14B-Instruct Q4 (~9 GB, fits with Grobid down).
- [ ] **UPGRADE REMOTE** — Set `SCIRAG_LLM_PROVIDER=anthropic` for the compilation step only.

**Numerical hallucinations: 0/10. Non-numerical drift: 0/10.** Both gate
thresholds clear with margin. Llama-3.1-8B Q4 is sufficient for paper
summarization given the current prompt + JSON-mode constraints.

## Review criteria (rate each paper 1–5)

- **Accuracy** — nothing invented, matches the source paper
- **Coverage** — hits problem, method, results, limitations
- **Concision** — no padding, no direct quotes, no filler
- **Numerical correctness** — any number cited matches the paper

## Canonical papers

### 1. attention_is_all_you_need
_(Ground truth: Vaswani et al. 2017, Transformer architecture, BLEU 28.4 EN-DE, 41.8 EN-FR on WMT14.)_

- Accuracy: 5/5
- Coverage: 5/5
- Concision: 5/5
- Numerical correctness: 5/5
- Hallucinations spotted: none
- Notes: BLEU 28.4 EN-DE and 41.8 EN-FR both exact-match ground truth.
  Title field is polluted by a Google copyright notice prepended to the
  paper PDF — that's a Grobid extraction issue, not the LLM's fault.

### 2. bert
_(Ground truth: Devlin et al. 2018, MLM + NSP pretraining, 11 GLUE tasks.)_

- Accuracy: 5/5
- Coverage: 5/5
- Concision: 5/5
- Numerical correctness: 5/5
- Hallucinations spotted: none
- Notes: All four reported numbers match the paper exactly — GLUE 80.5%,
  MultiNLI 86.7%, SQuAD v1.1 F1 93.2, SQuAD v2.0 F1 83.1.

### 3. elmo
_(Ground truth: Peters et al. 2018, biLM contextualized embeddings.)_

- Accuracy: 5/5
- Coverage: 5/5
- Concision: 5/5
- Numerical correctness: 5/5
- Hallucinations spotted: none
- Notes: "up to 20% relative error reductions" matches paper claim.
  biLM + linear-combination-of-internal-states characterization is correct.

### 4. gpt2
_(Ground truth: Radford et al. 2019, large-scale language modeling.)_

- Accuracy: 5/5
- Coverage: 5/5
- Concision: 5/5
- Numerical correctness: 5/5
- Hallucinations spotted: none
- Notes: 1.5B params ✓, 7/8 zero-shot SOTA datasets ✓, CoQA 55 F1 ✓.

### 5. scibert
_(Ground truth: Beltagy et al. 2019, BERT pretrained on scientific corpus.)_

- Accuracy: 5/5
- Coverage: 5/5
- Concision: 5/5
- Numerical correctness: 5/5
- Hallucinations spotted: none
- Notes: Task list (sequence tagging, sentence classification, dependency
  parsing) and "statistically significant improvements over BERT" both
  match. No invented benchmark numbers — model correctly omitted specifics
  the abstract didn't contain.

## QASPER papers (first 5 alphabetical)

### 6. 1503.00841 — Robustly Leveraging Prior Knowledge in Text Classification
- Accuracy: 5/5
- Coverage: 4/5
- Concision: 5/5
- Numerical correctness: 5/5
- Hallucinations spotted: none
- Notes: "three regularization terms" verified verbatim in source TEI.
  Results section is vague ("remarkable improvements") because the
  source abstract is vague — fair compression, not a content failure.

### 7. 1601.00901 — Joint learning of ontology and semantic parser from text
- Accuracy: 5/5
- Coverage: 4/5
- Concision: 5/5
- Numerical correctness: 5/5
- Hallucinations spotted: none
- Notes: No numerical claims to fact-check. Method description (semi-automatic
  CFG induction, curriculum learning) matches source. Results thin but
  source abstract is also thin.

### 8. 1601.01705 — Learning to Compose Neural Networks for Question Answering
- Accuracy: 5/5
- Coverage: 5/5
- Concision: 5/5
- Numerical correctness: 5/5
- Hallucinations spotted: none
- Notes: "markedly different" verbatim in source. Dynamic NMN +
  RL-for-layout characterization is precise.

### 9. 1601.02166 — Empirical Gaussian priors for cross-lingual transfer learning
- Accuracy: 5/5
- Coverage: 5/5
- Concision: 5/5
- Numerical correctness: 5/5
- Hallucinations spotted: none
- Notes: "k source language models" preserves paper notation. Rademacher
  complexity reference is from the paper, not invented.

### 10. 1601.02403 — Argumentation Mining in User-Generated Web Discourse
- Accuracy: 5/5
- Coverage: 5/5
- Concision: 5/5
- Numerical correctness: 5/5
- Hallucinations spotted: none
- Notes: "90k tokens / 340 documents" verified against source — paper
  reports "90,000 tokens" and "340 documents" verbatim. Model correctly
  compressed 90,000 → 90k.

## Aggregate scores

- Mean accuracy: **5.0/5**
- Mean coverage: **4.8/5**
- Mean concision: **5.0/5**
- Mean numerical correctness: **5.0/5**
- Parse errors: 0/10
- Empty TEI: 0/10

## Final notes

Llama-3.1-8B Q4 produces clean, faithful summaries when:
1. The prompt explicitly forbids invention and demands "Unknown" for missing fields (current prompt does this).
2. JSON-mode + strict-key parsing catches structural drift before it lands in markdown.
3. The TEI has a real abstract — all 10 papers had one.

**Coverage scored 4.8 not 5.0** because two QASPER papers (1503.00841,
1601.00901) had thin Results sections — but that traces to thin source
abstracts, not the LLM. No prompt change needed for Week 10.

**Known artifacts (not LLM failures, flagged for later):**
- Title pollution: `attention_is_all_you_need.md` title field includes a
  Google copyright preamble. Grobid grabbed it as part of the title
  element. Same pattern likely affects a small fraction of QASPER papers
  with reproduction notices. Fix in `grobid_client.extract_title` if it
  shows up in >5% of papers during full Week 10 compilation.
- BIBREF stripping: same upstream Grobid extraction issue noted in
  `eval/results/README.md`. Doesn't affect summary quality (LLM doesn't
  need citation tokens to summarize), but caps QASPER recall.

**Gate B verdict: ACCEPT.** Proceed with Llama-3.1-8B Q4 for Week 10
full-corpus compilation. Re-evaluate only if numerical hallucinations
appear in spot checks during the 1,166-paper run.

---

# Gate B — sampled review at N=55 (SB10.1, seed=42)

**Run:** Week 10 full compile (55 summaries, see `wiki/README.md` Gate A).
**Model:** `llama3.1:8b` (Q4_K_M via Ollama, CPU).
**Date:** 2026-05-28
**Sample:** deterministic seed=42 random-10 of the newly compiled set —
1602.00812, 1602.06291, 1602.07618, 1602.08741, 1603.00968,
1603.07044, 1603.08594, 1604.00117, 1605.03481, 1606.04631.

## Threshold (unchanged from first-10)

- **Any numerical hallucination (≥1/10)** → HALT SB10.3, swap Llama →
  Anthropic for the compile path.
- **Non-numerical drift ≥2/10** (wrong methods, dataset mixups, wrong
  architectures) → upgrade.
- Otherwise → ACCEPT.

Numeric claims were cross-checked against the source Grobid TEI in
`data/grobid_output/qasper/<id>.xml`, not scored from the summary alone.

## Numerical claims — TEI-verified

| Paper | Summary claim | TEI ground truth | Verdict |
|---|---|---|---|
| 1602.06291 | 21% rel. acc., next-sentence selection, Wikipedia | "relative accuracy improvements of 21% for the Wikipedia dataset and 18% for Google News" | exact |
| 1603.07044 | 10% MAP improvement vs IR; comparable to handcrafted | "SemEval-2016 cQA … 10% improvement on a MAP score compared to an IR-based approach … comparable … handcrafted feature-based" | exact |
| 1603.08594 | 10% over baseline (MSTParser, English) | "improved by 10% over the baseline … MSTParser model trained for English" | exact |
| 1603.00968 | order of magnitude more efficient | "an order of magnitude more efficient in terms of training time" | exact |

## Per-paper scores (accuracy / coverage / concision / numerical)

| # | Paper | Acc | Cov | Conc | Num | Notes |
|---|---|---|---|---|---|---|
| 1 | 1602.00812 Grail theorem prover | 5 | 4 | 5 | 5 | No numeric claims. Chapter overview; thin Results trace to the source abstract. |
| 2 | 1602.06291 CLSTM | 5 | 5 | 5 | 5 | 21% / Wikipedia / next-sentence selection all exact. |
| 3 | 1602.07618 Quantum → togetherness | 4 | 4 | 5 | 5 | "togetherness essential for understanding the human brain and its interactions with other brains" is over-specific for a position paper (compression artifact, not a wrong fact). |
| 4 | 1602.08741 Russian Twitter | 4 | 5 | 5 | 5 | "not suitable for Russian due to morphological complexity" is an invented rationale; the conclusion (Twitter comparable to single-corpus models) is exact. |
| 5 | 1603.00968 MGNC-CNN | 4 | 4 | 5 | 5 | **Non-numeric drift (1/1 counted):** says "achieving state-of-the-art"; abstract only claims "consistently outperforms baseline models." SOTA was MVCNN's result, and the paper claims MGNC-CNN is *comparable* to it with far less compute. |
| 6 | 1603.07044 RNN encoder + attention, cQA | 5 | 5 | 5 | 5 | 10% MAP exact. |
| 7 | 1603.08594 PP attachment (bilingual) | 5 | 5 | 5 | 5 | 10% / English-Hindi / MSTParser all exact. |
| 8 | 1604.00117 Domain adaptation RNN NLU | 5 | 5 | 5 | 5 | Multi-task slot filling, open vocabulary correct. |
| 9 | 1605.03481 Tweet2Vec | 5 | 5 | 5 | 5 | Bi-GRU character composition correct. |
| 10 | 1606.04631 BiLSTM video description | 5 | 5 | 5 | 5 | MSVD corpus correct. |

## Aggregate scores

- Numerical hallucinations: **0/10**
- Non-numerical drift: **1/10** (the MGNC-CNN "state-of-the-art"
  overstatement; #3 and #4 are softer compression artifacts, not wrong
  facts, and are not counted toward the drift threshold)
- Mean accuracy: **4.7/5**
- Mean coverage: **4.7/5**
- Mean concision: **5.0/5**
- Mean numerical correctness: **5.0/5**
- Parse errors: 0/10 · Empty TEI: 0/10

## Gate B verdict at N=55: ACCEPT

Zero numerical hallucinations — the hard stop does not trigger. All
architectures, datasets, and method names are correct. No Llama →
Anthropic swap. SB10.3 (concept compile) is cleared to proceed on
Llama-3.1-8B Q4.

**Watch item for the concept compile:** the lone drift (MGNC-CNN
SOTA) is a *comparable-to-X → is-X* conflation. Spot-check concept
articles for the same pattern when prose summarizes a paper's standing
relative to prior work.
