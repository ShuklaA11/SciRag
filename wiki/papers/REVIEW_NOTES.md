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
