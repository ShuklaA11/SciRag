# First-10 Summary Quality Review

**Run:** _(fill in git commit short hash)_
**Model:** _(e.g. llama3.1:8b)_
**Date:** _(YYYY-MM-DD)_

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

- [ ] **ACCEPT** — Llama-3.1-8B is good enough. Proceed to full wiki compilation in Week 10.
- [ ] **UPGRADE LOCAL** — Swap to Qwen2.5-14B-Instruct Q4 (~9 GB, fits with Grobid down).
- [ ] **UPGRADE REMOTE** — Set `SCIRAG_LLM_PROVIDER=anthropic` for the compilation step only.

## Review criteria (rate each paper 1–5)

- **Accuracy** — nothing invented, matches the source paper
- **Coverage** — hits problem, method, results, limitations
- **Concision** — no padding, no direct quotes, no filler
- **Numerical correctness** — any number cited matches the paper

## Canonical papers

### 1. attention_is_all_you_need
_(Ground truth known: Vaswani et al. 2017, Transformer architecture, BLEU 28.4 EN-DE, 41.8 EN-FR on WMT14.)_

- Accuracy: /5
- Coverage: /5
- Concision: /5
- Numerical correctness: /5
- Hallucinations spotted:
- Notes:

### 2. bert
_(Ground truth known: Devlin et al. 2018, MLM + NSP pretraining, 11 GLUE tasks.)_

- Accuracy: /5
- Coverage: /5
- Concision: /5
- Numerical correctness: /5
- Hallucinations spotted:
- Notes:

### 3. elmo
_(Ground truth known: Peters et al. 2018, biLM contextualized embeddings.)_

- Accuracy: /5
- Coverage: /5
- Concision: /5
- Numerical correctness: /5
- Hallucinations spotted:
- Notes:

### 4. gpt2
_(Ground truth known: Radford et al. 2019, large-scale language modeling.)_

- Accuracy: /5
- Coverage: /5
- Concision: /5
- Numerical correctness: /5
- Hallucinations spotted:
- Notes:

### 5. scibert
_(Ground truth known: Beltagy et al. 2019, BERT pretrained on scientific corpus.)_

- Accuracy: /5
- Coverage: /5
- Concision: /5
- Numerical correctness: /5
- Hallucinations spotted:
- Notes:

## QASPER papers (first 5 alphabetical)

_(Fill in arxiv IDs after running compile_first_10.py. Ground truth not pre-known — judge accuracy by re-reading the source TEI XML where needed.)_

### 6. _(arxiv id)_
- Accuracy: /5
- Coverage: /5
- Concision: /5
- Numerical correctness: /5
- Hallucinations spotted:
- Notes:

### 7. _(arxiv id)_
- Accuracy: /5
- Coverage: /5
- Concision: /5
- Numerical correctness: /5
- Hallucinations spotted:
- Notes:

### 8. _(arxiv id)_
- Accuracy: /5
- Coverage: /5
- Concision: /5
- Numerical correctness: /5
- Hallucinations spotted:
- Notes:

### 9. _(arxiv id)_
- Accuracy: /5
- Coverage: /5
- Concision: /5
- Numerical correctness: /5
- Hallucinations spotted:
- Notes:

### 10. _(arxiv id)_
- Accuracy: /5
- Coverage: /5
- Concision: /5
- Numerical correctness: /5
- Hallucinations spotted:
- Notes:

## Aggregate scores

- Mean accuracy: /5
- Mean coverage: /5
- Mean concision: /5
- Mean numerical correctness: /5
- Parse errors: /10
- Empty TEI: /10

## Final notes

_(Free-form: what the model gets right, where it drifts, whether section
headings need tweaking for Week 10 compilation, any prompt changes to
make before running over all 1,166 papers.)_
