# SciRAG Wiki — Week 10 deliverables

LLM-compiled scientific knowledge base over NLP/ML papers. Built from
Grobid TEI XML extracts, summarized + cross-linked, indexed for
Obsidian-style browsing.

```
wiki/
├── papers/    one summary per arxiv id (frontmatter + 5 fixed sections)
├── concepts/  one article per cross-paper concept
├── indices/   INDEX / GLOSSARY / TIMELINE / QUESTIONS / GRAPH
├── .state.json  sha256 hashes for incremental recompilation
└── README.md  this file
```

## Pre-registered success metrics (locked 2026-05-27, before results)

Same lock-targets-before-results convention as `eval/results/README.md`.
Numbers in the **Result** column get backfilled as each gate completes;
the target column never moves.

### Gate A — paper summary coverage (SB10.1) — ✅ PASS

Compile run: `llama3.1:8b` via Ollama (CPU), 4h 22min wall time on the
41 newly compiled papers. 14 pre-existing summaries skipped.

| | Target | Result |
|---|---|---|
| Total summaries (5 canonical + 50 alphabetical QASPER) | ≥ 55 | **55** ✅ |
| Status=ok rate (newly compiled) | ≥ 95% | **41/41 = 100%** ✅ |
| Parse errors | 0 | **0** ✅ |
| Empty TEI | < 5% | **0/41 = 0%** ✅ |

### Gate B — sampled quality review (SB10.1)

Sample 10 of the 45 newly compiled papers (seed=42, deterministic via
`scripts/compile_papers.py --sample-review 10`). Apply the same rubric
as `wiki/papers/REVIEW_NOTES.md`:

| | Target | Result |
|---|---|---|
| Numerical hallucinations across the sample | 0 / 10 | _pending review_ |
| Non-numerical drift (wrong methods, dataset mixups) | ≤ 1 / 10 | _pending review_ |
| Mean accuracy | ≥ 4.5 / 5 | _pending review_ |
| Mean coverage | ≥ 4.5 / 5 | _pending review_ |

**Stop condition.** Any numerical hallucination triggers SB10.1 halt
and forces the Llama → Qwen-14B or Anthropic swap per the LLM-choice
decision in `wiki/papers/REVIEW_NOTES.md`.

### Gate C — concept article coverage (SB10.2 + SB10.3)

| | Target | Result |
|---|---|---|
| Concept candidates surfaced (`min_paper_count=2`) | ≥ 30 | _pending compile_ |
| Concept articles compiled | 15–20 | _pending_ |
| Articles citing ≥ 3 source papers (verified via grep on Key Papers section) | ≥ 90% | _pending_ |
| Articles with parse_error status | 0 | _pending_ |

### Gate D — wiki integrity (SB10.4) — ✅ PASS (papers); concepts pending compile

| | Target | Result |
|---|---|---|
| All 5 indices regenerate from scratch | yes | ✅ (0.02s on 55-paper snapshot) |
| Every `[[arxiv_id]]` link resolves to a file in `wiki/papers/` | 100% | ✅ (0 broken links over 55 papers, all indices) |
| Every `[[concept_slug]]` link resolves to a file in `wiki/concepts/` | 100% | ✅ vacuous (no concepts yet — re-run after SB10.3 compile) |
| GRAPH.json edges only point to existing nodes | 100% | ✅ (enforced in `render_graph`) |

### Gate E — incremental loop (SB10.5) — ✅ PASS

| | Target | Result |
|---|---|---|
| Re-run on unchanged TEIs hits 0 LLM calls | yes | ✅ (verified: 55/55 `[skip:bootstrap]`, n_ok=0, 0.1s elapsed vs 4h22m for the live compile) |
| Hash-changed paper triggers concept stale marker | yes | ✅ (test coverage in `tests/test_incremental.py`) |
| Pre-existing summaries adopted without recompile | yes | ✅ (bootstrap path verified end-to-end; `wiki/.state.json` now has 55 entries) |

## Anti-targets (things that fail the wiki even if metrics pass)

- Any `[[arxiv_id]]` link pointing to a paper not in `wiki/papers/` —
  this is a fabricated citation, the headline failure mode this wiki
  is designed to prevent.
- Any concept article whose `depends_on:` frontmatter lists ids not in
  the live paper set.
- Any summary or concept article with `## Method` (or equivalent
  section) shorter than 50 chars on a non-empty TEI — likely an LLM
  collapse.

## Reproducing

```bash
# 1. Stop Grobid (Ollama needs the RAM)
docker compose stop grobid

# 2. Warm Llama
ollama run llama3.1:8b   # type /bye after the model loads

# 3. Compile summaries (resumable; ~5 min/paper on M1 Pro CPU)
.venv/bin/python scripts/compile_papers.py --paper-set all --limit 50

# 4. List concept candidates, pick 15-20
.venv/bin/python scripts/compile_concepts.py --list-candidates --top-n 50 > /tmp/candidates.txt
$EDITOR /tmp/curated_concepts.txt   # one concept name per line

# 5. Compile concept articles (3-article quality gate built in)
.venv/bin/python scripts/compile_concepts.py --concepts-file /tmp/curated_concepts.txt

# 6. Rebuild indices
.venv/bin/python scripts/build_indices.py
```

## Caveats — read before quoting these numbers

1. **Llama-3.1-8B Q4 on CPU.** Summarizer runs without Metal
   acceleration on the local M1 Pro; latency is 60–500s per paper.
   Acceptable for one-shot compile but not for interactive iteration.
   Anthropic fallback wired via `SCIRAG_LLM_PROVIDER=anthropic` if
   spot checks during compile surface hallucinations.

2. **Concept extraction is regex-based.** No spaCy, no NLTK. The
   `extract_candidates` function trades recall for zero-dependency
   simplicity; expect some noise in the candidate list (e.g.,
   "The Transformer" as a phrase candidate) that the human review
   pass filters out.

3. **Wiki-lint formal pass is W11.** This README's Gate D is a
   grep-level integrity check, not a full link-and-schema validator.

4. **Quality gate at N=55 is sampled, not exhaustive.** The first 10
   summaries were hand-reviewed in
   `wiki/papers/REVIEW_NOTES.md`; the next 45 use a seed=42 random-10
   sample.

5. **`wiki/indices/` files are committed snapshots.** They are
   deterministic outputs of `scripts/build_indices.py`. Committed for
   review diff convenience; not authoritative — always rebuild from
   the live `papers/` + `concepts/` before relying on them.
