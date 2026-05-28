"""LLM-backed concept article compiler for the SciRAG wiki.

Given a concept name + evidence ``[(arxiv_id, snippet), ...]`` from
``src/wiki/concept_extractor.py``, asks the LLM to synthesize a wiki
article with fixed sections (Definition, Origin, Key Papers, Variants,
Open Questions).

Quality contract mirrors ``src/wiki/summarizer.py``:
  * uses ONLY information present in the evidence snippets,
  * never invents arxiv_ids,
  * writes "Unknown" for fields it cannot determine,
  * returns JSON in strict mode and we structurally validate the keys.

Key Papers are auto-rendered as Obsidian wiki-links ``[[arxiv_id]]``
from the evidence, not from anything the LLM emits — this prevents
hallucinated paper citations by construction.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.llm.client import LLMClient
from src.wiki.concept_extractor import ConceptEvidence

MAX_EVIDENCE_PER_CONCEPT = 8
MAX_SNIPPET_CHARS = 240
NUM_CTX = 8192
MAX_TOKENS = 1024
TEMPERATURE = 0.2

REQUIRED_KEYS = ("definition", "origin", "variants", "open_questions")

SYSTEM_PROMPT = """You are a scientific wiki editor synthesizing a short concept article \
from evidence snippets across multiple papers. Return a JSON object with exactly these keys:
  definition      - one sentence defining the concept; max 40 words
  origin          - 2-3 sentences on where the concept comes from (which paper(s), what motivated it)
  variants        - 2-4 sentences on notable variants or alternative formulations seen across the evidence
  open_questions  - 1-2 sentences on unresolved tensions, limitations, or open research directions visible in the evidence

Use ONLY information present in the evidence snippets. Do NOT invent paper titles, \
arxiv ids, dataset names, or numbers. If a field cannot be supported by the evidence, \
write "Unknown". Output valid JSON only, no markdown fences, no prose before or after."""

MARKDOWN_TEMPLATE = """---
concept: {concept_yaml}
generated_by: {model}
generated_at: {timestamp}
status: {status}
depends_on: [{depends_on}]
---

# {concept}

## Definition
{definition}

## Origin
{origin}

## Key Papers
{key_papers}

## Variants
{variants}

## Open Questions
{open_questions}
"""


@dataclass(frozen=True)
class CompileResult:
    markdown: str
    status: str  # "ok" | "parse_error" | "empty_evidence"
    latency_ms: int
    raw_output: str | None


def _yaml_safe(s: str) -> str:
    return '"' + s.replace('"', '\\"') + '"'


def _strip_frontmatter_delimiters(s: str) -> str:
    return "\n".join(line for line in s.splitlines() if line.strip() != "---")


def _truncate_snippet(snippet: str) -> str:
    if len(snippet) <= MAX_SNIPPET_CHARS:
        return snippet
    return snippet[:MAX_SNIPPET_CHARS] + "..."


def _format_evidence_block(evidence: list[ConceptEvidence]) -> str:
    capped = evidence[:MAX_EVIDENCE_PER_CONCEPT]
    lines = []
    for e in capped:
        lines.append(f"- ({e.arxiv_id}) {_truncate_snippet(e.snippet)}")
    return "\n".join(lines)


def _render_key_papers(evidence: list[ConceptEvidence]) -> str:
    """Render Obsidian-style links from evidence arxiv_ids.

    Auto-generated (not LLM-emitted) to prevent fabricated citations.
    Preserves first-seen order, de-duplicates.
    """
    seen: set[str] = set()
    out: list[str] = []
    for e in evidence:
        if e.arxiv_id in seen:
            continue
        seen.add(e.arxiv_id)
        out.append(f"- [[{e.arxiv_id}]]")
    return "\n".join(out) if out else "Unknown"


def _depends_on_list(evidence: list[ConceptEvidence]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for e in evidence:
        if e.arxiv_id in seen:
            continue
        seen.add(e.arxiv_id)
        out.append(e.arxiv_id)
    return ", ".join(out)


def _parse_json_strict(raw: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if not all(k in obj for k in REQUIRED_KEYS):
        return None
    return obj


def _render_markdown(
    concept: str,
    model: str,
    fields: dict[str, str],
    key_papers: str,
    depends_on: str,
    status: str,
) -> str:
    return MARKDOWN_TEMPLATE.format(
        concept_yaml=_yaml_safe(concept),
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status=status,
        depends_on=depends_on,
        concept=concept,
        definition=_strip_frontmatter_delimiters(fields["definition"]),
        origin=_strip_frontmatter_delimiters(fields["origin"]),
        key_papers=key_papers,
        variants=_strip_frontmatter_delimiters(fields["variants"]),
        open_questions=_strip_frontmatter_delimiters(fields["open_questions"]),
    )


def compile_concept(
    concept: str,
    evidence: list[ConceptEvidence],
    llm_client: LLMClient,
    *,
    model_name: str = "unknown",
) -> CompileResult:
    """Compile one concept article. Returns CompileResult.

    Key Papers + frontmatter depends_on are derived from evidence
    arxiv_ids regardless of LLM output; only the prose fields come
    from the LLM and are validated via strict JSON parsing.
    """
    key_papers = _render_key_papers(evidence)
    depends_on = _depends_on_list(evidence)

    if not evidence:
        md = _render_markdown(
            concept=concept,
            model=model_name,
            fields={k: "Unknown" for k in REQUIRED_KEYS},
            key_papers="Unknown",
            depends_on="",
            status="empty_evidence",
        )
        return CompileResult(markdown=md, status="empty_evidence", latency_ms=0, raw_output=None)

    evidence_block = _format_evidence_block(evidence)
    user_prompt = (
        f"CONCEPT: {concept}\n\n"
        f"EVIDENCE (one bullet per paper, arxiv_id in parens):\n{evidence_block}\n\n"
        f"Return the JSON object now."
    )

    t0 = time.time()
    raw = llm_client.generate(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        response_format="json",
        num_ctx=NUM_CTX,
    )
    parsed = _parse_json_strict(raw)

    if parsed is None:
        retry_user = user_prompt + (
            "\n\nYour previous response was not valid JSON with the required keys. "
            "Return ONLY a JSON object with keys: "
            "definition, origin, variants, open_questions."
        )
        raw = llm_client.generate(
            system=SYSTEM_PROMPT,
            user=retry_user,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            response_format="json",
            num_ctx=NUM_CTX,
        )
        parsed = _parse_json_strict(raw)

    latency_ms = int((time.time() - t0) * 1000)

    if parsed is None:
        md = _render_markdown(
            concept=concept,
            model=model_name,
            fields={k: "[PARSE_ERROR]" for k in REQUIRED_KEYS},
            key_papers=key_papers,
            depends_on=depends_on,
            status="parse_error",
        )
        return CompileResult(
            markdown=md,
            status="parse_error",
            latency_ms=latency_ms,
            raw_output=raw[:500],
        )

    fields = {k: str(parsed[k]).strip() or "Unknown" for k in REQUIRED_KEYS}
    md = _render_markdown(
        concept=concept,
        model=model_name,
        fields=fields,
        key_papers=key_papers,
        depends_on=depends_on,
        status="ok",
    )
    return CompileResult(markdown=md, status="ok", latency_ms=latency_ms, raw_output=None)
