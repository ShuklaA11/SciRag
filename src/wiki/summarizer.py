"""LLM-backed paper summarizer for the wiki compilation pipeline.

Reads a Grobid TEI XML, sends title + abstract + section text to the
configured LLM in JSON mode, and returns a markdown summary with fixed
sections (TL;DR, Problem, Method, Results, Limitations).

Quality contract: uses only information present in the paper excerpts,
never invents numbers or citations, writes 'Unknown' for fields that
cannot be determined. Numerical hallucination is the headline failure
mode — caught at the wiki/papers/REVIEW_NOTES.md gate after running
scripts/compile_first_10.py.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from src.llm.client import LLMClient
from src.pipeline.grobid_client import extract_abstract, extract_sections, extract_title

MAX_BODY_CHARS = 6000
NUM_CTX = 8192
MAX_SUMMARY_TOKENS = 1024
TEMPERATURE = 0.2

REQUIRED_KEYS = ("tldr", "problem", "method", "results", "limitations")

SYSTEM_PROMPT = """You are a scientific paper summarizer. Read the paper excerpts below and \
return a JSON object with exactly these keys:
  tldr        - one sentence, max 30 words
  problem     - 2-3 sentences on the problem the paper addresses
  method      - 3-5 sentences on the key technical approach
  results     - 2-3 sentences on the headline findings; include numbers when the excerpts give them
  limitations - 1-2 sentences on stated or obvious limitations; "Unknown" if not stated

Use ONLY information present in the excerpts. Never invent numbers, datasets, \
citations, or methods. If a field cannot be determined from the excerpts, write "Unknown". \
Output valid JSON only, no markdown fences, no prose before or after."""

MARKDOWN_TEMPLATE = """---
arxiv_id: {arxiv_id}
title: {title_yaml}
generated_by: {model}
generated_at: {timestamp}
status: {status}
---

# {title}

## TL;DR
{tldr}

## Problem
{problem}

## Method
{method}

## Results
{results}

## Limitations
{limitations}
"""


def _build_paper_excerpt(tei_xml: str) -> tuple[str, str]:
    """Return (title, user_prompt_body) truncated to MAX_BODY_CHARS."""
    title = extract_title(tei_xml)
    abstract = extract_abstract(tei_xml)
    sections = extract_sections(tei_xml)

    parts: list[str] = [f"TITLE: {title}"]
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")

    budget_left = MAX_BODY_CHARS - sum(len(p) for p in parts)
    for s in sections:
        head = s["head"] if s["head"] != "[untitled]" else "SECTION"
        block = f"\n{head}:\n{s['text']}"
        if len(block) > budget_left:
            block = block[:budget_left] + " [...]"
            parts.append(block)
            break
        parts.append(block)
        budget_left -= len(block)
        if budget_left <= 0:
            break

    return title, "\n".join(parts)


def _strip_frontmatter_delimiters(s: str) -> str:
    """Prevent the LLM's output from breaking our YAML frontmatter."""
    return "\n".join(line for line in s.splitlines() if line.strip() != "---")


def _yaml_safe_title(title: str) -> str:
    escaped = title.replace('"', '\\"')
    return f'"{escaped}"'


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
    arxiv_id: str,
    title: str,
    model: str,
    fields: dict[str, str],
    status: str,
) -> str:
    return MARKDOWN_TEMPLATE.format(
        arxiv_id=arxiv_id,
        title_yaml=_yaml_safe_title(title),
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status=status,
        title=title,
        tldr=_strip_frontmatter_delimiters(fields["tldr"]),
        problem=_strip_frontmatter_delimiters(fields["problem"]),
        method=_strip_frontmatter_delimiters(fields["method"]),
        results=_strip_frontmatter_delimiters(fields["results"]),
        limitations=_strip_frontmatter_delimiters(fields["limitations"]),
    )


def summarize_paper(
    tei_xml: str,
    arxiv_id: str,
    llm_client: LLMClient,
    *,
    model_name: str = "unknown",
) -> dict[str, Any]:
    """Summarize one paper. Returns a dict with keys:
        markdown    - the rendered markdown (always a string)
        status      - "ok" | "parse_error" | "empty_tei"
        latency_ms  - total LLM+parse latency
        raw_output  - None on ok, truncated raw string on parse_error
    """
    title, excerpt = _build_paper_excerpt(tei_xml)
    sections = extract_sections(tei_xml)
    if not sections and not extract_abstract(tei_xml):
        md = _render_markdown(
            arxiv_id=arxiv_id,
            title=title,
            model=model_name,
            fields={k: "Unknown" for k in REQUIRED_KEYS},
            status="empty_tei",
        )
        return {"markdown": md, "status": "empty_tei", "latency_ms": 0, "raw_output": None}

    user_prompt = f"{excerpt}\n\nReturn the JSON object now."
    t0 = time.time()
    raw = llm_client.generate(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=MAX_SUMMARY_TOKENS,
        temperature=TEMPERATURE,
        response_format="json",
        num_ctx=NUM_CTX,
    )
    parsed = _parse_json_strict(raw)

    if parsed is None:
        # retry once with a stricter reminder
        retry_user = user_prompt + (
            "\n\nYour previous response was not valid JSON with the required keys. "
            "Return ONLY a JSON object with keys: tldr, problem, method, results, limitations."
        )
        raw = llm_client.generate(
            system=SYSTEM_PROMPT,
            user=retry_user,
            max_tokens=MAX_SUMMARY_TOKENS,
            temperature=0.0,
            response_format="json",
            num_ctx=NUM_CTX,
        )
        parsed = _parse_json_strict(raw)

    latency_ms = int((time.time() - t0) * 1000)

    if parsed is None:
        md = _render_markdown(
            arxiv_id=arxiv_id,
            title=title,
            model=model_name,
            fields={k: "[PARSE_ERROR]" for k in REQUIRED_KEYS},
            status="parse_error",
        )
        return {
            "markdown": md,
            "status": "parse_error",
            "latency_ms": latency_ms,
            "raw_output": raw[:500],
        }

    fields = {k: str(parsed[k]).strip() or "Unknown" for k in REQUIRED_KEYS}
    md = _render_markdown(
        arxiv_id=arxiv_id,
        title=title,
        model=model_name,
        fields=fields,
        status="ok",
    )
    return {"markdown": md, "status": "ok", "latency_ms": latency_ms, "raw_output": None}
