"""Tests for src/wiki/concept_compiler.py.

LLM is fully mocked (FakeLLM). Heavy end-to-end against real Ollama is
gated behind SCIRAG_RUN_HEAVY=1 (same convention as
test_concept_extractor.py / test_nli_classifier.py).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest

from src.llm.client import LLMClient
from src.wiki.concept_compiler import (
    REQUIRED_KEYS,
    CompileResult,
    _format_evidence_block,
    _parse_json_strict,
    _render_key_papers,
    compile_concept,
)
from src.wiki.concept_extractor import ConceptEvidence


# --------------------------------------------------------------- fake LLM


@dataclass
class FakeLLM(LLMClient):
    """Returns a queued list of responses in order. If queue is exhausted,
    the last response is repeated."""

    responses: list[str]
    calls: list[tuple[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.calls = []

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: str | None = None,
        num_ctx: int | None = None,
    ) -> str:
        self.calls.append((system, user))
        if not self.responses:
            return ""
        if len(self.calls) <= len(self.responses):
            return self.responses[len(self.calls) - 1]
        return self.responses[-1]


def _valid_json(**overrides: str) -> str:
    body = {
        "definition": "A pretraining objective that masks tokens.",
        "origin": "Introduced by BERT.",
        "variants": "Span-MLM, whole-word masking.",
        "open_questions": "Optimal mask ratio.",
    }
    body.update(overrides)
    return json.dumps(body)


SAMPLE_EVIDENCE = [
    ConceptEvidence(arxiv_id="bert", snippet="...Masked Language Model objective..."),
    ConceptEvidence(arxiv_id="scibert", snippet="...uses MLM pretraining on scientific text..."),
]


# ---------------------------------------------------------------- happy path


def test_compile_concept_ok_renders_markdown_with_all_sections() -> None:
    llm = FakeLLM(responses=[_valid_json()])
    result = compile_concept("Masked Language Model", SAMPLE_EVIDENCE, llm, model_name="fake-1.0")
    assert result.status == "ok"
    md = result.markdown
    assert "# Masked Language Model" in md
    assert "## Definition" in md
    assert "## Origin" in md
    assert "## Key Papers" in md
    assert "## Variants" in md
    assert "## Open Questions" in md
    assert "[[bert]]" in md
    assert "[[scibert]]" in md
    assert "generated_by: fake-1.0" in md


def test_compile_concept_writes_frontmatter_depends_on() -> None:
    llm = FakeLLM(responses=[_valid_json()])
    result = compile_concept("MLM", SAMPLE_EVIDENCE, llm)
    assert "depends_on: [bert, scibert]" in result.markdown


def test_compile_concept_key_papers_dedupes_arxiv_ids() -> None:
    evidence = [
        ConceptEvidence(arxiv_id="bert", snippet="a"),
        ConceptEvidence(arxiv_id="bert", snippet="b"),
        ConceptEvidence(arxiv_id="elmo", snippet="c"),
    ]
    out = _render_key_papers(evidence)
    assert out.count("[[bert]]") == 1
    assert "[[elmo]]" in out


def test_compile_concept_passes_evidence_into_prompt() -> None:
    llm = FakeLLM(responses=[_valid_json()])
    compile_concept("X", SAMPLE_EVIDENCE, llm)
    _system, user = llm.calls[0]
    assert "CONCEPT: X" in user
    assert "(bert)" in user
    assert "(scibert)" in user
    assert "Masked Language Model objective" in user


# ----------------------------------------------------------------- empty path


def test_compile_concept_empty_evidence_returns_unknown_no_llm_call() -> None:
    llm = FakeLLM(responses=[_valid_json()])
    result = compile_concept("Orphan", [], llm)
    assert result.status == "empty_evidence"
    assert llm.calls == []  # never invoked
    assert "Unknown" in result.markdown
    # Key Papers also Unknown
    assert "## Key Papers\nUnknown" in result.markdown


# --------------------------------------------------------------- parse errors


def test_compile_concept_retries_on_invalid_json() -> None:
    llm = FakeLLM(responses=["not json", _valid_json()])
    result = compile_concept("X", SAMPLE_EVIDENCE, llm)
    assert result.status == "ok"
    assert len(llm.calls) == 2


def test_compile_concept_parse_error_after_retry() -> None:
    llm = FakeLLM(responses=["not json", "still not json"])
    result = compile_concept("X", SAMPLE_EVIDENCE, llm)
    assert result.status == "parse_error"
    assert "[PARSE_ERROR]" in result.markdown
    assert result.raw_output is not None
    # Key Papers still rendered (auto-derived from evidence, not from LLM)
    assert "[[bert]]" in result.markdown


def test_compile_concept_missing_key_treated_as_parse_error() -> None:
    incomplete = json.dumps({"definition": "x", "origin": "y", "variants": "z"})
    llm = FakeLLM(responses=[incomplete, incomplete])
    result = compile_concept("X", SAMPLE_EVIDENCE, llm)
    assert result.status == "parse_error"


# ----------------------------------------------------------------- helpers


def test_parse_json_strict_accepts_valid() -> None:
    obj = _parse_json_strict(_valid_json())
    assert obj is not None
    assert set(REQUIRED_KEYS).issubset(obj.keys())


def test_parse_json_strict_rejects_list() -> None:
    assert _parse_json_strict("[]") is None


def test_parse_json_strict_rejects_missing_key() -> None:
    raw = json.dumps({"definition": "x"})
    assert _parse_json_strict(raw) is None


def test_format_evidence_block_caps_count() -> None:
    many = [ConceptEvidence(arxiv_id=f"id{i}", snippet=f"snip{i}") for i in range(20)]
    block = _format_evidence_block(many)
    # MAX_EVIDENCE_PER_CONCEPT = 8
    assert block.count("\n") == 7  # 8 lines, 7 newlines


def test_format_evidence_block_truncates_long_snippets() -> None:
    long = ConceptEvidence(arxiv_id="x", snippet="a" * 1000)
    block = _format_evidence_block([long])
    assert "..." in block
    assert len(block) < 1000


# --------------------------------------------------------- heavy E2E (gated)


@pytest.mark.skipif(
    os.environ.get("SCIRAG_RUN_HEAVY") != "1",
    reason="Set SCIRAG_RUN_HEAVY=1 to run the real-Ollama E2E.",
)
def test_compile_concept_e2e_against_real_ollama() -> None:
    from src.llm.client import get_client

    client = get_client(None)
    result = compile_concept("Masked Language Model", SAMPLE_EVIDENCE, client)
    assert result.status in ("ok", "parse_error")
    if result.status == "ok":
        # No fabricated arxiv_ids
        assert "[[bert]]" in result.markdown
        assert "[[scibert]]" in result.markdown
