"""Tests for the LLM query decomposer.

The decomposer turns a single question into 1-4 atomic sub-questions
via an LLM call, then dedups near-duplicates by BGE cosine similarity.
The real LLM and BGE model are bypassed via fakes; an opt-in
``SCIRAG_RUN_HEAVY=1`` test exercises the real Ollama+BGE path.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.retrieval.decomposer import QueryDecomposer


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system, user, **kwargs) -> str:
        self.calls.append((system, user))
        return self.response


class _FakeEmbedder:
    """Returns deterministic unit vectors keyed by query text.

    Tests pass in an explicit ``vectors`` dict so we can pin cosine
    similarity between any two sub-questions.
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode_query(self, queries: list[str], batch_size: int = 16) -> np.ndarray:
        out = np.array([self.vectors[q] for q in queries], dtype=np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-12, None)


def _make(llm_response: str, vectors: dict[str, list[float]] | None = None,
          dedup_threshold: float = 0.85) -> QueryDecomposer:
    return QueryDecomposer(
        llm=_FakeLLM(llm_response),
        embedder=_FakeEmbedder(vectors or {}),
        dedup_threshold=dedup_threshold,
        max_sub_questions=4,
    )


def test_atomic_question_skips_llm_via_heuristic_gate():
    """No compound marker → skip LLM, return [question] unchanged."""
    q = "What is the F1 score?"
    fake_llm = _FakeLLM('{"sub_questions": ["should not be used"]}')
    d = QueryDecomposer(
        llm=fake_llm,
        embedder=_FakeEmbedder({q: [1.0, 0.0]}),
    )
    assert d.decompose(q) == [q]
    assert fake_llm.calls == []  # gate prevented LLM call


def test_yes_no_question_skips_llm():
    """Yes/no questions are atomic and must skip the LLM."""
    q = "Did they use a crowdsourcing platform for manual annotations?"
    fake_llm = _FakeLLM('{"sub_questions": ["bogus"]}')
    d = QueryDecomposer(
        llm=fake_llm, embedder=_FakeEmbedder({q: [1.0, 0.0]}),
    )
    assert d.decompose(q) == [q]
    assert fake_llm.calls == []


def test_compound_question_returns_multiple():
    q = "What dataset and what model are used?"
    sub_a = "What dataset is used?"
    sub_b = "What model is used?"
    d = _make(
        llm_response=f'{{"sub_questions": ["{sub_a}", "{sub_b}"]}}',
        vectors={sub_a: [1.0, 0.0], sub_b: [0.0, 1.0]},
    )
    assert d.decompose(q) == [sub_a, sub_b]


def test_compound_marker_triggers_llm():
    q = "What models were evaluated and on what datasets?"
    sub_a = "What models were evaluated?"
    sub_b = "What datasets were used?"
    fake_llm = _FakeLLM(f'{{"sub_questions": ["{sub_a}", "{sub_b}"]}}')
    d = QueryDecomposer(
        llm=fake_llm,
        embedder=_FakeEmbedder({sub_a: [1.0, 0.0], sub_b: [0.0, 1.0]}),
    )
    out = d.decompose(q)
    assert out == [sub_a, sub_b]
    assert len(fake_llm.calls) == 1


def test_double_question_mark_triggers_llm():
    q = "What dataset? What model?"
    sub_a = "What dataset?"
    sub_b = "What model?"
    fake_llm = _FakeLLM(f'{{"sub_questions": ["{sub_a}", "{sub_b}"]}}')
    d = QueryDecomposer(
        llm=fake_llm,
        embedder=_FakeEmbedder({sub_a: [1.0, 0.0], sub_b: [0.0, 1.0]}),
    )
    out = d.decompose(q)
    assert out == [sub_a, sub_b]
    assert len(fake_llm.calls) == 1


def test_or_not_does_not_trigger_llm():
    """'or not' is part of yes/no phrasing, not a compound marker."""
    q = "Did they use crowdsourcing or not?"
    fake_llm = _FakeLLM('{"sub_questions": ["bogus"]}')
    d = QueryDecomposer(
        llm=fake_llm, embedder=_FakeEmbedder({q: [1.0, 0.0]}),
    )
    assert d.decompose(q) == [q]
    assert fake_llm.calls == []


def test_dedup_drops_near_duplicate_subqs():
    """Two sub-Qs with cosine > threshold collapse to one."""
    q = "What dataset and what model?"  # compound marker triggers LLM
    sub_a = "What dataset?"
    sub_b = "Which dataset?"  # near-duplicate
    sub_c = "What model?"
    d = _make(
        llm_response=(
            f'{{"sub_questions": ["{sub_a}", "{sub_b}", "{sub_c}"]}}'
        ),
        vectors={
            sub_a: [1.0, 0.0],
            sub_b: [0.99, 0.01],  # cos ≈ 0.9999 with sub_a
            sub_c: [0.0, 1.0],
        },
        dedup_threshold=0.85,
    )
    out = d.decompose(q)
    assert out == [sub_a, sub_c]


def test_caps_at_max_sub_questions():
    q = "What a, b, c, d, e, and f?"  # compound marker triggers LLM
    subs = [f"sub-{i}" for i in range(6)]
    payload = ", ".join(f'"{s}"' for s in subs)
    # orthogonal vectors so dedup doesn't drop any
    vectors = {s: [1.0 if i == j else 0.0 for j in range(6)]
               for i, s in enumerate(subs)}
    d = _make(
        llm_response=f'{{"sub_questions": [{payload}]}}',
        vectors=vectors,
    )
    out = d.decompose(q)
    assert len(out) == 4
    assert out == subs[:4]


# Compound-marker questions used so the heuristic gate routes to the LLM.
_COMPOUND_Q = "What is the F1 score and the dataset?"


def test_malformed_json_falls_back_to_original():
    d = _make(
        llm_response="not valid json at all",
        vectors={_COMPOUND_Q: [1.0, 0.0]},
    )
    assert d.decompose(_COMPOUND_Q) == [_COMPOUND_Q]


def test_empty_sub_questions_falls_back_to_original():
    d = _make(
        llm_response='{"sub_questions": []}',
        vectors={_COMPOUND_Q: [1.0, 0.0]},
    )
    assert d.decompose(_COMPOUND_Q) == [_COMPOUND_Q]


def test_missing_key_falls_back_to_original():
    d = _make(
        llm_response='{"other_key": ["foo"]}',
        vectors={_COMPOUND_Q: [1.0, 0.0]},
    )
    assert d.decompose(_COMPOUND_Q) == [_COMPOUND_Q]


def test_non_string_subqs_filtered_out():
    sub_a = "What dataset?"
    d = _make(
        llm_response=f'{{"sub_questions": ["{sub_a}", 42, null]}}',
        vectors={sub_a: [1.0, 0.0]},
    )
    assert d.decompose(_COMPOUND_Q) == [sub_a]


def test_all_invalid_subqs_falls_back_to_original():
    d = _make(
        llm_response='{"sub_questions": [42, null, ""]}',
        vectors={_COMPOUND_Q: [1.0, 0.0]},
    )
    assert d.decompose(_COMPOUND_Q) == [_COMPOUND_Q]


@pytest.mark.skipif(
    os.environ.get("SCIRAG_RUN_HEAVY") != "1",
    reason="hits real Ollama + loads BGE; set SCIRAG_RUN_HEAVY=1 to run",
)
def test_real_decomposer_handles_compound_question():
    from src.llm.client import get_client
    from src.pipeline.bge_embedder import BGEEmbedder

    d = QueryDecomposer(llm=get_client(), embedder=BGEEmbedder())
    out = d.decompose(
        "What dataset is used and what is the reported F1 score?"
    )
    assert 1 <= len(out) <= 4
    assert all(isinstance(s, str) and s.strip() for s in out)
