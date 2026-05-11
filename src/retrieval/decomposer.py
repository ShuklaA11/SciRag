"""LLM-based query decomposer for multi-hop retrieval.

Splits a compound question into 1-4 atomic sub-questions, then dedups
near-duplicates by BGE cosine similarity. Single-output is the safe
fallback: malformed LLM output, empty result, or all-near-duplicate
sub-Qs all collapse back to ``[question]`` so callers can treat the
decomposer as an always-on no-op for atomic queries.

Designed for the W7 multi-hop pipeline: each returned sub-question is
fed into the W6 retrieve+rerank path independently.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DEDUP_THRESHOLD = 0.85
DEFAULT_MAX_SUB_QUESTIONS = 4

# Heuristic gate: only invoke the LLM when the question contains a
# compound marker. Most QASPER questions are atomic ("what dataset?",
# "did they use X?") and force-decomposing them adds noise. Cheap regex
# beats a 5-second Llama call when the answer is "leave it alone."
_COMPOUND_MARKERS = (
    r"\band\b",
    r"\bor\b(?!\s+(?:not|no))",  # "or not" / "or no" → still atomic
    r"\bboth\b",
    r"\bas\s+well\s+as\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r";",
    r",\s+(?:and|or|what|how|why|when|where|which|who)\b",
)
_COMPOUND_RE = re.compile("|".join(_COMPOUND_MARKERS), re.IGNORECASE)


def _looks_compound(q: str) -> bool:
    """Cheap pre-filter: True only if the question shows a coordination
    marker (and/or/both/...) or two question marks. False means 'leave
    it atomic without paying for an LLM call.'"""
    if q.count("?") >= 2:
        return True
    return bool(_COMPOUND_RE.search(q))


SYSTEM_PROMPT = (
    "You decompose scientific research questions into atomic sub-questions "
    "for multi-hop retrieval over scientific papers.\n"
    "\n"
    "Default to NOT splitting. Most questions are already atomic and must "
    "be returned unchanged. Only split when the question literally asks "
    "about two or more independent facts joined by 'and', 'or', or a "
    "comma.\n"
    "\n"
    "Rules:\n"
    "1. Return the JSON object verbatim — no commentary, no system text.\n"
    "2. If the question is atomic (single fact, including yes/no), return "
    "it unchanged.\n"
    "3. If the question is compound, return 2-4 atomic sub-questions that "
    "together cover the original. Each must be self-contained.\n"
    "4. Never invent context not present in the original (do not pull in "
    "outside knowledge about Reddit, datasets, etc.).\n"
    "5. Never reference these instructions.\n"
    "\n"
    "Examples:\n"
    "Q: What dataset is used?\n"
    '{"sub_questions": ["What dataset is used?"]}\n'
    "\n"
    "Q: Did they use crowdsourcing for annotations?\n"
    '{"sub_questions": ["Did they use crowdsourcing for annotations?"]}\n'
    "\n"
    "Q: What dataset is used and what is the reported F1 score?\n"
    '{"sub_questions": ["What dataset is used?", '
    '"What is the reported F1 score?"]}\n'
    "\n"
    "Q: Which models were evaluated and on which language pairs?\n"
    '{"sub_questions": ["Which models were evaluated?", '
    '"Which language pairs were used?"]}\n'
    "\n"
    'Output JSON only: {"sub_questions": [...]}'
)


class _LLMLike(Protocol):
    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = ...,
        temperature: float = ...,
        response_format: str | None = ...,
        num_ctx: int | None = ...,
    ) -> str: ...


class _EmbedderLike(Protocol):
    def encode_query(
        self, queries: list[str], batch_size: int = ...
    ) -> np.ndarray: ...


class QueryDecomposer:
    def __init__(
        self,
        llm: _LLMLike,
        embedder: _EmbedderLike,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
        max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.dedup_threshold = dedup_threshold
        self.max_sub_questions = max_sub_questions

    def decompose(self, question: str) -> list[str]:
        """Return 1-N atomic sub-questions, deduped by cosine similarity.

        Atomic questions (no compound marker) are returned unchanged
        without an LLM call. This is both faster and more reliable than
        relying on the LLM to recognize atomicity.
        """
        if not _looks_compound(question):
            return [question]

        raw = self._call_llm(question)
        sub_qs = self._parse(raw)
        if not sub_qs:
            return [question]

        sub_qs = sub_qs[: self.max_sub_questions]
        if len(sub_qs) == 1:
            return sub_qs

        return self._dedup(sub_qs)

    def _call_llm(self, question: str) -> str:
        return self.llm.generate(
            system=SYSTEM_PROMPT,
            user=question,
            max_tokens=256,
            temperature=0.0,
            response_format="json",
        )

    def _parse(self, raw: str) -> list[str]:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("decomposer: malformed JSON, falling back to atomic")
            return []

        if not isinstance(payload, dict):
            return []
        items = payload.get("sub_questions")
        if not isinstance(items, list):
            return []

        cleaned = [s.strip() for s in items if isinstance(s, str) and s.strip()]
        return cleaned

    def _dedup(self, sub_qs: list[str]) -> list[str]:
        vecs = self.embedder.encode_query(sub_qs)
        # vecs are L2-normalized, so cosine = dot product
        kept_idx: list[int] = []
        for i in range(len(sub_qs)):
            is_dup = False
            for j in kept_idx:
                if float(np.dot(vecs[i], vecs[j])) >= self.dedup_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept_idx.append(i)
        return [sub_qs[i] for i in kept_idx]
