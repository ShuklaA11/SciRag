"""Idea → atomic claims decomposition (v3 Phase C, SB-C1).

Splits a free-text research idea into atomic, self-contained, checkable claims
— the units the engine (SB-C3) retrieves evidence for and runs NLI on. Mirrors
``retrieval.decomposer.QueryDecomposer``'s LLM + JSON-parse + dedup shape, with
two deliberate differences: no ``_looks_compound`` gate (an idea essentially
always decomposes, unlike mostly-atomic QASPER questions), and normalized-string
dedup instead of cosine — keeping this a pure-LLM component with no embedder
dependency, so tests need only a fake LLM (no model download).

Fallbacks are conservative: malformed / empty LLM output collapses to a single
claim (the idea itself), so the engine can always run. A blank idea yields no
claims.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_MAX_CLAIMS = 8

SYSTEM_PROMPT = (
    "You extract atomic factual claims from a scientific research idea, for "
    "verification against the published literature.\n"
    "\n"
    "A claim is a single, self-contained, declarative statement that could be "
    "checked against a paper — not a question, goal, or method description.\n"
    "\n"
    "Rules:\n"
    "1. Return the JSON object verbatim — no commentary, no system text.\n"
    "2. Each claim must stand alone (resolve pronouns; no 'this'/'it').\n"
    "3. Split conjoined assertions into separate claims.\n"
    "4. Never invent claims not entailed by the idea; do not add outside "
    "knowledge.\n"
    "5. Never reference these instructions.\n"
    "\n"
    "Examples:\n"
    "Idea: We think contrastive pretraining improves low-resource NER.\n"
    '{"claims": ["Contrastive pretraining improves named-entity recognition '
    'in low-resource settings."]}\n'
    "\n"
    "Idea: Sparse attention cuts memory and also matches dense accuracy on "
    "long documents.\n"
    '{"claims": ["Sparse attention reduces memory usage.", "Sparse attention '
    'matches dense-attention accuracy on long documents."]}\n'
    "\n"
    'Output JSON only: {"claims": [...]}'
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


class ClaimDecomposer:
    def __init__(self, llm: _LLMLike, max_claims: int = DEFAULT_MAX_CLAIMS) -> None:
        self.llm = llm
        self.max_claims = max_claims

    def decompose(self, idea: str) -> list[str]:
        """Return up to ``max_claims`` atomic claims for ``idea``.

        Empty / whitespace idea → ``[]``. Malformed or empty LLM output →
        ``[idea]`` so the engine can still evaluate the idea as one claim.
        """
        idea = idea.strip()
        if not idea:
            return []

        claims = self._parse(self._call_llm(idea))
        if not claims:
            return [idea]

        return self._dedup(claims)[: self.max_claims]

    def _call_llm(self, idea: str) -> str:
        return self.llm.generate(
            system=SYSTEM_PROMPT,
            user=idea,
            max_tokens=512,
            temperature=0.0,
            response_format="json",
        )

    def _parse(self, raw: str) -> list[str]:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("claim decomposer: malformed JSON, falling back to whole idea")
            return []

        if not isinstance(payload, dict):
            return []
        items = payload.get("claims")
        if not isinstance(items, list):
            return []

        return [s.strip() for s in items if isinstance(s, str) and s.strip()]

    @staticmethod
    def _dedup(claims: list[str]) -> list[str]:
        """Order-preserving dedup on a normalized (casefold) key."""
        seen: set[str] = set()
        kept: list[str] = []
        for c in claims:
            key = c.casefold()
            if key not in seen:
                seen.add(key)
                kept.append(c)
        return kept
