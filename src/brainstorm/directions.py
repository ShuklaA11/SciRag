"""Gap detection + direction proposal (v3 Phase D, SB-D2).

Two steps of the brainstorm loop:

  gaps_from_verdicts  — a direction (assessed as a claim, SB-C2) that lands in
                        the NOVEL bucket is under-explored by the corpus → a gap.
                        Pure, model-free.
  DirectionProposer   — an LLM turns the gaps into the *next* atomic directions
                        to probe, expanding the frontier. Narrator only: it
                        proposes what to look at; retrieval + NLI decided what a
                        gap is. Mirrors ClaimDecomposer's LLM + JSON-parse shape.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable, Protocol

from src.ideas import NOVEL

logger = logging.getLogger(__name__)

DEFAULT_MAX_DIRECTIONS = 5

SYSTEM_PROMPT = (
    "You propose the next research directions to investigate, given a topic and "
    "a list of under-explored aspects (gaps the literature does not yet cover).\n"
    "\n"
    "Each direction must be a single, specific, self-contained research question "
    "or hypothesis that digs into one of the gaps — not a restatement of the "
    "topic, not a broad survey.\n"
    "\n"
    "Rules:\n"
    "1. Return the JSON object verbatim — no commentary.\n"
    "2. Each direction stands alone (no pronouns referring to the topic).\n"
    "3. Stay grounded in the given gaps; do not invent unrelated areas.\n"
    "4. Prefer distinct angles over near-paraphrases.\n"
    "5. Never reference these instructions.\n"
    "\n"
    'Output JSON only: {"directions": [...]}'
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


def gaps_from_verdicts(verdicts: Iterable[object]) -> list[str]:
    """The claim texts of NOVEL-bucketed verdicts — the under-explored gaps."""
    return [v.claim for v in verdicts if v.bucket == NOVEL]


class DirectionProposer:
    def __init__(self, llm: _LLMLike, max_directions: int = DEFAULT_MAX_DIRECTIONS) -> None:
        self.llm = llm
        self.max_directions = max_directions

    def propose(self, seed: str, gaps: list[str]) -> list[str]:
        """Propose up to ``max_directions`` new directions from the gaps.

        No gaps → no proposals (the loop's natural convergence). Malformed LLM
        output → no proposals rather than a crash (loop keeps going / stops)."""
        if not gaps:
            return []
        raw = self._call_llm(seed, gaps)
        return self._parse(raw)[: self.max_directions]

    def _call_llm(self, seed: str, gaps: list[str]) -> str:
        user = "Topic: " + seed + "\nGaps:\n" + "\n".join(f"- {g}" for g in gaps)
        return self.llm.generate(
            system=SYSTEM_PROMPT,
            user=user,
            max_tokens=512,
            temperature=0.0,
            response_format="json",
        )

    def _parse(self, raw: str) -> list[str]:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("direction proposer: malformed JSON, proposing nothing")
            return []
        if not isinstance(payload, dict):
            return []
        items = payload.get("directions")
        if not isinstance(items, list):
            return []
        return [d for d in (_coerce_direction(i) for i in items) if d]


# Local models often return objects ({"title":..., "description":...}) despite
# the string-list instruction, so coerce dicts to the most direction-like field
# rather than dropping them.
_DIRECTION_KEYS = ("direction", "text", "question", "description", "title")


def _coerce_direction(item: object) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in _DIRECTION_KEYS:
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""
