"""Agentic brainstorm loop (v3 Phase D).

Explore the literature around a seed idea: retrieve → citation-expand → assess
novelty (Phase C) → surface gaps (NOVEL directions) → propose new directions →
dedup → loop (capped). A plain iterative loop, not a LangGraph state machine —
the flow is linear with a stop condition, matching the codebase precedent in
``retrieval.multihop``. The LLM narrates; retrieval + NLI decide what's a gap.
"""

from src.brainstorm.frontier import Frontier

__all__ = ["Frontier"]
