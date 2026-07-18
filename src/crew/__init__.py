"""Multi-agent research crew (Phase F) — LangGraph supervisor over the
benchmarked SciRAG components.

Each specialist agent wraps one measured subsystem (retrieval, idea
verification, novelty/gap discovery) as a grounded capability; a supervisor
LLM routes between them. Logic lives here (pure, DI'd, fake-tested); the
Streamlit "Workshop" page is a thin view and the real-model composition root
lives at the app edge.
"""

from src.crew.tools import CrewTools, ToolResult

__all__ = ["CrewTools", "ToolResult"]
