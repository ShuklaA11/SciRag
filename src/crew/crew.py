"""ResearchCrew — the public entry to the supervisor crew (SB-X3).

Thin wrapper over the compiled LangGraph: hand it an injected LLM and tools,
call ``run(query)``, get back a :class:`CrewResult` (the synthesized answer plus
the auditable per-specialist findings). Kept dependency-injected so the app edge
wires real models and tests wire fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.crew.graph import DEFAULT_MAX_ITERS, build_crew_graph
from src.crew.tools import ToolResult


@dataclass(frozen=True)
class CrewResult:
    query: str
    answer: str
    findings: tuple[ToolResult, ...]


class ResearchCrew:
    def __init__(self, llm: Any, tools: Any, *, max_iters: int = DEFAULT_MAX_ITERS) -> None:
        self._graph = build_crew_graph(llm, tools, max_iters=max_iters)

    def run(self, query: str) -> CrewResult:
        final = self._graph.invoke({"query": query})
        return CrewResult(
            query=query,
            answer=final.get("answer", ""),
            findings=tuple(final.get("findings", [])),
        )
