"""The LangGraph supervisor crew (SB-X2).

A supervisor node routes the query to one specialist at a time; each specialist
runs a grounded CrewTools capability and appends its ToolResult to shared state;
control returns to the supervisor until it emits FINISH (or the iteration cap
trips), then a synthesis node writes the final answer.

Everything is dependency-injected: ``build_crew_graph(llm, tools)`` closes over
an LLM (supervisor routing + synthesis) and a CrewTools-like object (the
grounded work), so tests drive the whole graph with a scripted fake LLM and
fake tools — deterministic multi-agent traversal, no models in the loop.

Robustness for weak local routers: routing output is parsed defensively
(malformed / unknown route -> FINISH) and an iteration cap force-finishes the
loop regardless of what the LLM says.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph

from src.crew.state import AGENTS, FINISH, CrewState

DEFAULT_MAX_ITERS = 6

SUPERVISOR_SYSTEM = (
    "You are the supervisor of a scientific-research crew. Given the user's "
    "query and the findings gathered so far, choose the single most useful "
    "next specialist, or FINISH when the findings are sufficient.\n\n"
    "Specialists:\n"
    "- search: retrieve grounding passages from the paper corpus.\n"
    "- verify: check the query's claims against the literature "
    "(entailed / contradicted / novel).\n"
    "- novelty: discover novel research directions / gaps from the query.\n\n"
    "Do not repeat a specialist whose findings already answer the need. "
    'Respond with JSON only: {"next": "search|verify|novelty|FINISH", '
    '"reason": "<one short clause>"}'
)

SYNTHESIS_SYSTEM = (
    "You are the lead of a scientific-research crew. Write a concise, grounded "
    "answer to the user's query using ONLY the crew's findings below. Cite "
    "retrieved papers by their arxiv id where relevant, state verification "
    "verdicts plainly, and surface novel directions if any were found. If the "
    "findings are thin, say so rather than inventing."
)


class _LLMLike(Protocol):
    def generate(self, system: str, user: str, **kwargs: Any) -> str: ...


class _ToolsLike(Protocol):
    def search_corpus(self, query: str, k: int = ...) -> Any: ...
    def verify_idea(self, idea: str) -> Any: ...
    def find_gaps(self, seed: str, seed_directions: list[str] | None = ...) -> Any: ...


def parse_route(raw: str) -> str:
    """Defensively extract the next-agent decision; anything odd -> FINISH."""
    try:
        payload = json.loads(raw)
        nxt = payload.get("next", FINISH)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return FINISH
    return nxt if (nxt in AGENTS or nxt == FINISH) else FINISH


def _digest(findings: list) -> str:
    return "\n\n".join(f"[{f.tool}] {f.summary}" for f in findings) or "(none yet)"


def build_crew_graph(
    llm: _LLMLike, tools: _ToolsLike, max_iters: int = DEFAULT_MAX_ITERS
):
    """Compile the supervisor crew graph over an injected LLM + tools."""

    def supervisor(state: CrewState) -> dict:
        iters = state.get("iterations", 0) + 1
        if iters > max_iters:
            return {"next_agent": FINISH, "iterations": iters}
        user = (
            f"Query: {state['query']}\n\n"
            f"Findings so far:\n{_digest(state.get('findings', []))}\n\n"
            "Which specialist next?"
        )
        raw = llm.generate(
            system=SUPERVISOR_SYSTEM, user=user,
            response_format="json", temperature=0.0,
        )
        return {"next_agent": parse_route(raw), "iterations": iters}

    def search(state: CrewState) -> dict:
        return {"findings": [tools.search_corpus(state["query"])]}

    def verify(state: CrewState) -> dict:
        return {"findings": [tools.verify_idea(state["query"])]}

    def novelty(state: CrewState) -> dict:
        return {"findings": [tools.find_gaps(state["query"])]}

    def synthesize(state: CrewState) -> dict:
        user = (
            f"Query: {state['query']}\n\n"
            f"Crew findings:\n{_digest(state.get('findings', []))}\n\n"
            "Write the final grounded answer."
        )
        answer = llm.generate(system=SYNTHESIS_SYSTEM, user=user, temperature=0.2)
        return {"answer": answer}

    def route(state: CrewState) -> str:
        return state.get("next_agent", FINISH)

    g = StateGraph(CrewState)
    g.add_node("supervisor", supervisor)
    g.add_node("search", search)
    g.add_node("verify", verify)
    g.add_node("novelty", novelty)
    g.add_node("synthesize", synthesize)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor", route,
        {"search": "search", "verify": "verify", "novelty": "novelty",
         FINISH: "synthesize"},
    )
    for agent in AGENTS:
        g.add_edge(agent, "supervisor")
    g.add_edge("synthesize", END)

    return g.compile()
