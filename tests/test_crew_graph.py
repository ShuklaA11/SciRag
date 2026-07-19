"""Deterministic tests for the supervisor crew graph (SB-X2).

A scripted fake LLM drives routing (JSON calls) and synthesis (plain calls);
fake tools return canned ToolResults. This exercises the full multi-agent
traversal — routing, accumulation, the iteration cap, and malformed-route
fallback — with no model in the loop.
"""

from __future__ import annotations

import json

from src.crew.graph import build_crew_graph, parse_route
from src.crew.tools import ToolResult


class FakeLLM:
    """Routing calls (response_format='json') pop from `routes`; synthesis
    calls return `answer`."""

    def __init__(self, routes, answer="FINAL ANSWER"):
        self._routes = list(routes)
        self._answer = answer
        self.route_calls = 0
        self.synth_calls = 0

    def generate(self, system, user, *, response_format=None, temperature=0.0, **kw):
        if response_format == "json":
            self.route_calls += 1
            nxt = self._routes.pop(0) if self._routes else "FINISH"
            return json.dumps({"next": nxt, "reason": "test"})
        self.synth_calls += 1
        return self._answer


class FakeTools:
    def __init__(self):
        self.calls = []

    def search_corpus(self, query, k=5):
        self.calls.append("search")
        return ToolResult("search_corpus", "found 2 chunks", {"chunks": [1, 2]})

    def verify_idea(self, idea):
        self.calls.append("verify")
        return ToolResult("verify_idea", "2 entailed", {})

    def find_gaps(self, seed, seed_directions=None):
        self.calls.append("novelty")
        return ToolResult("find_gaps", "1 novel direction", {})


def test_scripted_traversal_accumulates_and_synthesizes():
    llm = FakeLLM(routes=["search", "verify", "FINISH"])
    tools = FakeTools()
    graph = build_crew_graph(llm, tools, max_iters=6)

    final = graph.invoke({"query": "does sparse attention help?"})

    assert [f.tool for f in final["findings"]] == ["search_corpus", "verify_idea"]
    assert tools.calls == ["search", "verify"]
    assert final["answer"] == "FINAL ANSWER"
    assert llm.route_calls == 3 and llm.synth_calls == 1


def test_novelty_route_runs_gap_specialist():
    llm = FakeLLM(routes=["novelty", "FINISH"])
    tools = FakeTools()

    final = build_crew_graph(llm, tools).invoke({"query": "open problems in NER"})

    assert [f.tool for f in final["findings"]] == ["find_gaps"]


def test_iteration_cap_force_finishes():
    # Router never says FINISH; the cap must terminate and still synthesize.
    llm = FakeLLM(routes=["search"] * 50)
    tools = FakeTools()

    final = build_crew_graph(llm, tools, max_iters=3).invoke({"query": "loop?"})

    assert len(final["findings"]) == 3  # exactly max_iters specialist runs
    assert final["answer"] == "FINAL ANSWER"  # synthesized despite the cap


def test_malformed_route_falls_back_to_finish():
    class BadLLM:
        def generate(self, system, user, *, response_format=None, **kw):
            return "not json at all" if response_format == "json" else "done"

    final = build_crew_graph(BadLLM(), FakeTools()).invoke({"query": "q"})

    assert final.get("findings", []) == []  # no specialist ran
    assert final["answer"] == "done"


def test_parse_route_is_defensive():
    assert parse_route('{"next": "search"}') == "search"
    assert parse_route('{"next": "FINISH"}') == "FINISH"
    assert parse_route('{"next": "bogus"}') == "FINISH"
    assert parse_route('{"no_next": 1}') == "FINISH"
    assert parse_route("garbage") == "FINISH"
