"""Integration test for ResearchCrew's public API — fakes, no models."""

from __future__ import annotations

import json

from src.crew.crew import CrewResult, ResearchCrew
from src.crew.tools import ToolResult


class FakeLLM:
    def __init__(self, routes, answer="SYNTHESIZED"):
        self._routes = list(routes)
        self._answer = answer

    def generate(self, system, user, *, response_format=None, temperature=0.0, **kw):
        if response_format == "json":
            nxt = self._routes.pop(0) if self._routes else "FINISH"
            return json.dumps({"next": nxt})
        return self._answer


class FakeTools:
    def search_corpus(self, query, k=5):
        return ToolResult("search_corpus", "2 chunks", {"chunks": [1, 2]})

    def verify_idea(self, idea):
        return ToolResult("verify_idea", "1 entailed", {})

    def find_gaps(self, seed, seed_directions=None):
        return ToolResult("find_gaps", "1 direction", {})


def test_research_crew_run_returns_result():
    crew = ResearchCrew(FakeLLM(routes=["search", "verify", "FINISH"]), FakeTools())

    res = crew.run("does sparse attention help long documents?")

    assert isinstance(res, CrewResult)
    assert res.query == "does sparse attention help long documents?"
    assert res.answer == "SYNTHESIZED"
    assert [f.tool for f in res.findings] == ["search_corpus", "verify_idea"]


def test_research_crew_immediate_finish_gives_empty_findings():
    crew = ResearchCrew(FakeLLM(routes=["FINISH"]), FakeTools())

    res = crew.run("trivial query")

    assert res.findings == ()
    assert res.answer == "SYNTHESIZED"
