"""Unit tests for CrewTools — fake components, deterministic ToolResults.

No LLM and no real models: fakes stand in for FlatIndex / IdeaEvaluator /
BrainstormLoop so the grounded dispatch is asserted exactly.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.crew.tools import CrewTools, ToolResult


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    def search(self, query, k=5):
        self.calls.append((query, k))
        return self._chunks


class FakeEvaluator:
    def __init__(self, buckets):
        # buckets: list of (claim, bucket)
        self._verdicts = [
            SimpleNamespace(claim=c, bucket=b, best_support=0.9, best_contradict=0.1)
            for c, b in buckets
        ]

    def evaluate(self, idea):
        return SimpleNamespace(idea=idea, verdicts=tuple(self._verdicts))


class FakeBrainstorm:
    def __init__(self, directions, iterations=2):
        self._report = SimpleNamespace(
            seed="", directions=tuple(SimpleNamespace(claim=d) for d in directions),
            iterations=iterations,
        )

    def run(self, seed, seed_directions):
        self._report.seed = seed
        return self._report


def _tools(chunks=None, buckets=None, directions=None):
    return CrewTools(
        FakeRetriever(chunks or []),
        FakeEvaluator(buckets or []),
        FakeBrainstorm(directions or []),
    )


def test_search_corpus_returns_chunks_and_summary():
    chunks = [
        {"arxiv_id": "1901.001", "section_type": "method", "text": "We use BGE embeddings."},
        {"arxiv_id": "1902.002", "section_type": "results", "text": "Recall improves."},
    ]
    tools = _tools(chunks=chunks)

    res = tools.search_corpus("embedding retrieval", k=2)

    assert isinstance(res, ToolResult)
    assert res.tool == "search_corpus"
    assert res.data["chunks"] == chunks
    assert "Retrieved 2 chunk(s)" in res.summary
    assert "1901.001/method" in res.summary
    assert tools.retriever.calls == [("embedding retrieval", 2)]


def test_search_corpus_empty_is_handled():
    res = _tools(chunks=[]).search_corpus("nothing matches")
    assert res.data["chunks"] == []
    assert "No corpus matches" in res.summary


def test_verify_idea_buckets_and_counts():
    tools = _tools(buckets=[
        ("Sparse attention cuts memory.", "ENTAILED"),
        ("It beats dense accuracy.", "NOVEL"),
        ("It is O(n).", "CONTRADICTED"),
    ])

    res = tools.verify_idea("Sparse attention is efficient.")

    assert res.tool == "verify_idea"
    assert res.data["bucket_counts"] == {"ENTAILED": 1, "NOVEL": 1, "CONTRADICTED": 1}
    assert len(res.data["verdicts"]) == 3
    assert "Evaluated 3 claim(s)" in res.summary
    assert "[NOVEL]" in res.summary


def test_find_gaps_returns_directions():
    tools = _tools(directions=["contrastive pretraining for low-resource NER",
                               "sparse attention for long clinical notes"])

    res = tools.find_gaps("efficient transformers", seed_directions=["seed"])

    assert res.tool == "find_gaps"
    assert res.data["directions"] == [
        "contrastive pretraining for low-resource NER",
        "sparse attention for long clinical notes",
    ]
    assert "Discovered 2 novel direction(s)" in res.summary
    assert res.data["iterations"] == 2
