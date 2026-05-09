"""Tests for the citation expander."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import pytest

from src.retrieval.citation_expander import CitationExpander


@pytest.fixture
def synthetic_graph(tmp_path: Path) -> Path:
    """5-node graph:
        A -> B -> C        (in_corpus: A, B, C, D)
        A -> D
        E -> A             (E external, no chunks)
    """
    g = nx.DiGraph()
    for n in ["A", "B", "C", "D"]:
        g.add_node(n, in_corpus=True)
    g.add_node("E", in_corpus=False)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("A", "D")
    g.add_edge("E", "A")
    path = tmp_path / "graph.pickle"
    with path.open("wb") as f:
        pickle.dump(g, f)
    return path


def test_loads_graph_and_in_corpus_set(synthetic_graph):
    ex = CitationExpander(synthetic_graph)
    assert ex.in_corpus == frozenset({"A", "B", "C", "D"})


def test_neighbors_both_directions_in_corpus(synthetic_graph):
    ex = CitationExpander(synthetic_graph)
    # A -> B, A -> D (out); E -> A (in, but E is not in_corpus)
    assert ex.neighbors("A", directions="both") == {"B", "D"}


def test_neighbors_out_only(synthetic_graph):
    ex = CitationExpander(synthetic_graph)
    assert ex.neighbors("A", directions="out") == {"B", "D"}
    assert ex.neighbors("B", directions="out") == {"C"}


def test_neighbors_in_only(synthetic_graph):
    ex = CitationExpander(synthetic_graph)
    # B is cited by A; A is cited by E (external)
    assert ex.neighbors("B", directions="in") == {"A"}
    assert ex.neighbors("A", directions="in") == set()  # E filtered out


def test_neighbors_external_when_in_corpus_only_false(synthetic_graph):
    ex = CitationExpander(synthetic_graph)
    assert ex.neighbors("A", directions="in", in_corpus_only=False) == {"E"}


def test_unknown_paper_returns_empty(synthetic_graph):
    ex = CitationExpander(synthetic_graph)
    assert ex.neighbors("Z") == set()


def test_expanded_paper_ids_includes_self(synthetic_graph):
    ex = CitationExpander(synthetic_graph)
    assert ex.expanded_paper_ids("A") == {"A", "B", "D"}
    assert ex.expanded_paper_ids("C") == {"B", "C"}


def test_invalid_direction_raises(synthetic_graph):
    ex = CitationExpander(synthetic_graph)
    with pytest.raises(ValueError):
        ex.neighbors("A", directions="sideways")
