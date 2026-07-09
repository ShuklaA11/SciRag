"""SB-B1: hub persistence. The key guard is `test_persists_across_reopen` —
the daily-use acceptance criterion (state survives a restart)."""

from __future__ import annotations

import pytest

from src.hub import HubStore, Project


@pytest.fixture
def store(tmp_path):
    s = HubStore(tmp_path / "hub.db")
    yield s
    s.close()


def test_create_and_get_project(store):
    p = store.create_project("My NLP Project", "nlp_ml")
    assert isinstance(p, Project)
    assert (p.name, p.domain) == ("My NLP Project", "nlp_ml")
    assert p.id > 0
    assert store.get_project(p.id) == p


def test_list_projects(store):
    a = store.create_project("A", "nlp_ml")
    b = store.create_project("B", "biomedical")
    assert {p.id for p in store.list_projects()} == {a.id, b.id}


def test_create_rejects_unknown_domain(store):
    with pytest.raises(ValueError):
        store.create_project("bad", "astrophysics")


def test_create_rejects_empty_name(store):
    with pytest.raises(ValueError):
        store.create_project("   ", "nlp_ml")


def test_set_project_domain(store):
    p = store.create_project("switchable", "nlp_ml")
    assert store.set_project_domain(p.id, "biomedical").domain == "biomedical"
    with pytest.raises(ValueError):
        store.set_project_domain(p.id, "nope")


def test_set_domain_missing_project_raises(store):
    with pytest.raises(KeyError):
        store.set_project_domain(9999, "nlp_ml")


def test_get_missing_raises(store):
    with pytest.raises(KeyError):
        store.get_project(9999)


def test_persists_across_reopen(tmp_path):
    db = tmp_path / "hub.db"
    s1 = HubStore(db)
    pid = s1.create_project("persist me", "nlp_ml").id
    s1.close()

    s2 = HubStore(db)
    got = s2.get_project(pid)
    assert (got.name, got.domain) == ("persist me", "nlp_ml")
    s2.close()
