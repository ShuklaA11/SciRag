"""SB-B1: hub persistence. The key guard is `test_persists_across_reopen` —
the daily-use acceptance criterion (state survives a restart)."""

from __future__ import annotations

import pytest

from src.hub import Evaluation, HubStore, Project
from src.ideas import ENTAILED, ClaimVerdict, Evidence, IdeaReport, Provenance


def _report(idea: str = "my idea") -> IdeaReport:
    return IdeaReport(
        idea=idea,
        verdicts=(
            ClaimVerdict("c1", ENTAILED, 0.9, 0.05, 1, Evidence("d1", "evidence text", 3.0)),
        ),
        provenance=Provenance("deberta-x", 5, 0.5, 1),
    )


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


# --- evaluations (SB-C4) ----------------------------------------------------


def test_save_and_get_evaluation_roundtrips_report(store):
    pid = store.create_project("P", "nlp_ml").id
    saved = store.save_evaluation(pid, _report(), git_commit="abc1234")

    assert isinstance(saved, Evaluation)
    got = store.get_evaluation(saved.id)
    assert got == saved
    assert got.idea == "my idea"
    assert got.git_commit == "abc1234"
    assert got.report["verdicts"][0]["bucket"] == ENTAILED
    assert got.report["provenance"]["model"] == "deberta-x"


def test_save_evaluation_stamps_current_commit_by_default(store):
    pid = store.create_project("P", "nlp_ml").id
    saved = store.save_evaluation(pid, _report())
    assert saved.git_commit  # non-empty ("unknown" outside a repo, sha inside)


def test_list_evaluations_is_scoped_to_project(store):
    a = store.create_project("A", "nlp_ml").id
    b = store.create_project("B", "biomedical").id
    e1 = store.save_evaluation(a, _report("idea a1"), git_commit="c1")
    e2 = store.save_evaluation(a, _report("idea a2"), git_commit="c2")
    store.save_evaluation(b, _report("idea b1"), git_commit="c3")

    ids = [e.id for e in store.list_evaluations(a)]
    assert set(ids) == {e1.id, e2.id}  # project B's evaluation excluded


def test_save_evaluation_to_missing_project_raises(store):
    with pytest.raises(KeyError):
        store.save_evaluation(9999, _report())


def test_get_missing_evaluation_raises(store):
    with pytest.raises(KeyError):
        store.get_evaluation(9999)


def test_evaluation_persists_across_reopen(tmp_path):
    db = tmp_path / "hub.db"
    s1 = HubStore(db)
    pid = s1.create_project("P", "nlp_ml").id
    eid = s1.save_evaluation(pid, _report(), git_commit="deadbee").id
    s1.close()

    s2 = HubStore(db)
    got = s2.get_evaluation(eid)
    assert got.git_commit == "deadbee"
    assert got.report["idea"] == "my idea"
    s2.close()
