"""SB-B3: thin Streamlit view smoke test. Drives the New-project page through
the real widgets and asserts the project landed in SQLite — proving the seam
view → core → persistence end to end, independent of list rendering."""

from __future__ import annotations

import pytest

from src.hub import HubStore

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = "app/streamlit_app.py"


@pytest.fixture
def hub_db(tmp_path, monkeypatch):
    db = tmp_path / "hub.db"
    monkeypatch.setenv("SCIRAG_HUB_DB", str(db))
    return db


def test_app_boots(hub_db):
    at = AppTest.from_file(APP).run()
    assert not at.exception


def test_create_project_persists_via_view(hub_db):
    at = AppTest.from_file(APP).run()
    at.text_input[0].set_value("From the UI").run()
    at.selectbox[0].set_value("biomedical").run()
    at.button[0].click().run()

    assert not at.exception
    assert at.success  # view acknowledged

    persisted = HubStore(hub_db).list_projects()
    assert [(p.name, p.domain) for p in persisted] == [("From the UI", "biomedical")]


def test_empty_name_surfaces_error_not_crash(hub_db):
    at = AppTest.from_file(APP).run()
    at.button[0].click().run()  # name left blank

    assert not at.exception
    assert at.error
    assert HubStore(hub_db).list_projects() == []
