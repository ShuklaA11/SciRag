"""SB-X4: research-workshop view smoke test.

st.navigation non-default pages aren't AppTest-navigable, so the workshop page
is verified two ways: (1) its collaborator seam — build_crew() under
SCIRAG_FAKE_ENGINE=1 returns a runnable crew whose result the view renders; and
(2) the app still boots with the page registered (no import/registration break).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("app").resolve()))  # bare-name app-edge imports


def test_workshop_seam_returns_renderable_result(monkeypatch):
    monkeypatch.setenv("SCIRAG_FAKE_ENGINE", "1")
    from crew import build_crew

    result = build_crew().run("does sparse attention help long documents?")

    assert result.query == "does sparse attention help long documents?"
    assert result.answer  # something to render under "Answer"
    assert [f.tool for f in result.findings] == ["search_corpus", "verify_idea"]
    # every finding has a summary the view renders in its expander
    assert all(f.summary for f in result.findings)


def test_app_boots_with_workshop_page(monkeypatch, tmp_path):
    monkeypatch.setenv("SCIRAG_HUB_DB", str(tmp_path / "hub.db"))
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

    at = AppTest.from_file("app/streamlit_app.py").run()

    assert not at.exception
