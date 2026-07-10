"""SciRAG hub — thin Streamlit view (v3 Phase B, SB-B3).

Deliberately dumb: every stateful action calls the core API (``src.hub``) and
maps the result to a widget. No business logic, no must-persist state in
``st.session_state`` — that lives in SQLite via ``HubStore``. Swapping this
view for FastAPI+React later touches zero core.

Run: ``streamlit run app/streamlit_app.py``. The store's db path comes from
``$SCIRAG_HUB_DB`` (falls back to ``HubStore``'s default), mirroring the
``$SCIRAG_DOMAIN`` env idiom and letting tests point at a temp db.

A fresh ``HubStore`` is built per rerun on purpose — not cached as a resource:
a ``sqlite3`` connection is thread-affine, and ``st.cache_resource`` shares one
object across threads, so caching it would throw under Streamlit's threading.
Re-connecting is ~1ms and single-threaded per rerun.
"""

from __future__ import annotations

import os

import streamlit as st

from src.hub import HubStore, domain_options


def _store() -> HubStore:
    db_path = os.getenv("SCIRAG_HUB_DB")
    return HubStore(db_path) if db_path else HubStore()


def _new_project_page() -> None:
    st.header("New project")
    labels = {
        o.name: f"{o.name} · {o.eval_benchmark}" if o.eval_benchmark else o.name
        for o in domain_options()
    }

    name = st.text_input("Project name")
    domain = st.selectbox("Domain", list(labels), format_func=lambda n: labels[n])
    if st.button("Create", type="primary"):
        try:
            project = _store().create_project(name, domain)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success(f"Created “{project.name}” ({project.domain})")


def _projects_page() -> None:
    st.header("Projects")
    projects = _store().list_projects()
    if not projects:
        st.info("No projects yet — create one from **New project**.")
        return
    for p in projects:
        st.write(f"**{p.name}** — `{p.domain}` · {p.created_at}")


def main() -> None:
    st.set_page_config(page_title="SciRAG hub", page_icon="🔬")
    page = st.navigation(
        [
            st.Page(_new_project_page, title="New project", default=True),
            st.Page(_projects_page, title="Projects"),
        ]
    )
    page.run()


main()
