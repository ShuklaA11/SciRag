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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make sibling `engine` importable

import streamlit as st
from engine import build_evaluator

from src.hub import HubStore, domain_options
from src.ideas import CONTRADICTED, ENTAILED, NOVEL
from src.wiki.search import WikiSearchIndex


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


_BUCKET_STYLE = {
    ENTAILED: ("🟢", "already in the literature"),
    CONTRADICTED: ("🔴", "the corpus contradicts this"),
    NOVEL: ("🟡", "untested — a novelty gap"),
}


def _render_verdicts(report) -> None:
    if not report.verdicts:
        st.info("No claims were extracted from that idea.")
        return
    renderers = {ENTAILED: st.success, CONTRADICTED: st.error, NOVEL: st.warning}
    for v in report.verdicts:
        icon, gloss = _BUCKET_STYLE.get(v.bucket, ("", ""))
        renderers.get(v.bucket, st.write)(f"{icon} **{v.bucket}** — {v.claim}  \n_{gloss}_")


def _evaluate_page() -> None:
    st.header("Evaluate idea")
    projects = _store().list_projects()
    if not projects:
        st.info("Create a project first from **New project**.")
        return

    proj_by_label = {f"{p.name} ({p.domain})": p for p in projects}
    label = st.selectbox("Project", list(proj_by_label))
    idea = st.text_area("Research idea")
    if st.button("Evaluate", type="primary"):
        if not idea.strip():
            st.error("Enter an idea to evaluate.")
            return
        report = build_evaluator().evaluate(idea)
        _render_verdicts(report)
        project = proj_by_label[label]
        _store().save_evaluation(project.id, report)
        st.caption(f"Saved evaluation to “{project.name}”.")


@st.cache_resource(show_spinner="Indexing the compiled wiki …")
def _wiki_index() -> WikiSearchIndex:
    return WikiSearchIndex.from_wiki()


def _wiki_search_page() -> None:
    st.header("Wiki search")
    index = _wiki_index()
    st.caption(f"{len(index)} wiki entries (paper summaries + concept articles)")
    query = st.text_input("Search the compiled wiki")
    if not query.strip():
        return
    hits = index.search(query, k=10)
    if not hits:
        st.info("No matches.")
        return
    for h in hits:
        st.write(f"**{h.title}** &nbsp; `{h.kind}` · `{h.ident}` · score {h.score:.1f}")


def main() -> None:
    st.set_page_config(page_title="SciRAG hub", page_icon="🔬")
    page = st.navigation(
        [
            st.Page(_new_project_page, title="New project", default=True),
            st.Page(_evaluate_page, title="Evaluate idea"),
            st.Page(_wiki_search_page, title="Wiki search"),
            st.Page(_projects_page, title="Projects"),
        ]
    )
    page.run()


main()
