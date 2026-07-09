"""Domain-picker presenter (v3 Phase B, SB-B2).

The seam between the profile registry (``src.domain``) and the picker UI
(SB-B3). Core returns view-ready option snapshots so the Streamlit view is a
dumb renderer that never reaches into ``DomainProfile`` internals — the split
that lets the view be swapped (Streamlit → FastAPI+React) touching zero core.

Read-only over the registry. Persisting a chosen domain is ``HubStore``'s job
(``set_project_domain``); this module only presents the choices.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain import available, get_profile


@dataclass(frozen=True)
class DomainOption:
    """View-ready snapshot of one selectable domain.

    Carries only fields a picker legitimately displays, pulled straight from
    the profile — no invented copy. ``name`` is the value persisted via
    ``HubStore.set_project_domain``.
    """

    name: str
    embedder: str
    verification_strategy: str
    eval_benchmark: str
    data_sources: tuple[str, ...]


def domain_options() -> list[DomainOption]:
    """Selectable domains, sorted by name — the picker's full option list."""
    return [_to_option(name) for name in available()]


def _to_option(name: str) -> DomainOption:
    profile = get_profile(name)
    return DomainOption(
        name=profile.name,
        embedder=profile.embedder_name,
        verification_strategy=profile.verification_strategy,
        eval_benchmark=profile.eval_benchmark,
        data_sources=profile.data_sources,
    )
