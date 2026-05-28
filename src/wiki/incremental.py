"""Incremental compilation helpers for the SciRAG wiki.

Tracks ``sha256(tei_xml)`` per arxiv_id in ``wiki/.state.json`` so a
re-run of ``scripts/compile_papers.py`` only summarizes papers whose
TEI actually changed since the last successful compile.

Concept articles carry their source paper ids in the
``depends_on:`` frontmatter line emitted by
``src/wiki/concept_compiler.py``. When a source paper's hash changes,
:func:`flag_stale_concepts` writes a sibling ``{slug}.stale`` marker
next to the concept article. The compile script consults these markers
on next run; the operator decides when to actually rebuild
(per decision #4: flag-only, no auto-rebuild).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

STATE_FILENAME = ".state.json"


@dataclass(frozen=True)
class IncrementalDecision:
    needs_compile: bool
    reason: str  # "new" | "changed" | "unchanged" | "force_rebuild"


def hash_tei(tei_xml: str) -> str:
    return hashlib.sha256(tei_xml.encode("utf-8")).hexdigest()


def load_state(wiki_root: Path) -> dict[str, str]:
    """Return mapping ``{arxiv_id: tei_sha256}`` or {} if no state yet."""
    p = wiki_root / STATE_FILENAME
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def save_state(wiki_root: Path, state: dict[str, str]) -> None:
    wiki_root.mkdir(parents=True, exist_ok=True)
    p = wiki_root / STATE_FILENAME
    p.write_text(json.dumps(state, indent=2, sort_keys=True))


def decide(
    arxiv_id: str,
    tei_xml: str,
    state: dict[str, str],
    *,
    md_path_exists: bool,
    force_rebuild: bool,
) -> IncrementalDecision:
    """Decide whether a paper needs (re)compilation."""
    if force_rebuild:
        return IncrementalDecision(needs_compile=True, reason="force_rebuild")
    new_hash = hash_tei(tei_xml)
    old_hash = state.get(arxiv_id)
    if not md_path_exists:
        return IncrementalDecision(needs_compile=True, reason="new")
    if old_hash is None:
        # md exists from before .state.json was introduced — adopt it.
        return IncrementalDecision(needs_compile=False, reason="bootstrap")
    if old_hash != new_hash:
        return IncrementalDecision(needs_compile=True, reason="changed")
    return IncrementalDecision(needs_compile=False, reason="unchanged")


def update_state(state: dict[str, str], arxiv_id: str, tei_xml: str) -> dict[str, str]:
    """Return a new dict with arxiv_id's hash updated. Immutable update."""
    out = dict(state)
    out[arxiv_id] = hash_tei(tei_xml)
    return out


# --------------------------------------------------------- concept staleness


def _read_depends_on(concept_md_path: Path) -> list[str]:
    text = concept_md_path.read_text()
    if not text.startswith("---\n"):
        return []
    end = text.find("\n---\n", 4)
    if end == -1:
        return []
    for line in text[4:end].splitlines():
        if line.startswith("depends_on:"):
            raw = line.split(":", 1)[1].strip().lstrip("[").rstrip("]")
            if not raw:
                return []
            return [p.strip() for p in raw.split(",") if p.strip()]
    return []


def flag_stale_concepts(
    concepts_dir: Path,
    changed_arxiv_ids: set[str],
) -> list[Path]:
    """Write {slug}.stale marker next to each concept that depends on a
    changed paper. Returns the list of marker paths written.

    Markers contain the list of changed paper ids that triggered the
    flag, so the operator can decide whether to rebuild manually.
    """
    if not concepts_dir.exists() or not changed_arxiv_ids:
        return []
    written: list[Path] = []
    for md in sorted(concepts_dir.glob("*.md")):
        deps = _read_depends_on(md)
        triggers = sorted(set(deps) & changed_arxiv_ids)
        if not triggers:
            continue
        marker = md.with_suffix(".stale")
        marker.write_text(json.dumps({
            "concept_slug": md.stem,
            "triggered_by": triggers,
        }, indent=2))
        written.append(marker)
    return written


def clear_stale_marker(concept_md_path: Path) -> None:
    marker = concept_md_path.with_suffix(".stale")
    if marker.exists():
        marker.unlink()
