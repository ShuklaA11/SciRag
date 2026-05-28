"""Auto-generated indices for the SciRAG wiki.

Pure functions that walk ``wiki/papers/*.md`` + ``wiki/concepts/*.md``
and emit five index files into ``wiki/indices/``:

    INDEX.md      - alphabetical by paper title + concept name
    GLOSSARY.md   - concept name -> one-line definition
    TIMELINE.md   - papers sorted by year extracted from arxiv_id
    QUESTIONS.md  - bullet list pulled from each concept's
                    "Open Questions" section
    GRAPH.json    - paper <-> concept edges (Obsidian graph view)

No LLM, no model loads. Walks the markdown rendered by the summarizer
+ concept compiler, so this module is the contract enforcement layer:
if frontmatter or section heads drift, the indices stop being
correct and tests fail loudly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.wiki.concept_extractor import (
    Summary,
    _parse_sections,
    load_summary,
    load_summaries_dir,
)

ARXIV_YEAR_PATTERN = re.compile(r"^(\d{2})(\d{2})\.\d+")  # YYMM.NNNNN
WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")
REVIEW_FILES = frozenset({"REVIEW_NOTES.md"})


@dataclass(frozen=True)
class Concept:
    slug: str
    name: str
    definition: str
    depends_on: list[str]
    open_questions: str


def _extract_year(arxiv_id: str) -> int | None:
    """Return 4-digit year from a YYMM.NNNNN arxiv id, else None."""
    m = ARXIV_YEAR_PATTERN.match(arxiv_id)
    if not m:
        return None
    yy = int(m.group(1))
    # arxiv id namespace started 2007 (07XX.NNNN); anything 07-99 is 19YY
    # is wrong — actually all numeric-prefix ids are 2007+, so 07-99 => 20YY.
    return 2000 + yy if yy <= 99 else None


def _parse_depends_on(line_value: str) -> list[str]:
    inner = line_value.strip().lstrip("[").rstrip("]")
    if not inner:
        return []
    return [p.strip() for p in inner.split(",") if p.strip()]


def load_concept(path: Path) -> Concept | None:
    """Parse a wiki/concepts/*.md emitted by concept_compiler."""
    text = path.read_text()
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    frontmatter = text[4:end]
    body = text[end + 5 :]

    name = ""
    depends_on: list[str] = []
    for line in frontmatter.splitlines():
        if line.startswith("concept:"):
            raw = line.split(":", 1)[1].strip()
            name = raw.strip('"')
        elif line.startswith("depends_on:"):
            depends_on = _parse_depends_on(line.split(":", 1)[1])

    sections = _parse_concept_sections(body)
    return Concept(
        slug=path.stem,
        name=name or path.stem,
        definition=sections.get("Definition", "").strip(),
        depends_on=depends_on,
        open_questions=sections.get("Open Questions", "").strip(),
    )


def _parse_concept_sections(body: str) -> dict[str, str]:
    """Concept articles have different section heads than papers."""
    out: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in body.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if current is not None:
                out[current] = "\n".join(buf).strip()
            current = m.group(1)
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf).strip()
    return out


def load_concepts_dir(concepts_dir: Path) -> list[Concept]:
    if not concepts_dir.exists():
        return []
    out: list[Concept] = []
    for p in sorted(concepts_dir.glob("*.md")):
        if p.name in REVIEW_FILES:
            continue
        c = load_concept(p)
        if c is not None:
            out.append(c)
    return out


# ---------------------------------------------------------------- emitters


def render_index(summaries: list[Summary], concepts: list[Concept]) -> str:
    lines = ["# Index", "", "Alphabetical index of all wiki entries.", "", "## Papers", ""]
    for s in sorted(summaries, key=lambda x: (x.title.lower(), x.arxiv_id)):
        lines.append(f"- [[{s.arxiv_id}]] — {s.title}")
    lines += ["", "## Concepts", ""]
    for c in sorted(concepts, key=lambda x: x.name.lower()):
        lines.append(f"- [[{c.slug}]] — {c.name}")
    return "\n".join(lines) + "\n"


def render_glossary(concepts: list[Concept]) -> str:
    lines = ["# Glossary", "", "Concept -> one-line definition.", ""]
    for c in sorted(concepts, key=lambda x: x.name.lower()):
        definition = c.definition.replace("\n", " ").strip() or "_(no definition)_"
        lines.append(f"- **[[{c.slug}|{c.name}]]** — {definition}")
    return "\n".join(lines) + "\n"


def render_timeline(summaries: list[Summary]) -> str:
    by_year: dict[int | None, list[Summary]] = {}
    for s in summaries:
        y = _extract_year(s.arxiv_id)
        by_year.setdefault(y, []).append(s)

    lines = ["# Timeline", "", "Papers grouped by arxiv year. "
             "Canonical entries without a YYMM id are grouped under \"Undated\".", ""]
    years = sorted([y for y in by_year if y is not None])
    for y in years:
        lines.append(f"## {y}")
        lines.append("")
        for s in sorted(by_year[y], key=lambda x: x.arxiv_id):
            lines.append(f"- [[{s.arxiv_id}]] — {s.title}")
        lines.append("")
    if None in by_year:
        lines.append("## Undated")
        lines.append("")
        for s in sorted(by_year[None], key=lambda x: x.arxiv_id):
            lines.append(f"- [[{s.arxiv_id}]] — {s.title}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_questions(concepts: list[Concept]) -> str:
    lines = ["# Open Questions", "",
             "Aggregated from the \"Open Questions\" section of each concept article.", ""]
    any_questions = False
    for c in sorted(concepts, key=lambda x: x.name.lower()):
        q = c.open_questions.strip()
        if not q or q.lower() == "unknown":
            continue
        any_questions = True
        lines.append(f"## [[{c.slug}|{c.name}]]")
        lines.append("")
        lines.append(q)
        lines.append("")
    if not any_questions:
        lines.append("_(no open questions recorded yet)_")
    return "\n".join(lines).rstrip() + "\n"


def render_graph(summaries: list[Summary], concepts: list[Concept]) -> str:
    """JSON node/edge listing for Obsidian-style graph rendering."""
    nodes: list[dict] = []
    for s in summaries:
        nodes.append({"id": s.arxiv_id, "type": "paper", "label": s.title})
    for c in concepts:
        nodes.append({"id": c.slug, "type": "concept", "label": c.name})

    edges: list[dict] = []
    paper_ids = {s.arxiv_id for s in summaries}
    for c in concepts:
        for dep in c.depends_on:
            if dep in paper_ids:
                edges.append({"source": c.slug, "target": dep, "kind": "depends_on"})

    graph = {"nodes": nodes, "edges": edges}
    return json.dumps(graph, indent=2) + "\n"


# ----------------------------------------------------------- top-level build


def build_all(
    papers_dir: Path,
    concepts_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Rebuild all five indices. Returns a count summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries_dir(papers_dir)
    concepts = load_concepts_dir(concepts_dir)

    (output_dir / "INDEX.md").write_text(render_index(summaries, concepts))
    (output_dir / "GLOSSARY.md").write_text(render_glossary(concepts))
    (output_dir / "TIMELINE.md").write_text(render_timeline(summaries))
    (output_dir / "QUESTIONS.md").write_text(render_questions(concepts))
    (output_dir / "GRAPH.json").write_text(render_graph(summaries, concepts))

    return {
        "n_papers": len(summaries),
        "n_concepts": len(concepts),
        "n_edges": sum(
            1
            for c in concepts
            for dep in c.depends_on
            if dep in {s.arxiv_id for s in summaries}
        ),
    }
