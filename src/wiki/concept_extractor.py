"""Concept candidate extractor for the SciRAG wiki.

Walks the compiled paper summaries in ``wiki/papers/`` and returns a
ranked list of concept candidates with supporting evidence
``[(arxiv_id, snippet), ...]``. Arnav reviews the candidate list and
picks the 15-20 concepts that actually get articles compiled
(see ``scripts/compile_concepts.py`` in SB10.3).

Pure regex-based; no spaCy or NLTK dependency. Two signal sources:

1. Bare acronyms (``BERT``, ``GLUE``, ``SQuAD``) — 2-6 uppercase
   characters, optionally followed by a digit or lowercase suffix.
2. Capitalized multi-word phrases ("Masked Language Model",
   "Cross-Encoder Reranker") — 2-4 capitalized words in sequence.

Candidates are normalized (lowercase, plural-stripped via a tiny
heuristic) and counted across summaries. Stopword-like junk
("The Authors", "This Paper") is filtered out.

The module reads sections from the markdown rendered by
``src/wiki/summarizer.py`` — it relies on the fixed heading shape
(``## TL;DR``, ``## Problem``, ``## Method``, ``## Results``,
``## Limitations``).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SECTION_HEADS = ("TL;DR", "Problem", "Method", "Results", "Limitations")
SIGNAL_SECTIONS = ("TL;DR", "Method")  # where concepts actually live

ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2}[A-Za-z0-9]{0,5}(?:-?\d+)?\b")
PHRASE_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+(?:-[A-Z][a-z]+)?)(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?){1,3}\b")

STOPWORD_CONCEPTS = frozenset({
    "the authors",
    "this paper",
    "the paper",
    "this work",
    "the work",
    "the model",
    "the method",
    "the approach",
    "our model",
    "our approach",
    "our method",
    "the results",
    "we propose",
    "the authors propose",
    "the authors present",
    "unknown",
    "tl;dr",
})

ACRONYM_STOPWORDS = frozenset({
    "TL",
    "DR",
    "ID",
    "URL",
    "CPU",
    "GPU",
    "RAM",
    "API",
    "PDF",
    "HTML",
    "JSON",
    "XML",
    "YAML",
})


@dataclass(frozen=True)
class Summary:
    arxiv_id: str
    title: str
    sections: dict[str, str] = field(default_factory=dict)

    def signal_text(self) -> str:
        return " ".join(self.sections.get(s, "") for s in SIGNAL_SECTIONS).strip()


@dataclass(frozen=True)
class ConceptEvidence:
    arxiv_id: str
    snippet: str


def load_summary(path: Path) -> Summary | None:
    """Parse a wiki/papers/*.md file. Returns None if frontmatter is malformed."""
    text = path.read_text()
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    frontmatter = text[4:end]
    body = text[end + 5 :]

    arxiv_id = ""
    title = ""
    for line in frontmatter.splitlines():
        if line.startswith("arxiv_id:"):
            arxiv_id = line.split(":", 1)[1].strip()
        elif line.startswith("title:"):
            raw = line.split(":", 1)[1].strip()
            title = raw.strip('"')

    sections = _parse_sections(body)
    return Summary(arxiv_id=arxiv_id, title=title, sections=sections)


def _parse_sections(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in body.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m and m.group(1) in SECTION_HEADS:
            if current is not None:
                out[current] = "\n".join(buf).strip()
            current = m.group(1)
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf).strip()
    return out


def extract_candidates(text: str) -> list[str]:
    """Return raw concept-candidate strings from a block of text.

    No normalization, no dedup, no filtering — that happens in
    :func:`rank_concepts` so callers can inspect raw matches.
    """
    if not text:
        return []
    out: list[str] = []
    for m in ACRONYM_PATTERN.finditer(text):
        token = m.group(0)
        if token in ACRONYM_STOPWORDS:
            continue
        out.append(token)
    for m in PHRASE_PATTERN.finditer(text):
        out.append(m.group(0))
    return out


def _normalize(candidate: str) -> str:
    """Lowercase + strip a trailing 's' if it survives as a 4+ char word."""
    norm = candidate.lower().strip()
    # crude plural strip: only on the last word, only if >= 4 chars
    parts = norm.split()
    if parts and len(parts[-1]) >= 5 and parts[-1].endswith("s") and not parts[-1].endswith("ss"):
        parts[-1] = parts[-1][:-1]
    return " ".join(parts)


def _snippet_for(text: str, raw_candidate: str, *, window: int = 80) -> str:
    """Return a ±window-char snippet around the first occurrence of raw_candidate."""
    idx = text.find(raw_candidate)
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end = min(len(text), idx + len(raw_candidate) + window)
    snippet = text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def rank_concepts(
    summaries: list[Summary],
    *,
    top_n: int = 50,
    min_paper_count: int = 2,
) -> list[tuple[str, int, list[ConceptEvidence]]]:
    """Rank concept candidates by # of distinct papers they appear in.

    Returns list of (normalized_concept, paper_count, evidence) sorted
    descending. Concepts mentioned in fewer than ``min_paper_count``
    papers are dropped — singleton concepts aren't useful for cross-paper
    wiki articles.
    """
    paper_counts: dict[str, set[str]] = defaultdict(set)
    evidence: dict[str, list[ConceptEvidence]] = defaultdict(list)
    seen_in_paper: dict[str, set[str]] = defaultdict(set)

    for s in summaries:
        text = s.signal_text()
        if not text:
            continue
        for raw in extract_candidates(text):
            norm = _normalize(raw)
            if norm in STOPWORD_CONCEPTS or len(norm) < 2:
                continue
            paper_counts[norm].add(s.arxiv_id)
            if norm not in seen_in_paper[s.arxiv_id]:
                snippet = _snippet_for(text, raw)
                if snippet:
                    evidence[norm].append(ConceptEvidence(arxiv_id=s.arxiv_id, snippet=snippet))
                    seen_in_paper[s.arxiv_id].add(norm)

    ranked = [
        (concept, len(paper_ids), evidence[concept])
        for concept, paper_ids in paper_counts.items()
        if len(paper_ids) >= min_paper_count
    ]
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked[:top_n]


def load_summaries_dir(papers_dir: Path) -> list[Summary]:
    """Load every wiki/papers/*.md (excluding REVIEW_NOTES.md)."""
    out: list[Summary] = []
    for p in sorted(papers_dir.glob("*.md")):
        if p.name == "REVIEW_NOTES.md":
            continue
        s = load_summary(p)
        if s is not None:
            out.append(s)
    return out
