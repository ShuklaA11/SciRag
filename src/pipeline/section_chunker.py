"""Section-aware chunker over Grobid TEI XML.

Tags each chunk with one of 8 canonical section_type buckets so retrieval
can be filtered/routed downstream. Reuses the offset-mapping windowing
strategy from the flat chunker; the only structural change is that each
section is windowed independently — chunks never bleed across sections.

Section taxonomy (8 buckets):
  abstract, introduction, related_work, method, experiments,
  results, conclusion, other
"""

from __future__ import annotations

import re

from .grobid_client import extract_abstract, extract_sections, extract_title

SECTION_TYPES = (
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiments",
    "results",
    "conclusion",
    "other",
)

# Order matters: more-specific patterns first; "introduction" runs late so
# heads like "Model Overview" claim "method" before introduction's
# "overview" alternative fires.
_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("abstract",     re.compile(r"\babstract\b", re.IGNORECASE)),
    ("related_work", re.compile(
        r"\brelated\s+work\b|\bbackground\b|\bprior\s+work\b|\bliterature\b",
        re.IGNORECASE)),
    ("method",       re.compile(
        r"\bmethod|\bapproach|\barchitecture|\bmodel\b|\balgorithm|"
        r"\bformulation|\bproposed\b",
        re.IGNORECASE)),
    ("experiments",  re.compile(
        r"\bexperiment|\bsetup\b|\bimplementation\b|\btraining\b|\bdataset",
        re.IGNORECASE)),
    ("results",      re.compile(
        r"\bresult|\bevaluation\b|\bfinding|\banalysis\b|\bablation",
        re.IGNORECASE)),
    ("conclusion",   re.compile(
        r"\bconclusion|\bdiscussion\b|\bfuture\s+work\b|\blimitation",
        re.IGNORECASE)),
    ("introduction", re.compile(
        r"\bintroduction\b|\bmotivation\b|\boverview\b",
        re.IGNORECASE)),
]

# Strip leading numbering like "3.", "4.1", "IV.", "1)" etc.
_LEAD_NUM_RE = re.compile(
    r"^\s*(?:[IVXLCDM]+\.|[0-9]+(?:\.[0-9]+)*\.?|[0-9]+\))\s*",
    re.IGNORECASE,
)


def section_type_for_head(head: str) -> str:
    """Map a Grobid section head string to one of SECTION_TYPES.

    Strips leading numbering (e.g. "3.", "4.1", "IV.") before matching.
    Returns 'other' for [untitled], empty, or no-match.
    """
    if not head:
        return "other"
    s = _LEAD_NUM_RE.sub("", head).strip()
    if not s or s == "[untitled]":
        return "other"
    for label, pat in _SECTION_PATTERNS:
        if pat.search(s):
            return label
    return "other"


def _parent_n(n: str) -> str:
    """'3.1.2' -> '3.1'; '3' -> ''; '' -> ''."""
    if not n or "." not in n:
        return ""
    return n.rsplit(".", 1)[0]


def resolve_section_types(sections: list[dict]) -> list[str]:
    """Classify each section, inheriting parent type when own head has no
    structural keyword.

    Walks sections in document order. A section's type is its own
    head-classified type unless that's 'other' and its dotted-numeric
    parent (or any ancestor) has a non-'other' resolved type — in which
    case it inherits the closest such ancestor.

    Sections with no `n` (or no ancestor in the seen set) stay 'other'.
    """
    resolved_by_n: dict[str, str] = {}
    out: list[str] = []
    for s in sections:
        own = section_type_for_head(s.get("head", ""))
        n = s.get("n", "") or ""
        if own != "other" or not n:
            out.append(own)
            if n:
                resolved_by_n[n] = own
            continue
        # own == "other" and n is set: walk ancestors.
        ancestor = _parent_n(n)
        inherited = "other"
        while ancestor:
            t = resolved_by_n.get(ancestor)
            if t and t != "other":
                inherited = t
                break
            ancestor = _parent_n(ancestor)
        out.append(inherited)
        resolved_by_n[n] = inherited
    return out


def _window_section(
    text: str,
    section_type: str,
    section_head: str,
    tokenizer,
    chunk_size: int,
    overlap: int,
    start_chunk_idx: int,
) -> list[dict]:
    """Slice one section's text into overlapping token windows.

    Returns chunks with global chunk_idx assigned starting from
    `start_chunk_idx`.
    """
    if not text.strip():
        return []
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    if not token_ids:
        return []

    out: list[dict] = []
    stride = chunk_size - overlap
    start = 0
    idx = start_chunk_idx
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk_text = text[char_start:char_end].strip()
        if chunk_text:
            out.append({
                "chunk_idx": idx,
                "text": chunk_text,
                "token_count": end - start,
                "section_type": section_type,
                "section_head": section_head,
            })
            idx += 1
        if end >= len(token_ids):
            break
        start += stride
    return out


def _dominant_section(
    char_start: int,
    char_end: int,
    spans: list[tuple[int, int, str, str]],
) -> tuple[str, str]:
    """Return (section_type, section_head) of the span that contributes
    the most characters to ``[char_start, char_end)``. Ties broken by
    earliest occurrence in document order. Returns ('other', '[untitled]')
    when no span overlaps (shouldn't happen if spans cover full_text).
    """
    overlaps: dict[str, int] = {}
    first_idx: dict[str, int] = {}
    head_for_type: dict[str, str] = {}
    for i, (s, e, t, h) in enumerate(spans):
        ovl = min(char_end, e) - max(char_start, s)
        if ovl > 0:
            overlaps[t] = overlaps.get(t, 0) + ovl
            if t not in first_idx:
                first_idx[t] = i
                head_for_type[t] = h
    if not overlaps:
        return "other", "[untitled]"
    best = max(overlaps.keys(), key=lambda t: (overlaps[t], -first_idx[t]))
    return best, head_for_type[best]


def chunk_paper_section_tagged(
    tei_xml: str,
    tokenizer,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    """Flat-style 512-token windowing with per-chunk section tags.

    Builds the same concatenated full_text as ``chunker.chunk_paper`` so
    chunks are byte-for-byte identical, then tags each chunk with the
    section_type that contributes the most characters to it (ties go to
    the earliest section in document order).

    Returns chunks with the same shape as ``chunk_paper_by_section``:
    ``{chunk_idx, text, token_count, section_type, section_head}``.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
    if not tokenizer.is_fast:
        raise ValueError(
            "chunk_paper_section_tagged requires a fast tokenizer (offset_mapping)."
        )

    title = extract_title(tei_xml)
    abstract = extract_abstract(tei_xml)
    sections = extract_sections(tei_xml)

    parts: list[tuple[str, str, str]] = []
    if title and title != "[unknown title]":
        if abstract:
            title_type = "abstract"
        elif sections:
            title_type = section_type_for_head(sections[0].get("head", ""))
        else:
            title_type = "other"
        parts.append((title, title_type, "Title"))

    if abstract:
        parts.append((abstract, "abstract", "Abstract"))

    resolved_types = resolve_section_types(sections)
    for s, sec_type in zip(sections, resolved_types):
        head = s["head"] if s["head"] and s["head"] != "[untitled]" else ""
        body = s["text"]
        sec_text = f"{head}\n{body}" if head else body
        parts.append((sec_text, sec_type, head or "[untitled]"))

    if not parts:
        return []

    full_text = "\n\n".join(p[0] for p in parts)
    if not full_text.strip():
        return []

    spans: list[tuple[int, int, str, str]] = []
    cursor = 0
    for i, (text, sec_type, head) in enumerate(parts):
        if i > 0:
            cursor += 2  # "\n\n"
        spans.append((cursor, cursor + len(text), sec_type, head))
        cursor += len(text)

    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    if not token_ids:
        return []

    chunks: list[dict] = []
    stride = chunk_size - overlap
    start = 0
    idx = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        text = full_text[char_start:char_end].strip()
        if text:
            sec_type, head = _dominant_section(char_start, char_end, spans)
            chunks.append({
                "chunk_idx": idx,
                "text": text,
                "token_count": end - start,
                "section_type": sec_type,
                "section_head": head,
            })
            idx += 1
        if end >= len(token_ids):
            break
        start += stride

    return chunks


def chunk_paper_by_section(
    tei_xml: str,
    tokenizer,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    """Per-section windowed chunking.

    Layout:
      - Title is prepended to the abstract chunk's text. If no abstract,
        title is prepended to the first chunk of the first section.
      - Abstract becomes one section with section_type='abstract' and
        section_head='Abstract'.
      - Each <div> from extract_sections gets its own windowing pass.
        section_type is computed once per section via section_type_for_head.
      - chunk_idx is global across the whole paper; chunks within one
        section are contiguous.

    Returns []: paper has no extractable text.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
    if not tokenizer.is_fast:
        raise ValueError(
            "chunk_paper_by_section requires a fast tokenizer (offset_mapping)."
        )

    title = extract_title(tei_xml)
    abstract = extract_abstract(tei_xml)
    sections = extract_sections(tei_xml)

    title_prefix = ""
    if title and title != "[unknown title]":
        title_prefix = title + "\n\n"

    chunks: list[dict] = []
    next_idx = 0
    title_consumed = False

    if abstract:
        abs_text = (title_prefix + abstract) if title_prefix else abstract
        title_consumed = True
        new = _window_section(
            abs_text, "abstract", "Abstract",
            tokenizer, chunk_size, overlap, next_idx,
        )
        chunks.extend(new)
        next_idx += len(new)

    resolved_types = resolve_section_types(sections)
    for s, sec_type in zip(sections, resolved_types):
        head = s["head"] if s["head"] and s["head"] != "[untitled]" else ""
        body = s["text"]
        sec_text_parts: list[str] = []
        if not title_consumed:
            sec_text_parts.append(title.strip())
            title_consumed = True
        if head:
            sec_text_parts.append(head)
        sec_text_parts.append(body)
        sec_text = "\n".join(p for p in sec_text_parts if p)
        new = _window_section(
            sec_text, sec_type, head or "[untitled]",
            tokenizer, chunk_size, overlap, next_idx,
        )
        chunks.extend(new)
        next_idx += len(new)

    return chunks
