"""Flat token-window chunker over Grobid TEI XML.

This is the Week 3 *flat* baseline: title + abstract + all sections concatenated
in document order, then sliced into fixed token windows with overlap. No
section awareness — that's Week 4.
"""

from __future__ import annotations

from .grobid_client import extract_abstract, extract_sections, extract_title


def _build_full_text(tei_xml: str) -> str:
    title = extract_title(tei_xml)
    abstract = extract_abstract(tei_xml)
    sections = extract_sections(tei_xml)

    parts: list[str] = []
    if title and title != "[unknown title]":
        parts.append(title)
    if abstract:
        parts.append(abstract)
    for s in sections:
        head = s["head"]
        if head and head != "[untitled]":
            parts.append(f"{head}\n{s['text']}")
        else:
            parts.append(s["text"])
    return "\n\n".join(parts)


def chunk_paper(
    tei_xml: str,
    tokenizer,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    """Slice a paper into overlapping token windows.

    Args:
        tei_xml: Raw Grobid TEI XML.
        tokenizer: A HuggingFace tokenizer (used without special tokens).
        chunk_size: Max tokens per chunk. Should match the embedder's max input.
        overlap: Tokens shared between consecutive chunks.

    Returns:
        [{"chunk_idx": 0, "text": "...", "token_count": 487}, ...]
        Empty list if the paper has no extractable text.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")

    full_text = _build_full_text(tei_xml)
    if not full_text.strip():
        return []

    token_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if not token_ids:
        return []

    chunks: list[dict] = []
    stride = chunk_size - overlap
    start = 0
    idx = 0
    while start < len(token_ids):
        window = token_ids[start : start + chunk_size]
        text = tokenizer.decode(window, skip_special_tokens=True).strip()
        if text:
            chunks.append(
                {"chunk_idx": idx, "text": text, "token_count": len(window)}
            )
            idx += 1
        if start + chunk_size >= len(token_ids):
            break
        start += stride

    return chunks
