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
        tokenizer: A HuggingFace fast tokenizer (used without special tokens).
        chunk_size: Max tokens per chunk. Should match the embedder's max input.
        overlap: Tokens shared between consecutive chunks.

    Returns:
        [{"chunk_idx": 0, "text": "...", "token_count": 487}, ...]
        Empty list if the paper has no extractable text.

    Chunk text is sliced directly from the source string using the
    tokenizer's offset_mapping — case, hyphens, and citation tokens like
    BIBREF19 are preserved verbatim. Encoding through the BERT-uncased
    tokenizer and decoding back lowercases everything and inserts spaces
    around punctuation, so we never round-trip through decode().
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
    if not tokenizer.is_fast:
        raise ValueError(
            "chunk_paper requires a fast tokenizer (offset_mapping). "
            "Load with AutoTokenizer.from_pretrained(..., use_fast=True)."
        )

    full_text = _build_full_text(tei_xml)
    if not full_text.strip():
        return []

    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
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
            chunks.append(
                {"chunk_idx": idx, "text": text, "token_count": end - start}
            )
            idx += 1
        if end >= len(token_ids):
            break
        start += stride

    return chunks
