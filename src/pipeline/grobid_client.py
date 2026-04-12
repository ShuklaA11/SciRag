"""Thin client for the Grobid REST API + TEI XML section extraction."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import requests

GROBID_URL = "http://localhost:8070"
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def process_pdf(pdf_path: str | Path, grobid_url: str = GROBID_URL) -> str:
    """Send a PDF to Grobid's fulltext endpoint. Returns raw TEI XML string."""
    pdf_path = Path(pdf_path)
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{grobid_url}/api/processFulltextDocument",
            files={"input": (pdf_path.name, f, "application/pdf")},
            timeout=120,
        )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(
            f"Grobid failed on '{pdf_path.name}' (HTTP {resp.status_code}): {resp.text[:200]}"
        ) from e
    return resp.text


def extract_sections(tei_xml: str, *, min_text_len: int = 10) -> list[dict]:
    """Parse TEI XML and return a list of sections with head + text.

    Args:
        tei_xml: Raw TEI XML string from Grobid.
        min_text_len: Drop sections with fewer chars than this (filters figure
            captions and table labels that Grobid mis-classifies as sections).

    Returns:
        [{"head": "Introduction", "text": "...", "n": "1"}, ...]
        Sections without a heading get head="[untitled]".
    """
    root = ET.fromstring(tei_xml)
    body = root.find(".//tei:body", TEI_NS)
    if body is None:
        return []

    sections = []
    for div in body.findall(".//tei:div", TEI_NS):
        head_el = div.find("tei:head", TEI_NS)
        head = head_el.text.strip() if head_el is not None and head_el.text else "[untitled]"
        n = head_el.get("n", "") if head_el is not None else ""

        paragraphs = []
        for p in div.findall("tei:p", TEI_NS):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

        joined = "\n".join(paragraphs)
        if len(joined) >= min_text_len:
            sections.append({"head": head, "n": n, "text": joined})

    return sections


def extract_title(tei_xml: str) -> str:
    """Extract the paper title from TEI XML."""
    root = ET.fromstring(tei_xml)
    title_el = root.find(".//tei:titleStmt/tei:title", TEI_NS)
    if title_el is not None and title_el.text:
        return title_el.text.strip()
    return "[unknown title]"


def extract_abstract(tei_xml: str) -> str:
    """Extract the abstract text from TEI XML."""
    root = ET.fromstring(tei_xml)
    abstract_el = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
    if abstract_el is None:
        return ""
    return " ".join("".join(p.itertext()).strip() for p in abstract_el.findall(".//tei:p", TEI_NS))
