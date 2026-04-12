"""Tests for src/pipeline/grobid_client.py — section filtering and error handling."""

import pytest
import requests

from src.pipeline.grobid_client import extract_sections, process_pdf

# Minimal valid TEI XML with a short and a long section
_TEI_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader/>
  <text>
    <body>
      <div>
        <head n="1">Introduction</head>
        <p>This is the introduction with enough text to pass the filter.</p>
      </div>
      <div>
        <head n="">Fig 1</head>
        <p>xy</p>
      </div>
      <div>
        <head n="2">Methods</head>
        <p>Detailed methods section describing the experimental setup and procedures.</p>
      </div>
    </body>
  </text>
</TEI>
"""


# --- min_text_len filtering ---


def test_extract_sections_filters_short_sections():
    sections = extract_sections(_TEI_XML, min_text_len=10)
    heads = [s["head"] for s in sections]
    assert "Introduction" in heads
    assert "Methods" in heads
    assert "Fig 1" not in heads


def test_extract_sections_default_filters_tiny():
    sections = extract_sections(_TEI_XML)
    # "xy" is 2 chars, below default min_text_len=10
    assert all(len(s["text"]) >= 10 for s in sections)


def test_extract_sections_min_text_len_zero_keeps_all():
    sections = extract_sections(_TEI_XML, min_text_len=0)
    heads = [s["head"] for s in sections]
    assert "Fig 1" in heads


def test_extract_sections_high_threshold_filters_more():
    sections = extract_sections(_TEI_XML, min_text_len=10000)
    assert sections == []


def test_extract_sections_empty_body():
    xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader/>
  <text><body/></text>
</TEI>
"""
    assert extract_sections(xml) == []


# --- process_pdf error context ---


def _grobid_is_up() -> bool:
    try:
        return requests.get("http://localhost:8070/api/isalive", timeout=2).ok
    except requests.ConnectionError:
        return False


@pytest.mark.skipif(not _grobid_is_up(), reason="Grobid not running")
def test_process_pdf_error_includes_filename(tmp_path):
    """Grobid errors should mention the PDF filename."""
    bad_pdf = tmp_path / "broken.pdf"
    bad_pdf.write_bytes(b"not a pdf")

    with pytest.raises(requests.HTTPError, match="broken.pdf"):
        process_pdf(bad_pdf)
