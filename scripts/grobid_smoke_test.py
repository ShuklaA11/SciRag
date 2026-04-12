"""Grobid smoke test: process 5 NLP papers and verify section extraction.

Usage:
    python scripts/grobid_smoke_test.py

Requires Grobid running on localhost:8070 and PDFs in raw/papers/.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.grobid_client import (
    extract_abstract,
    extract_sections,
    extract_title,
    process_pdf,
)

PAPERS_DIR = Path(__file__).resolve().parent.parent / "raw" / "papers"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "grobid_output"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PAPERS_DIR}")
        return

    print(f"Found {len(pdfs)} PDFs in {PAPERS_DIR}\n")
    results = []

    for pdf in pdfs:
        print(f"Processing: {pdf.name}")
        try:
            tei_xml = process_pdf(pdf)

            # Save raw TEI XML
            xml_path = OUTPUT_DIR / f"{pdf.stem}.xml"
            xml_path.write_text(tei_xml)

            title = extract_title(tei_xml)
            abstract = extract_abstract(tei_xml)
            sections = extract_sections(tei_xml)

            result = {
                "file": pdf.name,
                "title": title,
                "abstract_len": len(abstract),
                "num_sections": len(sections),
                "sections": [
                    {"n": s["n"], "head": s["head"], "text_len": len(s["text"])}
                    for s in sections
                ],
            }
            results.append(result)

            print(f"  Title: {title}")
            print(f"  Abstract: {len(abstract)} chars")
            print(f"  Sections: {len(sections)}")
            for s in sections:
                print(f"    {s['n']:>4s} {s['head'][:50]:<50s} ({len(s['text']):,} chars)")
            print()

        except Exception as e:
            print(f"  FAILED: {e}\n")
            results.append({"file": pdf.name, "error": str(e)})

    # Save summary
    summary_path = OUTPUT_DIR / "smoke_test_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    # Report
    ok = sum(1 for r in results if "error" not in r)
    print(f"{'=' * 60}")
    print(f"Results: {ok}/{len(results)} papers processed successfully")
    print(f"TEI XML saved to: {OUTPUT_DIR}")

    if ok < len(results):
        failed = [r["file"] for r in results if "error" in r]
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
