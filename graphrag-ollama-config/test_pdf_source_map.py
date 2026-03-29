"""
Unit-style smoke test for PDF source map resolution.

This test does not require a real PDF parser. It verifies that the source-map
resolution logic can map a text span back to page / paragraph / bbox metadata.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_source_map import resolve_locator_from_source_map


def main():
    source_map = {
        "schema_version": 1,
        "document_title": "demo_pdf.txt",
        "source_type": "pdf",
        "text": "第一页第一段。\n\n第一页第二段。\n\n第二页第一段。",
        "segments": [
            {
                "id": "p1-seg1",
                "page": 1,
                "paragraph_index": 1,
                "char_start": 0,
                "char_end": 7,
                "bbox": {"x0": 10, "top": 10, "x1": 100, "bottom": 30},
                "text": "第一页第一段。",
            },
            {
                "id": "p1-seg2",
                "page": 1,
                "paragraph_index": 2,
                "char_start": 9,
                "char_end": 16,
                "bbox": {"x0": 10, "top": 40, "x1": 100, "bottom": 60},
                "text": "第一页第二段。",
            },
            {
                "id": "p2-seg1",
                "page": 2,
                "paragraph_index": 1,
                "char_start": 18,
                "char_end": 25,
                "bbox": {"x0": 10, "top": 10, "x1": 100, "bottom": 30},
                "text": "第二页第一段。",
            },
        ],
    }

    locator = resolve_locator_from_source_map(
        source_map=source_map,
        char_start=9,
        char_end=16,
        snippet="第一页第二段。",
    )

    assert locator["page"] == 1, locator
    assert locator["paragraph_index"] == 2, locator
    assert locator["bbox"]["top"] == 40, locator

    print("✅ source map resolution ok")
    print(locator)


if __name__ == "__main__":
    main()
