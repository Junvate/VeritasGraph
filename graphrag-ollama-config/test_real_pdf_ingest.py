"""
Real PDF ingest smoke test.

Usage:
    ../.venv/bin/python test_real_pdf_ingest.py "../VeritasGraph - A Sovereign GraphRAG Framework for Enterprise-Grade AI with Verifiable Attribution.pdf"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingest import ingest_pdf_file
from pdf_source_map import load_source_map, resolve_locator_from_source_map


def main():
    if len(sys.argv) < 2:
        print("❌ 请提供 PDF 路径")
        return

    pdf_path = sys.argv[1]
    success, message, filepath = ingest_pdf_file(pdf_path)
    print(message)
    if not success or not filepath:
        return

    document_title = os.path.basename(filepath)
    source_map = load_source_map(document_title)
    print(f"source_map loaded: {source_map is not None}")
    if not source_map:
        return

    print(f"segments: {len(source_map.get('segments', []))}")
    locator = resolve_locator_from_source_map(
        source_map=source_map,
        char_start=684,
        char_end=886,
        snippet="Organizations everywhere are rushing to deploy RAG solutions",
    )
    print("locator:", locator)


if __name__ == "__main__":
    main()
