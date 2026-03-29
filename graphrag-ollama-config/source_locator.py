"""
Source location utilities for traceable GraphRAG answers.

Current implementation focuses on text-based sources and returns paragraph/character
locators. The schema intentionally leaves room for future page/bbox enrichment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import pandas as pd

join = os.path.join


@dataclass
class ParagraphLocator:
    paragraph_index: int
    paragraph_text: str
    char_start: Optional[int]
    char_end: Optional[int]
    page: Optional[int] = None
    bbox: Optional[dict[str, float]] = None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip().lower()


def load_source_tables(input_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load document and text unit tables for evidence tracing."""
    text_units = pd.read_parquet(join(input_dir, "create_final_text_units.parquet"))
    documents = pd.read_parquet(join(input_dir, "create_final_documents.parquet"))
    return text_units, documents


def split_document_paragraphs(raw_content: str) -> List[Dict[str, Any]]:
    """Split a document into simple paragraph blocks with character ranges."""
    raw_content = str(raw_content or "")
    paragraphs: List[Dict[str, Any]] = []
    cursor = 0
    for block in re.split(r"\n\s*\n", raw_content):
        text = block.strip()
        if not text:
            continue
        start = raw_content.find(text, cursor)
        if start < 0:
            start = cursor
        end = start + len(text)
        paragraphs.append(
            {
                "paragraph_index": len(paragraphs) + 1,
                "paragraph_text": text,
                "char_start": start,
                "char_end": end,
            }
        )
        cursor = end
    return paragraphs


def locate_text_in_document(snippet: str, raw_content: str) -> ParagraphLocator:
    """Locate the best matching paragraph for a text snippet."""
    snippet = str(snippet or "").strip()
    raw_content = str(raw_content or "")
    paragraphs = split_document_paragraphs(raw_content)

    if not paragraphs:
        return ParagraphLocator(
            paragraph_index=1,
            paragraph_text=raw_content[:300],
            char_start=0 if raw_content else None,
            char_end=min(len(raw_content), 300) if raw_content else None,
        )

    exact_start = raw_content.find(snippet) if snippet else -1
    if exact_start >= 0:
        exact_end = exact_start + len(snippet)
        for paragraph in paragraphs:
            if paragraph["char_start"] <= exact_start < paragraph["char_end"]:
                return ParagraphLocator(
                    paragraph_index=paragraph["paragraph_index"],
                    paragraph_text=paragraph["paragraph_text"],
                    char_start=exact_start,
                    char_end=exact_end,
                )

    normalized_snippet = _normalize_text(snippet)
    best_paragraph = paragraphs[0]
    best_score = -1.0
    for paragraph in paragraphs:
        score = SequenceMatcher(
            None,
            _normalize_text(paragraph["paragraph_text"]),
            normalized_snippet,
        ).ratio()
        if score > best_score:
            best_score = score
            best_paragraph = paragraph

    return ParagraphLocator(
        paragraph_index=best_paragraph["paragraph_index"],
        paragraph_text=best_paragraph["paragraph_text"],
        char_start=best_paragraph["char_start"],
        char_end=best_paragraph["char_end"],
    )


def build_evidence_item(
    text_unit_row: pd.Series,
    document_row: Optional[pd.Series],
    source_type: str = "text_unit",
) -> Dict[str, Any]:
    """Build a normalized evidence item with document locator details."""
    text = str(text_unit_row.get("text", "") or "")
    locator = locate_text_in_document(
        snippet=text,
        raw_content=str(document_row.get("raw_content", "") if document_row is not None else ""),
    )

    document_title = None
    document_id = None
    if document_row is not None:
        document_title = document_row.get("title")
        document_id = document_row.get("id")

    return {
        "source_type": source_type,
        "text_unit_id": str(text_unit_row.get("id", "")),
        "document_id": str(document_id) if document_id is not None else None,
        "document_title": str(document_title) if document_title is not None else None,
        "snippet": text[:260] + ("..." if len(text) > 260 else ""),
        "locator": {
            "paragraph_index": locator.paragraph_index,
            "paragraph_text": locator.paragraph_text[:320]
            + ("..." if len(locator.paragraph_text) > 320 else ""),
            "char_start": locator.char_start,
            "char_end": locator.char_end,
            "page": locator.page,
            "bbox": locator.bbox,
        },
    }
