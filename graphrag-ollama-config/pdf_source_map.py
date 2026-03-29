"""
PDF source-map utilities.

This module extracts page / paragraph / bbox metadata from PDFs into a sidecar
JSON file that can be used later by the explanation engine for provenance.

Design goals:
- do not require GraphRAG parquet schema changes
- keep the source map independent and reusable
- degrade gracefully when PDF parsing libraries are unavailable
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

try:
    import pdfplumber  # type: ignore
    PDFPLUMBER_AVAILABLE = True
except Exception:
    pdfplumber = None
    PDFPLUMBER_AVAILABLE = False


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
SOURCE_MAP_DIR = os.path.join(INPUT_DIR, ".source_maps")


def ensure_source_map_dir() -> str:
    os.makedirs(SOURCE_MAP_DIR, exist_ok=True)
    return SOURCE_MAP_DIR


def source_map_available() -> bool:
    return PDFPLUMBER_AVAILABLE


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _document_title_hash(document_title: str) -> str:
    return hashlib.md5(document_title.encode("utf-8")).hexdigest()


def source_map_path_for_title(document_title: str) -> str:
    ensure_source_map_dir()
    return os.path.join(SOURCE_MAP_DIR, f"{_document_title_hash(document_title)}.json")


def save_source_map(document_title: str, source_map: dict[str, Any]) -> str:
    path = source_map_path_for_title(document_title)
    payload = dict(source_map)
    payload["document_title"] = document_title
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_source_map(document_title: str) -> dict[str, Any] | None:
    path = source_map_path_for_title(document_title)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_source_map(document_title: str) -> None:
    path = source_map_path_for_title(document_title)
    if os.path.exists(path):
        os.remove(path)


def shift_source_map_offsets(source_map: dict[str, Any], char_offset: int) -> dict[str, Any]:
    payload = dict(source_map)
    payload["segments"] = []
    payload["text"] = (" " * char_offset) + str(source_map.get("text", "") or "")
    for segment in source_map.get("segments", []):
        item = dict(segment)
        if item.get("char_start") is not None:
            item["char_start"] = int(item["char_start"]) + char_offset
        if item.get("char_end") is not None:
            item["char_end"] = int(item["char_end"]) + char_offset
        payload["segments"].append(item)
    return payload


def _group_words_to_paragraphs(words: list[dict[str, Any]], page_number: int) -> list[dict[str, Any]]:
    if not words:
        return []

    lines: list[list[dict[str, Any]]] = []
    current_line: list[dict[str, Any]] = []
    current_top: float | None = None
    line_tol = 3.0

    for word in words:
        top = _safe_float(word.get("top"), 0.0)
        if current_top is None or abs(top - current_top) <= line_tol:
            current_line.append(word)
            current_top = top if current_top is None else (current_top + top) / 2
        else:
            lines.append(current_line)
            current_line = [word]
            current_top = top
    if current_line:
        lines.append(current_line)

    paragraphs: list[dict[str, Any]] = []
    current_para: list[list[dict[str, Any]]] = []
    prev_bottom: float | None = None
    para_gap = 14.0

    for line in lines:
        line_top = min(_safe_float(item.get("top"), 0.0) for item in line)
        line_bottom = max(_safe_float(item.get("bottom"), 0.0) for item in line)
        if prev_bottom is None or (line_top - prev_bottom) <= para_gap:
            current_para.append(line)
        else:
            paragraphs.append(_build_paragraph(current_para, page_number, len(paragraphs) + 1))
            current_para = [line]
        prev_bottom = line_bottom
    if current_para:
        paragraphs.append(_build_paragraph(current_para, page_number, len(paragraphs) + 1))

    return [item for item in paragraphs if item.get("text", "").strip()]


def _build_paragraph(lines: list[list[dict[str, Any]]], page_number: int, paragraph_index: int) -> dict[str, Any]:
    line_texts = []
    x0_values = []
    top_values = []
    x1_values = []
    bottom_values = []

    for line in lines:
        line_sorted = sorted(line, key=lambda item: _safe_float(item.get("x0"), 0.0))
        text = " ".join(str(item.get("text", "")).strip() for item in line_sorted if str(item.get("text", "")).strip())
        if text:
            line_texts.append(text)
        for item in line_sorted:
            x0_values.append(_safe_float(item.get("x0"), 0.0))
            top_values.append(_safe_float(item.get("top"), 0.0))
            x1_values.append(_safe_float(item.get("x1"), 0.0))
            bottom_values.append(_safe_float(item.get("bottom"), 0.0))

    return {
        "page": page_number,
        "paragraph_index": paragraph_index,
        "text": "\n".join(line_texts).strip(),
        "bbox": {
            "x0": round(min(x0_values), 2) if x0_values else 0.0,
            "top": round(min(top_values), 2) if top_values else 0.0,
            "x1": round(max(x1_values), 2) if x1_values else 0.0,
            "bottom": round(max(bottom_values), 2) if bottom_values else 0.0,
        },
    }


def extract_pdf_source_map(pdf_path: str) -> tuple[bool, str, dict[str, Any] | None]:
    """
    Extract full text plus segment-level page/bbox metadata from a PDF.
    Returns (success, message, source_map).
    """
    if not PDFPLUMBER_AVAILABLE:
        return False, "pdfplumber 未安装，暂时无法提取 PDF 页码与版面坐标。", None

    if not pdf_path or not os.path.exists(pdf_path):
        return False, "PDF 文件不存在。", None

    try:
        full_parts: list[str] = []
        segments: list[dict[str, Any]] = []
        cursor = 0

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                words = page.extract_words(
                    x_tolerance=2,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=True,
                ) or []

                if words:
                    paragraphs = _group_words_to_paragraphs(words, page_number)
                else:
                    page_text = (page.extract_text() or "").strip()
                    paragraphs = [{
                        "page": page_number,
                        "paragraph_index": 1,
                        "text": page_text,
                        "bbox": None,
                    }] if page_text else []

                for idx, paragraph in enumerate(paragraphs, start=1):
                    text = str(paragraph.get("text", "") or "").strip()
                    if not text:
                        continue

                    if full_parts:
                        separator = "\n\n"
                        full_parts.append(separator)
                        cursor += len(separator)

                    char_start = cursor
                    full_parts.append(text)
                    cursor += len(text)
                    char_end = cursor

                    segments.append({
                        "id": f"p{page_number}-seg{idx}",
                        "page": page_number,
                        "paragraph_index": int(paragraph.get("paragraph_index", idx)),
                        "char_start": char_start,
                        "char_end": char_end,
                        "bbox": paragraph.get("bbox"),
                        "text": text,
                    })

        full_text = "".join(full_parts)
        source_map = {
            "schema_version": 1,
            "source_type": "pdf",
            "source_file": os.path.abspath(pdf_path),
            "text": full_text,
            "segments": segments,
        }
        return True, "ok", source_map
    except Exception as e:
        return False, f"PDF 解析失败：{str(e)}", None


def resolve_locator_from_source_map(
    source_map: dict[str, Any] | None,
    char_start: int | None = None,
    char_end: int | None = None,
    snippet: str = "",
) -> dict[str, Any]:
    """Resolve page / paragraph / bbox info from a source map."""
    if not source_map:
        return {
            "page": None,
            "paragraph_index": None,
            "bbox": None,
        }

    segments = source_map.get("segments", []) or []
    if not segments:
        return {"page": None, "paragraph_index": None, "bbox": None}

    matches: list[dict[str, Any]] = []
    if char_start is not None:
        for segment in segments:
            seg_start = segment.get("char_start")
            seg_end = segment.get("char_end")
            if seg_start is None or seg_end is None:
                continue
            overlaps = not (char_end is not None and char_end <= seg_start) and not (char_start >= seg_end)
            if overlaps:
                matches.append(segment)

    if not matches and snippet:
        lowered = snippet.lower()
        for segment in segments:
            if lowered and lowered in str(segment.get("text", "")).lower():
                matches.append(segment)

    if not matches:
        return {"page": None, "paragraph_index": None, "bbox": None}

    pages = sorted({item.get("page") for item in matches if item.get("page") is not None})
    paragraphs = sorted({item.get("paragraph_index") for item in matches if item.get("paragraph_index") is not None})
    bboxes = [item.get("bbox") for item in matches if item.get("bbox")]

    page_value: Any = None
    if len(pages) == 1:
        page_value = pages[0]
    elif pages:
        page_value = pages

    paragraph_value: Any = None
    if len(paragraphs) == 1:
        paragraph_value = paragraphs[0]
    elif paragraphs:
        paragraph_value = paragraphs

    bbox_value: Any = None
    if len(bboxes) == 1:
        bbox_value = bboxes[0]
    elif bboxes:
        bbox_value = bboxes

    return {
        "page": page_value,
        "paragraph_index": paragraph_value,
        "bbox": bbox_value,
    }
