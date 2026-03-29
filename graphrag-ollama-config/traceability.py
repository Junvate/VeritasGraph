from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import networkx as nx
import pandas as pd

join = os.path.join

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_]+")
CITATION_PATTERN = re.compile(
    r"\[Data:\s*Entities\s*\((?P<entities>[^)]*)\)\s*;\s*Relationships\s*\((?P<relationships>[^)]*)\)\]"
)


def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return [value]
    return [value]


def _normalize_short_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def _paragraphs(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "" and not (isinstance(value, float) and pd.isna(value)):
            return value
    return None


def _parse_json_like(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value or not isinstance(value, str):
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _html(text: Any) -> str:
    return html.escape("" if text is None else str(text))


@dataclass
class EvidenceLocator:
    document_title: str | None = None
    document_id: str | None = None
    text_unit_id: str | None = None
    page: str | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    bbox: str | None = None


@dataclass
class EvidenceSnippet:
    locator: EvidenceLocator
    excerpt: str
    reason: str
    score: float = 0.0
    supporting_relationships: list[str] = field(default_factory=list)
    supporting_entities: list[str] = field(default_factory=list)


@dataclass
class ReasoningStep:
    title: str
    explanation: str
    relation_ids: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


@dataclass
class QueryTrace:
    query: str
    answer: str
    mode: str
    reasoning_steps: list[ReasoningStep] = field(default_factory=list)
    evidence: list[EvidenceSnippet] = field(default_factory=list)
    focus_entities: list[str] = field(default_factory=list)
    subgraph_nodes: list[str] = field(default_factory=list)
    subgraph_edges: list[tuple[str, str]] = field(default_factory=list)
    highlighted_path_edges: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class ArtifactRepository:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.nodes = self._read("create_final_nodes.parquet")
        self.relationships = self._read("create_final_relationships.parquet")
        self.text_units = self._read("create_final_text_units.parquet")
        self.documents = self._read("create_final_documents.parquet")
        self.reports = self._read("create_final_community_reports.parquet")
        self.entities = self._read("create_final_entities.parquet")

        self.entity_by_short = {
            _normalize_short_id(row.get("human_readable_id")): row
            for row in self.nodes.to_dict(orient="records")
        }
        self.relationship_by_short = {
            _normalize_short_id(row.get("human_readable_id")): row
            for row in self.relationships.to_dict(orient="records")
        }
        self.text_unit_by_id = {str(row.get("id")): row for row in self.text_units.to_dict(orient="records")}
        self.document_by_id = {str(row.get("id")): row for row in self.documents.to_dict(orient="records")}
        self.report_by_id = {
            _normalize_short_id(row.get("community")): row
            for row in self.reports.to_dict(orient="records")
        }
        self.entity_details_by_id = {
            str(row.get("id")): row for row in self.entities.to_dict(orient="records")
        }

    def _read(self, filename: str) -> pd.DataFrame:
        path = join(self.input_dir, filename)
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)

    def get_entity_by_short(self, short_id: Any) -> dict[str, Any] | None:
        return self.entity_by_short.get(_normalize_short_id(short_id))

    def get_relationship_by_short(self, short_id: Any) -> dict[str, Any] | None:
        return self.relationship_by_short.get(_normalize_short_id(short_id))

    def get_text_unit(self, text_unit_id: Any) -> dict[str, Any] | None:
        if text_unit_id is None:
            return None
        return self.text_unit_by_id.get(str(text_unit_id))

    def get_report(self, report_id: Any) -> dict[str, Any] | None:
        return self.report_by_id.get(_normalize_short_id(report_id))

    def get_entity_detail(self, entity_id: Any) -> dict[str, Any] | None:
        if entity_id is None:
            return None
        return self.entity_details_by_id.get(str(entity_id))

    def get_document(self, document_id: Any) -> dict[str, Any] | None:
        if document_id is None:
            return None
        return self.document_by_id.get(str(document_id))

    @lru_cache(maxsize=256)
    def build_graph(self) -> nx.Graph:
        graph = nx.Graph()
        for row in self.nodes.to_dict(orient="records"):
            title = row.get("title")
            if not title:
                continue
            graph.add_node(title, **row)
        for row in self.relationships.to_dict(orient="records"):
            source = row.get("source")
            target = row.get("target")
            if not source or not target:
                continue
            if source not in graph or target not in graph:
                continue
            graph.add_edge(
                source,
                target,
                **row,
                short_id=_normalize_short_id(row.get("human_readable_id")),
            )
        return graph


def parse_report_citations(text: str) -> dict[str, list[str]]:
    entities: list[str] = []
    relationships: list[str] = []
    for match in CITATION_PATTERN.finditer(text or ""):
        entity_part = match.group("entities") or ""
        relationship_part = match.group("relationships") or ""
        entities.extend([
            _normalize_short_id(piece)
            for piece in entity_part.split(",")
            if _normalize_short_id(piece)
        ])
        relationships.extend([
            _normalize_short_id(piece)
            for piece in relationship_part.split(",")
            if _normalize_short_id(piece)
        ])
    return {"entities": entities, "relationships": relationships}


def score_relationship(query: str, answer: str, rel_row: dict[str, Any]) -> float:
    text = " ".join([
        str(rel_row.get("source", "")),
        str(rel_row.get("target", "")),
        str(rel_row.get("description", "")),
    ]).lower()
    score = 0.0
    for token in _tokenize(query):
        if len(token) > 1 and token in text:
            score += 2.0
    for token in _tokenize(answer):
        if len(token) > 1 and token in text:
            score += 1.0
    try:
        score += float(rel_row.get("rank") or 0) * 0.15
    except Exception:
        pass
    try:
        score += float(rel_row.get("weight") or 0) * 0.1
    except Exception:
        pass
    return score


def infer_focus_entities(query: str, answer: str, candidate_entities: list[str]) -> list[str]:
    chosen: list[tuple[float, str]] = []
    query_lower = (query or "").lower()
    answer_lower = (answer or "").lower()
    for entity in candidate_entities:
        if not entity:
            continue
        entity_lower = entity.lower()
        score = 0.0
        if entity_lower in query_lower:
            score += 3.0
        if entity_lower in answer_lower:
            score += 2.0
        if score > 0:
            chosen.append((score, entity))
    chosen.sort(key=lambda item: (-item[0], item[1]))
    return [entity for _, entity in chosen[:6]]


def _best_excerpt(
    query: str,
    answer: str,
    text_unit_text: str,
    document_text: str | None = None,
) -> tuple[str, int | None, int | None]:
    source_text = document_text or text_unit_text
    paragraphs = _paragraphs(source_text)
    if not paragraphs:
        snippet = (text_unit_text or source_text or "").strip()
        return snippet[:280], None, None

    query_tokens = set(_tokenize(query) + _tokenize(answer))
    ranked: list[tuple[float, int, str]] = []
    for idx, paragraph in enumerate(paragraphs, start=1):
        score = 0.0
        paragraph_lower = paragraph.lower()
        for token in query_tokens:
            if len(token) > 1 and token in paragraph_lower:
                score += 1.0
        ranked.append((score, idx, paragraph))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    best = ranked[0]
    return best[2][:320], best[1], best[1]


def _locator_from_rows(
    text_unit_row: dict[str, Any] | None,
    document_row: dict[str, Any] | None,
    paragraph_start: int | None,
    paragraph_end: int | None,
) -> EvidenceLocator:
    metadata = {}
    for row in (text_unit_row, document_row):
        if row:
            metadata.update(_parse_json_like(row.get("metadata")))

    page = _coalesce(
        metadata.get("page"),
        metadata.get("page_number"),
        metadata.get("page_num"),
        text_unit_row.get("page") if text_unit_row else None,
        text_unit_row.get("page_number") if text_unit_row else None,
        document_row.get("page") if document_row else None,
        document_row.get("page_number") if document_row else None,
    )
    bbox = _coalesce(
        metadata.get("bbox"),
        metadata.get("bounding_box"),
        metadata.get("coordinates"),
        text_unit_row.get("bbox") if text_unit_row else None,
        text_unit_row.get("coordinates") if text_unit_row else None,
        document_row.get("bbox") if document_row else None,
    )

    return EvidenceLocator(
        document_title=_coalesce(
            document_row.get("title") if document_row else None,
            metadata.get("title"),
        ),
        document_id=str(document_row.get("id")) if document_row and document_row.get("id") else None,
        text_unit_id=str(text_unit_row.get("id")) if text_unit_row and text_unit_row.get("id") else None,
        page=str(page) if page is not None else None,
        paragraph_start=paragraph_start,
        paragraph_end=paragraph_end,
        bbox=str(bbox) if bbox is not None else None,
    )


def resolve_evidence_snippet(
    repo: ArtifactRepository,
    text_unit_id: str,
    query: str,
    answer: str,
    reason: str,
    score: float,
    supporting_relationships: list[str] | None = None,
    supporting_entities: list[str] | None = None,
) -> EvidenceSnippet | None:
    text_unit_row = repo.get_text_unit(text_unit_id)
    if not text_unit_row:
        return None

    document_ids = _safe_list(text_unit_row.get("document_ids"))
    document_row = repo.get_document(document_ids[0]) if document_ids else None
    document_text = None
    if document_row:
        document_text = str(_coalesce(document_row.get("raw_content"), document_row.get("text"), ""))

    excerpt, paragraph_start, paragraph_end = _best_excerpt(
        query=query,
        answer=answer,
        text_unit_text=str(text_unit_row.get("text", "")),
        document_text=document_text,
    )
    locator = _locator_from_rows(text_unit_row, document_row, paragraph_start, paragraph_end)
    return EvidenceSnippet(
        locator=locator,
        excerpt=excerpt,
        reason=reason,
        score=score,
        supporting_relationships=supporting_relationships or [],
        supporting_entities=supporting_entities or [],
    )


def _collect_candidates_from_local_context(
    repo: ArtifactRepository,
    context_data: dict[str, pd.DataFrame],
    query: str,
    answer: str,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[ReasoningStep]]:
    relationship_rows: dict[str, dict[str, Any]] = {}
    entity_names: set[str] = set()
    text_unit_ids: set[str] = set()
    reasoning_steps: list[ReasoningStep] = []

    relationships_df = context_data.get("relationships")
    if isinstance(relationships_df, pd.DataFrame) and not relationships_df.empty:
        for row in relationships_df.to_dict(orient="records"):
            rel = repo.get_relationship_by_short(row.get("id"))
            if not rel:
                rel = {
                    "source": row.get("source"),
                    "target": row.get("target"),
                    "description": row.get("description"),
                    "rank": row.get("rank"),
                    "weight": row.get("weight"),
                    "human_readable_id": row.get("id"),
                    "text_unit_ids": [],
                }
            short_id = _normalize_short_id(row.get("id") or rel.get("human_readable_id"))
            rel["short_id"] = short_id
            rel["score"] = score_relationship(query, answer, rel)
            relationship_rows[short_id] = rel
            entity_names.add(str(rel.get("source", "")))
            entity_names.add(str(rel.get("target", "")))
            text_unit_ids.update([str(item) for item in _safe_list(rel.get("text_unit_ids"))])

    entities_df = context_data.get("entities")
    if isinstance(entities_df, pd.DataFrame) and not entities_df.empty:
        for row in entities_df.to_dict(orient="records"):
            entity = repo.get_entity_by_short(row.get("id"))
            if entity:
                entity_names.add(str(entity.get("title", "")))
                source_ids = [str(item) for item in _safe_list(entity.get("source_id"))]
                text_unit_ids.update(source_ids)
            else:
                entity_names.add(str(row.get("entity", "")))

    sources_df = context_data.get("sources")
    if isinstance(sources_df, pd.DataFrame) and not sources_df.empty and "id" in sources_df.columns:
        for source_id in sources_df["id"].tolist():
            if source_id:
                text_unit_ids.add(str(source_id))

    sorted_relationships = sorted(
        relationship_rows.values(),
        key=lambda item: (-float(item.get("score") or 0), -float(item.get("rank") or 0), item.get("source", "")),
    )
    for rel in sorted_relationships[:3]:
        reasoning_steps.append(
            ReasoningStep(
                title=f"{rel.get('source', '实体')} → {rel.get('target', '实体')}",
                explanation=str(rel.get("description") or "检索到该关系是回答的重要支撑。"),
                relation_ids=[str(rel.get("short_id"))] if rel.get("short_id") else [],
                entities=[str(rel.get("source", "")), str(rel.get("target", ""))],
            )
        )

    return sorted_relationships, sorted(entity_names), sorted(text_unit_ids), reasoning_steps


def _collect_candidates_from_global_context(
    repo: ArtifactRepository,
    context_data: dict[str, pd.DataFrame],
    query: str,
    answer: str,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[ReasoningStep]]:
    relationship_rows: dict[str, dict[str, Any]] = {}
    entity_names: set[str] = set()
    text_unit_ids: set[str] = set()
    reasoning_steps: list[ReasoningStep] = []

    reports_df = context_data.get("reports")
    if not isinstance(reports_df, pd.DataFrame) or reports_df.empty:
        return [], [], [], []

    for report_stub in reports_df.to_dict(orient="records"):
        report = repo.get_report(report_stub.get("id")) or report_stub
        findings = _safe_list(report.get("findings"))
        for finding in findings[:4]:
            if not isinstance(finding, dict):
                continue
            explanation = str(finding.get("explanation") or "")
            citations = parse_report_citations(explanation)
            related_entities: list[str] = []
            for entity_id in citations["entities"]:
                entity = repo.get_entity_by_short(entity_id)
                if entity:
                    entity_name = str(entity.get("title", ""))
                    entity_names.add(entity_name)
                    related_entities.append(entity_name)
                    text_unit_ids.update([str(item) for item in _safe_list(entity.get("source_id"))])
            for rel_id in citations["relationships"]:
                rel = repo.get_relationship_by_short(rel_id)
                if not rel:
                    continue
                rel["short_id"] = rel_id
                rel["score"] = score_relationship(query, answer, rel) + 1.5
                relationship_rows[rel_id] = rel
                entity_names.add(str(rel.get("source", "")))
                entity_names.add(str(rel.get("target", "")))
                text_unit_ids.update([str(item) for item in _safe_list(rel.get("text_unit_ids"))])
            reasoning_steps.append(
                ReasoningStep(
                    title=str(finding.get("summary") or report.get("title") or "社区发现"),
                    explanation=explanation,
                    relation_ids=citations["relationships"][:4],
                    entities=related_entities[:6],
                )
            )

    sorted_relationships = sorted(
        relationship_rows.values(),
        key=lambda item: (-float(item.get("score") or 0), -float(item.get("rank") or 0), item.get("source", "")),
    )
    return sorted_relationships, sorted(entity_names), sorted(text_unit_ids), reasoning_steps[:5]


def _select_path_edges(
    repo: ArtifactRepository,
    focus_entities: list[str],
    ranked_relationships: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    if not ranked_relationships:
        return []

    graph = nx.Graph()
    for rel in ranked_relationships[:20]:
        source = str(rel.get("source", "")).strip()
        target = str(rel.get("target", "")).strip()
        if not source or not target:
            continue
        cost = 1.0 / max(float(rel.get("score") or 0.1), 0.1)
        graph.add_edge(source, target, weight=cost, score=float(rel.get("score") or 0))

    focus = [entity for entity in focus_entities if entity in graph]
    if len(focus) >= 2:
        best_path: list[str] | None = None
        best_cost = float("inf")
        for idx in range(len(focus) - 1):
            for jdx in range(idx + 1, len(focus)):
                try:
                    path = nx.shortest_path(graph, focus[idx], focus[jdx], weight="weight")
                    cost = nx.path_weight(graph, path, weight="weight")
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path
                except Exception:
                    continue
        if best_path and len(best_path) >= 2:
            return [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]

    top = ranked_relationships[0]
    return [(str(top.get("source", "")), str(top.get("target", "")))]


def build_query_trace(
    query: str,
    answer: str,
    query_type: str,
    input_dir: str,
    search_result: Any = None,
) -> QueryTrace:
    repo = ArtifactRepository(input_dir)
    notes = [
        "关键路径为基于检索到的图关系和证据自动归纳的“证据链”，用于解释答案来源，不声称完全等同于模型内部思维链。",
        "若原始索引中缺少页码/版面坐标，本界面会退化显示到文档与段落级定位。",
    ]
    trace = QueryTrace(query=query, answer=answer, mode=query_type, notes=notes)

    ranked_relationships: list[dict[str, Any]] = []
    candidate_entities: list[str] = []
    evidence_text_units: list[str] = []
    reasoning_steps: list[ReasoningStep] = []

    context_data = getattr(search_result, "context_data", None)
    if isinstance(context_data, dict):
        if query_type == "global":
            (
                ranked_relationships,
                candidate_entities,
                evidence_text_units,
                reasoning_steps,
            ) = _collect_candidates_from_global_context(repo, context_data, query, answer)
        else:
            (
                ranked_relationships,
                candidate_entities,
                evidence_text_units,
                reasoning_steps,
            ) = _collect_candidates_from_local_context(repo, context_data, query, answer)

    trace.focus_entities = infer_focus_entities(query, answer, candidate_entities)
    if not trace.focus_entities:
        trace.focus_entities = candidate_entities[:4]

    path_edges = _select_path_edges(repo, trace.focus_entities, ranked_relationships)
    trace.highlighted_path_edges = path_edges

    if not reasoning_steps:
        for edge in path_edges:
            rel = next(
                (
                    row
                    for row in ranked_relationships
                    if {
                        str(row.get("source", "")),
                        str(row.get("target", "")),
                    }
                    == {edge[0], edge[1]}
                ),
                None,
            )
            if rel:
                reasoning_steps.append(
                    ReasoningStep(
                        title=f"{edge[0]} → {edge[1]}",
                        explanation=str(rel.get("description") or "该关系被用于连接答案中的关键实体。"),
                        relation_ids=[str(rel.get("short_id"))] if rel.get("short_id") else [],
                        entities=[edge[0], edge[1]],
                    )
                )
    trace.reasoning_steps = reasoning_steps[:5]

    subgraph_nodes = set(trace.focus_entities)
    subgraph_edges = set(path_edges)
    for rel in ranked_relationships[:8]:
        source = str(rel.get("source", ""))
        target = str(rel.get("target", ""))
        if source and target:
            subgraph_nodes.update([source, target])
            subgraph_edges.add((source, target))
            evidence_text_units.extend([str(item) for item in _safe_list(rel.get("text_unit_ids"))])
    trace.subgraph_nodes = sorted([node for node in subgraph_nodes if node])
    trace.subgraph_edges = sorted(subgraph_edges)

    evidence_items: list[EvidenceSnippet] = []
    seen_text_units: set[str] = set()
    for text_unit_id in evidence_text_units:
        if not text_unit_id or text_unit_id in seen_text_units:
            continue
        seen_text_units.add(text_unit_id)
        linked_relationships = [
            str(rel.get("short_id"))
            for rel in ranked_relationships
            if text_unit_id in {str(item) for item in _safe_list(rel.get("text_unit_ids"))}
            and rel.get("short_id")
        ][:4]
        linked_entities = [
            entity
            for entity in trace.focus_entities
            if entity and entity in " ".join([str(text_unit_id), answer, query])
        ]
        snippet = resolve_evidence_snippet(
            repo=repo,
            text_unit_id=text_unit_id,
            query=query,
            answer=answer,
            reason="该原文片段与关键关系/实体直接关联。",
            score=float(len(linked_relationships)),
            supporting_relationships=linked_relationships,
            supporting_entities=linked_entities,
        )
        if snippet:
            evidence_items.append(snippet)

    evidence_items.sort(key=lambda item: (-item.score, item.locator.document_title or "", item.locator.text_unit_id or ""))
    trace.evidence = evidence_items[:6]
    return trace


def render_trace_html(trace: QueryTrace) -> str:
    steps_html = []
    for idx, step in enumerate(trace.reasoning_steps, start=1):
        refs = []
        if step.entities:
            refs.append("实体：" + "、".join([_html(item) for item in step.entities[:6]]))
        if step.relation_ids:
            refs.append("关系ID：" + "、".join([_html(item) for item in step.relation_ids[:6]]))
        refs_html = "<div style='margin-top:6px;color:#888;font-size:12px;'>" + " ｜ ".join(refs) + "</div>" if refs else ""
        steps_html.append(
            f"""
            <div style='padding:12px 14px;margin-bottom:10px;border:1px solid #2d3748;border-radius:10px;background:#111827;'>
              <div style='font-weight:600;color:#f9fafb;'>步骤 {idx}：{_html(step.title)}</div>
              <div style='margin-top:6px;color:#d1d5db;line-height:1.6;'>{_html(step.explanation)}</div>
              {refs_html}
            </div>
            """
        )

    evidence_html = []
    for item in trace.evidence:
        locator_bits = []
        if item.locator.document_title:
            locator_bits.append(f"文档：{_html(item.locator.document_title)}")
        if item.locator.page:
            locator_bits.append(f"页码：{_html(item.locator.page)}")
        if item.locator.paragraph_start:
            if item.locator.paragraph_end and item.locator.paragraph_end != item.locator.paragraph_start:
                locator_bits.append(f"段落：{item.locator.paragraph_start}-{item.locator.paragraph_end}")
            else:
                locator_bits.append(f"段落：{item.locator.paragraph_start}")
        if item.locator.bbox:
            locator_bits.append(f"坐标：{_html(item.locator.bbox)}")

        support_bits = []
        if item.supporting_relationships:
            support_bits.append("关系ID：" + "、".join([_html(rel) for rel in item.supporting_relationships[:6]]))
        if item.supporting_entities:
            support_bits.append("实体：" + "、".join([_html(entity) for entity in item.supporting_entities[:6]]))

        evidence_html.append(
            f"""
            <div style='padding:12px 14px;margin-bottom:10px;border:1px solid #2d3748;border-radius:10px;background:#0f172a;'>
              <div style='color:#93c5fd;font-weight:600;'>{_html(item.reason)}</div>
              <div style='margin-top:6px;color:#e5e7eb;line-height:1.6;'>“{_html(item.excerpt)}”</div>
              <div style='margin-top:8px;color:#9ca3af;font-size:12px;'>{' ｜ '.join(locator_bits) if locator_bits else '暂缺更细粒度定位元数据'}</div>
              <div style='margin-top:4px;color:#6b7280;font-size:12px;'>{' ｜ '.join(support_bits) if support_bits else ''}</div>
            </div>
            """
        )

    notes_html = "".join([f"<li>{_html(note)}</li>" for note in trace.notes])
    focus_entities = "、".join([_html(entity) for entity in trace.focus_entities[:8]]) or "暂无"
    path_text = " → ".join(
        [_html(trace.highlighted_path_edges[0][0])]
        + [_html(edge[1]) for edge in trace.highlighted_path_edges]
    ) if trace.highlighted_path_edges else "暂无可高亮路径"

    return f"""
    <div style='padding:16px;background:#0b1020;border-radius:12px;color:#f3f4f6;'>
      <div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;'>
        <span style='padding:4px 10px;border-radius:999px;background:#1d4ed8;color:white;font-size:12px;'>模式：{_html(trace.mode)}</span>
        <span style='padding:4px 10px;border-radius:999px;background:#374151;color:white;font-size:12px;'>焦点实体：{focus_entities}</span>
      </div>
      <div style='padding:12px 14px;border:1px solid #334155;border-radius:10px;background:#111827;margin-bottom:14px;'>
        <div style='font-weight:700;margin-bottom:8px;'>关键证据链</div>
        <div style='color:#fbbf24;font-size:15px;'>{path_text}</div>
      </div>
      <div style='margin-bottom:8px;font-weight:700;'>关键推理步骤</div>
      {''.join(steps_html) if steps_html else "<div style='color:#9ca3af;'>暂无结构化推理步骤。</div>"}
      <div style='margin:16px 0 8px;font-weight:700;'>溯源证据</div>
      {''.join(evidence_html) if evidence_html else "<div style='color:#9ca3af;'>暂无可展示的证据片段。</div>"}
      <div style='margin-top:16px;padding:12px 14px;border:1px dashed #334155;border-radius:10px;background:#0f172a;'>
        <div style='font-weight:700;margin-bottom:6px;'>说明</div>
        <ul style='margin:0;padding-left:18px;color:#cbd5e1;line-height:1.7;'>{notes_html}</ul>
      </div>
    </div>
    """
