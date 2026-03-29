"""
Explainability helpers for VeritasGraph.

This module builds a reusable explanation bundle for a query:
1. key reasoning path
2. supporting subgraph
3. traceable evidence snippets
4. graceful fallbacks for page / paragraph / coordinate metadata
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx
import pandas as pd

from graph_visualizer import create_graph_html_for_query


join = os.path.join


@dataclass
class ExplanationBundle:
    query: str
    answer: str
    key_path_nodes: list[str]
    key_path_edges: list[dict[str, Any]]
    subgraph_nodes: list[str]
    subgraph_stats: dict[str, int]
    evidence_items: list[dict[str, Any]]
    anchor_entities: list[str]
    citations: dict[str, list[str]]
    graph_html: str
    markdown: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "key_path_nodes": self.key_path_nodes,
            "key_path_edges": self.key_path_edges,
            "subgraph_nodes": self.subgraph_nodes,
            "subgraph_stats": self.subgraph_stats,
            "evidence_items": self.evidence_items,
            "anchor_entities": self.anchor_entities,
            "citations": self.citations,
            "graph_html": self.graph_html,
            "markdown": self.markdown,
        }


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _safe_int_list(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value is None or value == "":
            continue
        result.append(str(value).strip())
    return result


def parse_data_citations(answer: str) -> dict[str, list[str]]:
    """Parse GraphRAG style [Data: Sources (...); Entities (...); Relationships (...)] citations."""
    citations = {"sources": [], "entities": [], "relationships": []}
    if not answer:
        return citations

    pattern = re.compile(r"\[Data:\s*([^\]]+)\]")
    for block in pattern.findall(answer):
        for label, values in re.findall(r"(Sources|Entities|Relationships)\s*\(([^)]*)\)", block):
            key = label.lower()
            citations[key].extend(
                [item.strip() for item in values.split(",") if item.strip()]
            )

    for key, values in citations.items():
        citations[key] = list(dict.fromkeys(values))
    return citations


def load_traceability_data(input_dir: str) -> dict[str, pd.DataFrame]:
    return {
        "nodes": pd.read_parquet(join(input_dir, "create_final_nodes.parquet")),
        "relationships": pd.read_parquet(join(input_dir, "create_final_relationships.parquet")),
        "text_units": pd.read_parquet(join(input_dir, "create_final_text_units.parquet")),
        "documents": pd.read_parquet(join(input_dir, "create_final_documents.parquet")),
    }


def _build_document_lookup(documents_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in documents_df.iterrows():
        row_dict = row.to_dict()
        lookup[str(row_dict["id"])] = row_dict
    return lookup


def _build_text_unit_lookup(text_units_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in text_units_df.iterrows():
        row_dict = row.to_dict()
        lookup[str(row_dict["id"])] = row_dict
    return lookup


def _entity_anchor_score(entity: str, query: str, answer: str) -> int:
    score = 0
    entity_text = str(entity or "")
    if not entity_text:
        return score
    if entity_text in query:
        score += 5
    if entity_text in answer:
        score += 3
    return score


def select_anchor_entities(
    query: str,
    answer: str,
    entities_df: pd.DataFrame,
    citations: dict[str, list[str]],
) -> list[str]:
    if entities_df is None or entities_df.empty:
        return []

    working = entities_df.copy()
    working["id"] = working["id"].astype(str)
    cited_entity_ids = set(_safe_int_list(citations.get("entities", [])))

    candidate_rows: list[tuple[int, int, str]] = []
    for _, row in working.iterrows():
        entity_name = str(row.get("entity", ""))
        if not entity_name:
            continue
        score = _entity_anchor_score(entity_name, query, answer)
        if row["id"] in cited_entity_ids:
            score += 4
        relationship_count = int(str(row.get("number of relationships", "0")) or 0)
        if score > 0:
            candidate_rows.append((score, relationship_count, entity_name))

    candidate_rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
    anchors = [item[2] for item in candidate_rows[:4]]

    if len(anchors) >= 2:
        return anchors

    if not working.empty:
        fallback = (
            working.assign(
                relationship_count=lambda df: pd.to_numeric(
                    df["number of relationships"], errors="coerce"
                ).fillna(0)
            )
            .sort_values(["relationship_count", "entity"], ascending=[False, True])
        )
        for entity_name in fallback["entity"].tolist():
            if entity_name not in anchors:
                anchors.append(entity_name)
            if len(anchors) >= 2:
                break

    return anchors[:4]


def _build_context_graph(relationships_df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    if relationships_df is None or relationships_df.empty:
        return graph

    working = relationships_df.copy()
    working["weight_num"] = pd.to_numeric(working["weight"], errors="coerce").fillna(1.0)
    working["rank_num"] = pd.to_numeric(working["rank"], errors="coerce").fillna(1.0)

    for _, row in working.iterrows():
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()
        if not source or not target:
            continue
        graph.add_edge(
            source,
            target,
            id=str(row.get("id", "")),
            description=str(row.get("description", "")),
            weight=float(row.get("weight_num", 1.0)),
            rank=float(row.get("rank_num", 1.0)),
            cost=1.0 / max(float(row.get("rank_num", 1.0)) + float(row.get("weight_num", 1.0)), 1.0),
        )
    return graph


def _score_path(
    path_nodes: list[str],
    graph: nx.Graph,
    anchor_entities: list[str],
) -> float:
    if len(path_nodes) < 2:
        return 0.0

    anchor_bonus = sum(3 for node in path_nodes if node in anchor_entities)
    edge_score = 0.0
    for source, target in zip(path_nodes[:-1], path_nodes[1:]):
        edge = graph.get_edge_data(source, target, default={})
        edge_score += float(edge.get("rank", 0.0)) + float(edge.get("weight", 0.0))
    return anchor_bonus + edge_score - 0.5 * (len(path_nodes) - 1)


def extract_key_path(
    anchor_entities: list[str],
    relationships_df: pd.DataFrame,
) -> tuple[list[str], list[dict[str, Any]], nx.Graph]:
    graph = _build_context_graph(relationships_df)
    if graph.number_of_edges() == 0:
        return [], [], graph

    best_path: list[str] = []
    best_score = float("-inf")

    anchors = [entity for entity in anchor_entities if entity in graph.nodes]
    if len(anchors) >= 2:
        for i, source in enumerate(anchors):
            for target in anchors[i + 1 :]:
                try:
                    path = nx.shortest_path(graph, source, target, weight="cost")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                score = _score_path(path, graph, anchors)
                if score > best_score:
                    best_path = path
                    best_score = score

    if len(best_path) < 2:
        ranked_edges = []
        for source, target, data in graph.edges(data=True):
            ranked_edges.append(
                (
                    float(data.get("rank", 0.0)) + float(data.get("weight", 0.0)),
                    [source, target],
                )
            )
        ranked_edges.sort(key=lambda item: -item[0])
        best_path = ranked_edges[0][1] if ranked_edges else []

    key_path_edges: list[dict[str, Any]] = []
    for source, target in zip(best_path[:-1], best_path[1:]):
        edge = graph.get_edge_data(source, target, default={})
        key_path_edges.append(
            {
                "id": str(edge.get("id", "")),
                "source": source,
                "target": target,
                "description": str(edge.get("description", "")),
                "weight": float(edge.get("weight", 0.0)),
                "rank": float(edge.get("rank", 0.0)),
            }
        )

    return best_path, key_path_edges, graph


def select_supporting_subgraph_nodes(
    graph: nx.Graph,
    key_path_nodes: list[str],
    anchor_entities: list[str],
    max_nodes: int = 12,
) -> list[str]:
    if graph.number_of_nodes() == 0:
        return []

    priority_nodes: list[str] = []
    for node in key_path_nodes + anchor_entities:
        if node in graph and node not in priority_nodes:
            priority_nodes.append(node)

    selected = list(priority_nodes)
    for node in priority_nodes:
        neighbors = sorted(graph.neighbors(node), key=lambda item: graph.degree(item), reverse=True)
        for neighbor in neighbors:
            if neighbor not in selected:
                selected.append(neighbor)
            if len(selected) >= max_nodes:
                return selected[:max_nodes]

    if len(selected) < max_nodes:
        for node in sorted(graph.nodes(), key=lambda item: graph.degree(item), reverse=True):
            if node not in selected:
                selected.append(node)
            if len(selected) >= max_nodes:
                break

    return selected[:max_nodes]


def _extract_dynamic_location_fields(text_unit: dict[str, Any]) -> dict[str, Any]:
    location: dict[str, Any] = {}
    for key in [
        "page",
        "page_number",
        "page_numbers",
        "pages",
        "bbox",
        "bboxes",
        "coordinates",
        "layout_bbox",
        "polygon",
        "section",
        "section_path",
        "source_uri",
        "file_path",
    ]:
        value = text_unit.get(key)
        if value is not None and value != "":
            location[key] = value
    return location


def _find_relevant_paragraphs(text: str, terms: list[str]) -> list[dict[str, Any]]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    raw_parts = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    paragraphs: list[dict[str, Any]] = []
    body_parts = raw_parts[1:] if len(raw_parts) > 1 else raw_parts

    for index, paragraph in enumerate(body_parts, start=1):
        matched_terms = [term for term in terms if term and term in paragraph]
        if matched_terms:
            paragraphs.append(
                {
                    "paragraph_index": index,
                    "matched_terms": matched_terms,
                    "snippet": paragraph,
                }
            )

    if paragraphs:
        return paragraphs

    fallback_snippet = body_parts[0] if body_parts else cleaned
    return [{"paragraph_index": 1, "matched_terms": [], "snippet": fallback_snippet}]


def build_evidence_items(
    query: str,
    answer: str,
    key_path_edges: list[dict[str, Any]],
    data_tables: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    relationships_df = data_tables["relationships"].copy()
    relationships_df["human_readable_id"] = relationships_df["human_readable_id"].astype(str)
    text_unit_lookup = _build_text_unit_lookup(data_tables["text_units"])
    document_lookup = _build_document_lookup(data_tables["documents"])

    evidence_items: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for edge in key_path_edges:
        relationship_id = str(edge.get("id", ""))
        matching = relationships_df[relationships_df["human_readable_id"] == relationship_id]
        if matching.empty:
            continue
        relationship_row = matching.iloc[0].to_dict()
        text_unit_ids = relationship_row.get("text_unit_ids") or []
        for text_unit_id in text_unit_ids:
            text_unit = text_unit_lookup.get(str(text_unit_id))
            if not text_unit:
                continue
            document_ids = text_unit.get("document_ids") or []
            document = document_lookup.get(str(document_ids[0]), {}) if len(document_ids) else {}
            terms = [
                edge.get("source", ""),
                edge.get("target", ""),
            ]
            paragraphs = _find_relevant_paragraphs(
                text_unit.get("text", ""),
                [term for term in terms if term] + [token for token in [query, answer] if token],
            )

            item_key = (relationship_id, str(text_unit_id))
            if item_key in seen_keys:
                continue
            seen_keys.add(item_key)

            location = {
                "document_title": document.get("title", "未知文档"),
                "text_unit_id": str(text_unit_id),
                "document_id": str(document.get("id", "")),
                "paragraph_indices": [item["paragraph_index"] for item in paragraphs],
            }
            location.update(_extract_dynamic_location_fields(text_unit))

            evidence_items.append(
                {
                    "relationship_id": relationship_id,
                    "relationship": f'{edge.get("source", "")} → {edge.get("target", "")}',
                    "relationship_description": edge.get("description", ""),
                    "location": location,
                    "paragraph_hits": paragraphs[:2],
                }
            )

    return evidence_items


def render_explanation_markdown(bundle: ExplanationBundle) -> str:
    lines: list[str] = []
    lines.append("## 🧠 关键推理路径")
    if bundle.key_path_edges:
        for index, edge in enumerate(bundle.key_path_edges, start=1):
            lines.append(
                f"{index}. **{edge['source']} → {edge['target']}**：{edge['description'] or '未提供关系描述'}"
            )
    else:
        lines.append("- 未提取到稳定路径，当前仅展示关联子图。")

    lines.append("")
    lines.append("## 🔗 支撑子图")
    lines.append(
        f"- 节点数：**{bundle.subgraph_stats.get('nodes', 0)}** ｜ 边数：**{bundle.subgraph_stats.get('edges', 0)}**"
    )
    if bundle.anchor_entities:
        lines.append(f"- 锚点实体：**{'、'.join(bundle.anchor_entities)}**")

    lines.append("")
    lines.append("## 📚 溯源证据")
    if bundle.evidence_items:
        for item in bundle.evidence_items:
            location = item["location"]
            paragraph_label = "、".join(
                [f"第{idx}段" for idx in location.get("paragraph_indices", [])]
            ) or "段落未知"
            lines.append(
                f"- **{location.get('document_title', '未知文档')}** ｜ {paragraph_label}"
            )
            for hit in item.get("paragraph_hits", [])[:1]:
                lines.append(f"  - 证据片段：{hit.get('snippet', '')}")
            lines.append(f"  - 支撑关系：{item.get('relationship', '')}")
            dynamic_fields = {
                key: value
                for key, value in location.items()
                if key not in {"document_title", "text_unit_id", "document_id", "paragraph_indices"}
            }
            if dynamic_fields:
                lines.append(f"  - 扩展定位：`{dynamic_fields}`")
    else:
        lines.append("- 暂未找到可显示的证据片段。")

    return "\n".join(lines)


def build_explanation_bundle(
    query: str,
    answer: str,
    input_dir: str,
    context_records: dict[str, pd.DataFrame] | None = None,
    max_graph_nodes: int = 16,
) -> ExplanationBundle:
    context_records = context_records or {}
    entities_df = context_records.get("entities", pd.DataFrame())
    relationships_df = context_records.get("relationships", pd.DataFrame())
    citations = parse_data_citations(answer)
    anchor_entities = select_anchor_entities(query, answer, entities_df, citations)
    key_path_nodes, key_path_edges, graph = extract_key_path(anchor_entities, relationships_df)
    subgraph_nodes = select_supporting_subgraph_nodes(graph, key_path_nodes, anchor_entities, max_graph_nodes)

    data_tables = load_traceability_data(input_dir)
    evidence_items = build_evidence_items(query, answer, key_path_edges, data_tables)
    graph_html = create_graph_html_for_query(
        input_dir=input_dir,
        query_entities=anchor_entities,
        max_nodes=max_graph_nodes,
        focus_nodes=subgraph_nodes,
        highlight_path_nodes=key_path_nodes,
        highlight_path_edges=[(edge["source"], edge["target"]) for edge in key_path_edges],
    )

    bundle = ExplanationBundle(
        query=query,
        answer=answer,
        key_path_nodes=key_path_nodes,
        key_path_edges=key_path_edges,
        subgraph_nodes=subgraph_nodes,
        subgraph_stats={
            "nodes": len(subgraph_nodes),
            "edges": len(key_path_edges),
        },
        evidence_items=evidence_items,
        anchor_entities=anchor_entities,
        citations=citations,
        graph_html=graph_html,
        markdown="",
    )
    bundle.markdown = render_explanation_markdown(bundle)
    return bundle
