"""
Query trace builder for answer + reasoning path + supporting subgraph + evidence.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd

from graph_visualizer import extract_entities_from_response
from source_locator import build_evidence_item, load_source_tables

join = os.path.join


@dataclass
class ReasoningStep:
    source: str
    target: str
    relationship_id: str
    description: str
    weight: float
    rank: Optional[float]
    text_unit_ids: List[str]


def _safe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if hasattr(value, "tolist"):
        return [str(item) for item in value.tolist() if item not in (None, "")]
    if isinstance(value, tuple):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _load_trace_tables(input_dir: str) -> Dict[str, pd.DataFrame]:
    tables = {
        "nodes": pd.read_parquet(join(input_dir, "create_final_nodes.parquet")),
        "relationships": pd.read_parquet(join(input_dir, "create_final_relationships.parquet")),
        "entities": pd.read_parquet(join(input_dir, "create_final_entities.parquet")),
        "text_units": pd.read_parquet(join(input_dir, "create_final_text_units.parquet")),
        "documents": pd.read_parquet(join(input_dir, "create_final_documents.parquet")),
        "community_reports": pd.read_parquet(join(input_dir, "create_final_community_reports.parquet")),
        "communities": pd.read_parquet(join(input_dir, "create_final_communities.parquet")),
    }

    merged_entities = tables["nodes"].merge(
        tables["entities"][["id", "name", "text_unit_ids", "description"]].rename(
            columns={"id": "entity_id", "description": "entity_detail_description"}
        ),
        left_on="title",
        right_on="name",
        how="left",
    )
    tables["entity_lookup"] = merged_entities
    return tables


def _map_context_entities(context_data: Any, entity_lookup: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(context_data, dict) or "entities" not in context_data:
        return pd.DataFrame()
    ctx = context_data["entities"]
    if ctx is None or len(ctx) == 0:
        return pd.DataFrame()
    ids = {str(v) for v in ctx.get("id", []).tolist()} if "id" in ctx.columns else set()
    titles = {str(v) for v in ctx.get("entity", []).tolist()} if "entity" in ctx.columns else set()
    matched = entity_lookup.copy()
    if ids:
        matched = matched[matched["human_readable_id"].astype(str).isin(ids) | matched["entity_id"].astype(str).isin(ids)]
    if titles:
        matched = pd.concat(
            [matched, entity_lookup[entity_lookup["title"].isin(titles)]],
            ignore_index=True,
        ).drop_duplicates(subset=["title"])
    return matched.drop_duplicates(subset=["title"])


def _map_context_relationships(context_data: Any, relationship_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(context_data, dict) or "relationships" not in context_data:
        return pd.DataFrame()
    ctx = context_data["relationships"]
    if ctx is None or len(ctx) == 0:
        return pd.DataFrame()
    matched = relationship_df.copy()
    if "id" in ctx.columns:
        ids = {str(v) for v in ctx["id"].tolist()}
        matched = matched[matched["human_readable_id"].astype(str).isin(ids) | matched["id"].astype(str).isin(ids)]
    if {"source", "target"}.issubset(set(ctx.columns)):
        edge_pairs = {(str(row["source"]), str(row["target"])) for _, row in ctx.iterrows()}
        extra = relationship_df[
            relationship_df.apply(
                lambda row: (str(row["source"]), str(row["target"])) in edge_pairs
                or (str(row["target"]), str(row["source"])) in edge_pairs,
                axis=1,
            )
        ]
        matched = pd.concat([matched, extra], ignore_index=True)
    return matched.drop_duplicates(subset=["id"])


def _map_context_communities(context_data: Any, tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(context_data, dict) or "reports" not in context_data:
        return pd.DataFrame(), pd.DataFrame()
    ctx = context_data["reports"]
    if ctx is None or len(ctx) == 0:
        return pd.DataFrame(), pd.DataFrame()
    ids = {str(v) for v in ctx.get("id", []).tolist()} if "id" in ctx.columns else set()
    reports = tables["community_reports"][tables["community_reports"]["community"].astype(str).isin(ids)]
    communities = tables["communities"][tables["communities"]["id"].astype(str).isin(ids)]
    return reports, communities


def _resolve_source_context_ids(context_data: Any, text_unit_df: pd.DataFrame) -> List[str]:
    if not isinstance(context_data, dict) or "sources" not in context_data:
        return []
    ctx = context_data["sources"]
    if ctx is None or len(ctx) == 0 or "text" not in ctx.columns:
        return []
    text_map = {str(row["text"]): str(row["id"]) for _, row in text_unit_df.iterrows()}
    resolved = []
    for text in ctx["text"].tolist():
        text = str(text)
        if text in text_map:
            resolved.append(text_map[text])
    return resolved


def _build_graph(relationship_df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for _, row in relationship_df.iterrows():
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        if not source or not target:
            continue
        rank = row.get("rank")
        weight = float(row.get("weight", 1) or 1)
        graph.add_edge(
            source,
            target,
            relationship_id=str(row.get("id", "")),
            description=str(row.get("description", "") or ""),
            weight=weight,
            rank=float(rank) if pd.notna(rank) else None,
            cost=1 / max(weight, 1e-6),
            text_unit_ids=_safe_list(row.get("text_unit_ids")),
        )
    return graph


def _select_seed_entities(
    query: str,
    response: str,
    entity_lookup: pd.DataFrame,
    context_entities: pd.DataFrame,
) -> Dict[str, int]:
    seed_scores: Dict[str, int] = defaultdict(int)
    title_df = entity_lookup.rename(columns={"title": "title"})
    query_entities = extract_entities_from_response(query, title_df)
    answer_entities = extract_entities_from_response(response, title_df)

    for entity in query_entities:
        seed_scores[str(entity)] += 2
    for entity in answer_entities:
        seed_scores[str(entity)] += 3
    for entity in context_entities.get("title", []).tolist() if len(context_entities) else []:
        seed_scores[str(entity)] += 2

    if not seed_scores:
        top_entities = entity_lookup.sort_values("degree", ascending=False)["title"].head(3).tolist()
        for entity in top_entities:
            seed_scores[str(entity)] += 1
    return dict(seed_scores)


def _edge_step(graph: nx.Graph, source: str, target: str) -> ReasoningStep:
    edge = graph[source][target]
    return ReasoningStep(
        source=source,
        target=target,
        relationship_id=str(edge.get("relationship_id", "")),
        description=str(edge.get("description", "") or ""),
        weight=float(edge.get("weight", 1) or 1),
        rank=edge.get("rank"),
        text_unit_ids=_safe_list(edge.get("text_unit_ids")),
    )


def _find_best_path(graph: nx.Graph, seed_scores: Dict[str, int]) -> List[ReasoningStep]:
    seeds = [node for node in seed_scores if node in graph.nodes]
    best_steps: List[ReasoningStep] = []
    best_score = -1.0

    for index, source in enumerate(seeds):
        for target in seeds[index + 1:]:
            try:
                path = nx.shortest_path(graph, source, target, weight="cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            steps = [_edge_step(graph, path[i], path[i + 1]) for i in range(len(path) - 1)]
            edge_score = sum(step.weight for step in steps)
            score = seed_scores[source] + seed_scores[target] + edge_score
            if score > best_score:
                best_score = score
                best_steps = steps

    if best_steps:
        return best_steps

    if seeds:
        source = seeds[0]
        neighbors = list(graph.neighbors(source))
        if neighbors:
            best_neighbor = max(neighbors, key=lambda node: graph[source][node].get("weight", 1))
            return [_edge_step(graph, source, best_neighbor)]

    if graph.number_of_edges() > 0:
        source, target, _ = max(graph.edges(data=True), key=lambda item: item[2].get("weight", 1))
        return [_edge_step(graph, source, target)]

    return []


def _select_subgraph_nodes(
    graph: nx.Graph,
    path_steps: List[ReasoningStep],
    seed_scores: Dict[str, int],
    max_nodes: int = 24,
) -> List[str]:
    selected = set()
    for step in path_steps:
        selected.add(step.source)
        selected.add(step.target)

    for node in sorted(seed_scores, key=seed_scores.get, reverse=True):
        if node in graph:
            selected.add(node)
        if len(selected) >= max_nodes:
            break

    frontier = list(selected)
    for node in frontier:
        neighbors = sorted(
            graph.neighbors(node),
            key=lambda item: graph[node][item].get("weight", 1),
            reverse=True,
        )
        for neighbor in neighbors:
            selected.add(neighbor)
            if len(selected) >= max_nodes:
                return list(selected)
    return list(selected)


def _collect_evidence_ids(
    path_steps: List[ReasoningStep],
    path_nodes: Iterable[str],
    entity_lookup: pd.DataFrame,
    matched_communities: pd.DataFrame,
    source_context_ids: List[str],
) -> List[str]:
    evidence_ids: List[str] = []
    for step in path_steps:
        evidence_ids.extend(step.text_unit_ids)

    if path_nodes:
        node_rows = entity_lookup[entity_lookup["title"].isin(list(path_nodes))]
        for _, row in node_rows.iterrows():
            evidence_ids.extend(_safe_list(row.get("text_unit_ids")))

    if len(matched_communities):
        for _, row in matched_communities.iterrows():
            evidence_ids.extend(_safe_list(row.get("text_unit_ids")))

    evidence_ids.extend(source_context_ids)

    seen = set()
    ordered = []
    for evidence_id in evidence_ids:
        if evidence_id and evidence_id not in seen:
            seen.add(evidence_id)
            ordered.append(evidence_id)
    return ordered


def _build_evidence(
    evidence_ids: List[str],
    text_unit_df: pd.DataFrame,
    document_df: pd.DataFrame,
    max_items: int = 6,
) -> List[Dict[str, Any]]:
    if not evidence_ids:
        return []
    documents_by_id = {str(row["id"]): row for _, row in document_df.iterrows()}
    text_unit_lookup = {str(row["id"]): row for _, row in text_unit_df.iterrows()}

    evidence_items: List[Dict[str, Any]] = []
    for evidence_id in evidence_ids:
        row = text_unit_lookup.get(str(evidence_id))
        if row is None:
            continue
        document_ids = _safe_list(row.get("document_ids"))
        document_row = documents_by_id.get(document_ids[0]) if document_ids else None
        evidence_items.append(build_evidence_item(row, document_row))
        if len(evidence_items) >= max_items:
            break
    return evidence_items


def render_trace_markdown(trace: Dict[str, Any]) -> str:
    """Render the trace package as Markdown for the Gradio demo."""
    lines = [
        "## 🧭 推理路径",
        "",
        f"- **子图节点数：** {trace['subgraph'].get('node_count', 0)}",
        f"- **子图边数：** {trace['subgraph'].get('edge_count', 0)}",
        f"- **核心实体：** {'、'.join(trace.get('focus_entities', [])[:8]) or '未识别'}",
        "",
    ]

    steps = trace.get("reasoning_path", [])
    if steps:
        for idx, step in enumerate(steps, 1):
            desc = step["description"] or "存在直接关联"
            lines.append(f"{idx}. **{step['source']} → {step['target']}**：{desc}")
    else:
        lines.append("- 未能稳定提取显式多跳路径，已退化为相关子图与证据展示。")

    lines.extend(["", "## 📚 溯源证据", ""])
    evidence = trace.get("evidence", [])
    if evidence:
        for idx, item in enumerate(evidence, 1):
            locator = item.get("locator", {})
            title = item.get("document_title") or "未知文档"
            paragraph_index = locator.get("paragraph_index")
            lines.append(
                f"{idx}. **{title}**｜段落 {paragraph_index or 'N/A'}"
            )
            lines.append(f"   - 片段：{item.get('snippet', '')}")
    else:
        lines.append("- 未找到稳定的文本级证据。")

    return "\n".join(lines)


def build_query_trace(
    query: str,
    response: str,
    input_dir: str,
    query_type: str = "local",
    search_result: Any = None,
    max_nodes: int = 24,
) -> Dict[str, Any]:
    """Build a structured trace package for a query response."""
    tables = _load_trace_tables(input_dir)
    entity_lookup = tables["entity_lookup"]
    relationship_df = tables["relationships"].copy()

    context_data = getattr(search_result, "context_data", None) if search_result is not None else None
    context_entities = _map_context_entities(context_data, entity_lookup)
    context_relationships = _map_context_relationships(context_data, relationship_df)
    _, matched_communities = _map_context_communities(context_data, tables)
    source_context_ids = _resolve_source_context_ids(context_data, tables["text_units"])

    seed_scores = _select_seed_entities(query, response, entity_lookup, context_entities)

    candidate_relationships = context_relationships
    if len(matched_communities):
        community_relationship_ids = []
        for _, row in matched_communities.iterrows():
            community_relationship_ids.extend(_safe_list(row.get("relationship_ids")))
        community_relationships = relationship_df[relationship_df["id"].astype(str).isin(community_relationship_ids)]
        candidate_relationships = pd.concat([candidate_relationships, community_relationships], ignore_index=True)

    if len(candidate_relationships) == 0 and seed_scores:
        seed_entities = set(seed_scores.keys())
        candidate_relationships = relationship_df[
            relationship_df["source"].isin(seed_entities) | relationship_df["target"].isin(seed_entities)
        ]

    if len(candidate_relationships) == 0:
        candidate_relationships = relationship_df.nlargest(min(max_nodes, len(relationship_df)), "weight")

    candidate_relationships = candidate_relationships.drop_duplicates(subset=["id"])
    graph = _build_graph(candidate_relationships)
    path_steps = _find_best_path(graph, seed_scores)
    path_nodes = {step.source for step in path_steps} | {step.target for step in path_steps}
    selected_nodes = _select_subgraph_nodes(graph, path_steps, seed_scores, max_nodes=max_nodes)

    subgraph_df = candidate_relationships[
        candidate_relationships["source"].isin(selected_nodes) & candidate_relationships["target"].isin(selected_nodes)
    ]

    evidence_ids = _collect_evidence_ids(
        path_steps=path_steps,
        path_nodes=path_nodes or set(selected_nodes),
        entity_lookup=entity_lookup,
        matched_communities=matched_communities,
        source_context_ids=source_context_ids,
    )

    evidence = _build_evidence(
        evidence_ids=evidence_ids,
        text_unit_df=tables["text_units"],
        document_df=tables["documents"],
    )

    trace = {
        "query": query,
        "query_type": query_type,
        "focus_entities": list(sorted(seed_scores, key=seed_scores.get, reverse=True))[:8],
        "reasoning_path": [asdict(step) for step in path_steps],
        "subgraph": {
            "node_titles": selected_nodes,
            "edge_count": int(len(subgraph_df)),
            "node_count": int(len(set(selected_nodes))),
            "path_edges": [(step.source, step.target) for step in path_steps],
        },
        "evidence": evidence,
    }
    trace["markdown"] = render_trace_markdown(trace)
    return trace
