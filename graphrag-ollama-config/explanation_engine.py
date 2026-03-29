"""
Explanation engine for VeritasGraph.

Builds a reusable explanation payload for:
- answer text
- relevant subgraph
- key reasoning paths
- provenance evidence mapped back to source text units/documents

Current provenance granularity is document/snippet/character-range.
Page and layout coordinates require richer PDF ingestion metadata.
"""

from __future__ import annotations

import math
import os
import re
import html
import hashlib
from dataclasses import dataclass
from typing import Any

import networkx as nx
import pandas as pd

from explanation_presenter import (
    render_explanation_html as render_explanation_html_from_payload,
    render_explanation_text as render_explanation_text_from_payload,
)
from pdf_source_map import load_source_map, resolve_locator_from_source_map

join = os.path.join


@dataclass
class ArtifactBundle:
    entity_df: pd.DataFrame
    relationship_df: pd.DataFrame
    text_unit_df: pd.DataFrame
    document_df: pd.DataFrame


class GraphExplanationEngine:
    """Reusable explanation builder around GraphRAG parquet artifacts."""

    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.bundle = self._load_artifacts(input_dir)

        self.entity_df = self.bundle.entity_df.copy()
        self.relationship_df = self.bundle.relationship_df.copy()
        self.text_unit_df = self.bundle.text_unit_df.copy()
        self.document_df = self.bundle.document_df.copy()

        self.entity_by_title = {
            str(row["title"]): row for _, row in self.entity_df.iterrows()
            if pd.notna(row.get("title"))
        }
        self.entity_by_id = {
            str(row["id"]): row for _, row in self.entity_df.iterrows()
            if pd.notna(row.get("id"))
        }
        self.relationship_by_id = {
            str(row["id"]): row for _, row in self.relationship_df.iterrows()
            if pd.notna(row.get("id"))
        }
        self.text_unit_by_id = {
            str(row["id"]): row for _, row in self.text_unit_df.iterrows()
            if pd.notna(row.get("id"))
        }
        self.document_by_id = {
            str(row["id"]): row for _, row in self.document_df.iterrows()
            if pd.notna(row.get("id"))
        }

        self.graph = self._build_graph()

    @staticmethod
    def _load_artifacts(input_dir: str) -> ArtifactBundle:
        def _read(name: str) -> pd.DataFrame:
            path = join(input_dir, f"{name}.parquet")
            if not os.path.exists(path):
                return pd.DataFrame()
            return pd.read_parquet(path)

        return ArtifactBundle(
            entity_df=_read("create_final_nodes"),
            relationship_df=_read("create_final_relationships"),
            text_unit_df=_read("create_final_text_units"),
            document_df=_read("create_final_documents"),
        )

    def _build_graph(self) -> nx.Graph:
        graph = nx.Graph()

        for _, row in self.entity_df.iterrows():
            title = str(row.get("title", "")).strip()
            if not title:
                continue
            graph.add_node(
                title,
                id=str(row.get("id", "")),
                title=title,
                type=str(row.get("type", "ENTITY")),
                description=str(row.get("description", "") or ""),
                degree=self._safe_int(row.get("degree", 0)),
                community=self._safe_int(row.get("community", 0)),
                source_id=str(row.get("source_id", "") or ""),
            )

        for _, row in self.relationship_df.iterrows():
            source = str(row.get("source", "")).strip()
            target = str(row.get("target", "")).strip()
            if not source or not target:
                continue
            if source not in graph:
                graph.add_node(source, title=source, type="ENTITY", description="", degree=0)
            if target not in graph:
                graph.add_node(target, title=target, type="ENTITY", description="", degree=0)
            weight = self._safe_float(row.get("weight", 1.0), 1.0)
            graph.add_edge(
                source,
                target,
                id=str(row.get("id", "")),
                description=str(row.get("description", "") or ""),
                weight=weight,
                rank=self._safe_float(row.get("rank", 0), 0),
                text_unit_ids=self._normalize_array(row.get("text_unit_ids")),
            )

        return graph

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            if pd.isna(value):
                return default
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _normalize_array(value: Any) -> list[str]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if str(item)]
        if hasattr(value, "tolist"):
            return [str(item) for item in value.tolist() if str(item)]
        return [str(value)]

    @staticmethod
    def _text_contains_entity(text: str, entity: str) -> bool:
        text = text or ""
        entity = entity or ""
        if not text or not entity:
            return False
        return entity.lower() in text.lower()

    def detect_entities(self, text: str, limit: int = 12) -> list[str]:
        if not text:
            return []

        matches = []
        lowered = text.lower()
        for title, row in self.entity_by_title.items():
            candidate = title.strip()
            if len(candidate) < 2:
                continue
            if candidate.lower() in lowered:
                matches.append(
                    (
                        len(candidate),
                        self._safe_int(row.get("degree", 0)),
                        candidate,
                    )
                )

        matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
        seen = set()
        results = []
        for _, _, title in matches:
            if title not in seen:
                seen.add(title)
                results.append(title)
            if len(results) >= limit:
                break
        return results

    def _fallback_entities(self, query: str, response: str, limit: int = 6) -> list[str]:
        query_terms = [t for t in re.split(r"[\s,，。；;：:、\n]+", f"{query} {response}") if len(t.strip()) >= 2]
        scored = []
        for title, row in self.entity_by_title.items():
            score = 0
            for term in query_terms:
                if term and (term in title or title in term):
                    score += len(term)
            if score > 0:
                scored.append((score, self._safe_int(row.get("degree", 0)), title))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [title for _, _, title in scored[:limit]]

    def select_seed_entities(self, query: str, response: str = "") -> dict[str, list[str]]:
        query_entities = self.detect_entities(query, limit=8)
        answer_entities = [item for item in self.detect_entities(response, limit=12) if item not in query_entities]

        if not query_entities:
            query_entities = self._fallback_entities(query, response, limit=4)

        if not answer_entities:
            answer_entities = self._expand_answer_targets(query_entities)

        combined = list(dict.fromkeys(query_entities + answer_entities))
        return {
            "query_entities": query_entities[:6],
            "answer_entities": answer_entities[:6],
            "all_entities": combined[:10],
        }

    def _expand_answer_targets(self, query_entities: list[str], limit: int = 4) -> list[str]:
        candidates = []
        for title in query_entities:
            if title not in self.graph:
                continue
            for neighbor in self.graph.neighbors(title):
                edge = self.graph.get_edge_data(title, neighbor, default={})
                candidates.append((
                    self._safe_float(edge.get("weight"), 0.0),
                    self.graph.degree(neighbor),
                    neighbor,
                ))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        results = []
        for _, _, title in candidates:
            if title not in query_entities and title not in results:
                results.append(title)
            if len(results) >= limit:
                break
        return results

    def _find_paths(self, query_entities: list[str], answer_entities: list[str], max_paths: int = 3) -> list[dict[str, Any]]:
        paths = []
        seen_signatures = set()

        targets = answer_entities or self._expand_answer_targets(query_entities, limit=4)

        for source in query_entities:
            if source not in self.graph:
                continue
            for target in targets:
                if target not in self.graph or source == target:
                    continue
                try:
                    nodes = nx.shortest_path(
                        self.graph,
                        source=source,
                        target=target,
                        weight=lambda _u, _v, data: 1 / max(self._safe_float(data.get("weight"), 1.0), 0.01),
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

                if len(nodes) < 2:
                    continue

                edges = []
                total_weight = 0.0
                text_unit_ids: list[str] = []
                for left, right in zip(nodes[:-1], nodes[1:]):
                    edge = self.graph.get_edge_data(left, right, default={})
                    edge_id = str(edge.get("id", ""))
                    edges.append({
                        "id": edge_id,
                        "source": left,
                        "target": right,
                        "description": str(edge.get("description", "") or ""),
                        "weight": self._safe_float(edge.get("weight"), 0.0),
                    })
                    total_weight += self._safe_float(edge.get("weight"), 0.0)
                    text_unit_ids.extend(self._normalize_array(edge.get("text_unit_ids")))

                signature = tuple(nodes)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                paths.append({
                    "id": f"path-{len(paths) + 1}",
                    "source": source,
                    "target": target,
                    "nodes": nodes,
                    "edges": edges,
                    "score": round(total_weight / max(len(edges), 1), 3),
                    "evidence_text_unit_ids": list(dict.fromkeys(text_unit_ids)),
                    "explanation": " → ".join(nodes),
                })

        paths.sort(key=lambda item: (-item["score"], len(item["nodes"]), item["explanation"]))
        return paths[:max_paths]

    @staticmethod
    def _extract_answer_claims(response: str, max_claims: int = 4) -> list[str]:
        text = str(response or "").strip()
        if not text:
            return []
        parts = [
            part.strip(" \n\t;；,，。")
            for part in re.split(r"[\n。！？!?；;]+", text)
            if part.strip(" \n\t;；,，。")
        ]
        claims: list[str] = []
        for part in parts:
            if len(part) < 6:
                continue
            claims.append(part)
            if len(claims) >= max_claims:
                break
        return claims or ([text[:160]] if text else [])

    @staticmethod
    def _extract_terms(text: str) -> list[str]:
        tokens = []
        for token in re.split(r"[\s,，。；;：:、\n()\[\]{}]+", str(text or "")):
            token = token.strip()
            if len(token) >= 2:
                tokens.append(token.lower())
        return list(dict.fromkeys(tokens))

    def _score_claim_against_path(
        self,
        claim: str,
        claim_entities: list[str],
        path: dict[str, Any],
    ) -> float:
        claim_terms = self._extract_terms(claim)
        path_entities = [str(item) for item in path.get("nodes", []) if str(item)]
        entity_overlap = len(set(claim_entities) & set(path_entities))
        explanation_terms = self._extract_terms(path.get("explanation", ""))
        term_overlap = len(set(claim_terms) & set(explanation_terms))
        return entity_overlap * 4 + term_overlap + float(path.get("score", 0))

    def _score_claim_against_evidence(
        self,
        claim: str,
        claim_entities: list[str],
        evidence: dict[str, Any],
    ) -> float:
        claim_terms = self._extract_terms(claim)
        evidence_terms = self._extract_terms(
            f"{evidence.get('document_title', '')} {evidence.get('snippet', '')}"
        )
        evidence_entities = evidence.get("graph_refs", {}).get("entity_titles", []) or []
        entity_overlap = len(set(claim_entities) & set(evidence_entities))
        term_overlap = len(set(claim_terms) & set(evidence_terms))
        rel_bonus = len(evidence.get("graph_refs", {}).get("relationship_ids", []) or [])
        return entity_overlap * 4 + term_overlap + rel_bonus

    def _build_reasoning_steps(
        self,
        response: str,
        paths: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        claims = self._extract_answer_claims(response)
        if not claims:
            return []

        reasoning_steps: list[dict[str, Any]] = []
        for idx, claim in enumerate(claims, start=1):
            claim_entities = self.detect_entities(claim, limit=6)
            scored_paths = []
            for path in paths:
                score = self._score_claim_against_path(claim, claim_entities, path)
                if score > 0:
                    scored_paths.append((score, path))
            scored_paths.sort(key=lambda item: (-item[0], item[1].get("id", "")))

            scored_evidence = []
            for item in evidence:
                score = self._score_claim_against_evidence(claim, claim_entities, item)
                if score > 0:
                    scored_evidence.append((score, item))
            scored_evidence.sort(key=lambda item: (-item[0], item[1].get("id", "")))

            top_paths = [item[1] for item in scored_paths[:2]]
            top_evidence = [item[1] for item in scored_evidence[:2]]
            referenced_entities = list(
                dict.fromkeys(
                    claim_entities
                    + [node for path in top_paths for node in path.get("nodes", [])]
                    + [
                        entity
                        for item in top_evidence
                        for entity in item.get("graph_refs", {}).get("entity_titles", [])
                    ]
                )
            )[:8]
            confidence = round(
                min(
                    1.0,
                    0.2
                    + 0.15 * len(top_paths)
                    + 0.15 * len(top_evidence)
                    + 0.05 * len(claim_entities),
                ),
                2,
            )
            reasoning_steps.append({
                "id": f"step-{idx}",
                "claim": claim,
                "claim_entities": claim_entities,
                "path_ids": [item.get("id", "") for item in top_paths if item.get("id")],
                "evidence_ids": [item.get("id", "") for item in top_evidence if item.get("id")],
                "supporting_paths": top_paths,
                "supporting_evidence": top_evidence,
                "referenced_entities": referenced_entities,
                "confidence": confidence,
            })

        return reasoning_steps

    @staticmethod
    def _format_location_text(location: dict[str, Any]) -> str:
        page = location.get("page")
        paragraph = location.get("paragraph_index")
        bbox = location.get("bbox")

        location_parts = []
        if page is not None:
            if isinstance(page, list):
                location_parts.append(f"页码 {', '.join(str(item) for item in page)}")
            else:
                location_parts.append(f"页码 {page}")
        if paragraph is not None:
            if isinstance(paragraph, list):
                location_parts.append(f"段落 {', '.join(str(item) for item in paragraph)}")
            else:
                location_parts.append(f"段落 {paragraph}")
        if location.get("char_start") is not None and location.get("char_end") is not None:
            location_parts.append(f"字符 {location.get('char_start')}-{location.get('char_end')}")
        if bbox:
            location_parts.append(f"bbox {bbox}")
        return " / ".join(location_parts) or "未定位"

    def _collect_evidence(self, seed_entities: dict[str, list[str]], paths: list[dict[str, Any]], max_evidence: int = 6) -> list[dict[str, Any]]:
        evidence_by_text_unit: dict[str, dict[str, Any]] = {}

        # Prefer path-backed evidence.
        for path in paths:
            for text_unit_id in path.get("evidence_text_unit_ids", []):
                evidence = self._build_evidence_record(
                    text_unit_id=text_unit_id,
                    supporting_entities=path.get("nodes", []),
                    supporting_relationships=[edge["id"] for edge in path.get("edges", []) if edge.get("id")],
                )
                if evidence:
                    evidence_by_text_unit[text_unit_id] = evidence

        # Backfill from entity source text units.
        for title in seed_entities.get("all_entities", []):
            row = self.entity_by_title.get(title)
            if row is None:
                continue
            text_unit_id = str(row.get("source_id", "") or "")
            if not text_unit_id or text_unit_id in evidence_by_text_unit:
                continue
            evidence = self._build_evidence_record(
                text_unit_id=text_unit_id,
                supporting_entities=[title],
                supporting_relationships=[],
            )
            if evidence:
                evidence_by_text_unit[text_unit_id] = evidence

        evidence_items = list(evidence_by_text_unit.values())
        evidence_items.sort(
            key=lambda item: (
                -len(item.get("graph_refs", {}).get("relationship_ids", [])),
                -len(item.get("graph_refs", {}).get("entity_titles", [])),
                item.get("document_title", ""),
            )
        )
        return evidence_items[:max_evidence]

    @staticmethod
    def _link_paths_and_evidence(paths: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> None:
        evidence_by_text_unit = {
            str(item.get("text_unit_id", "")): item for item in evidence if item.get("text_unit_id")
        }

        for item in evidence:
            item["path_ids"] = []

        for path in paths:
            linked_evidence_ids: list[str] = []
            supporting_documents: list[str] = []

            for text_unit_id in path.get("evidence_text_unit_ids", []):
                evidence_item = evidence_by_text_unit.get(str(text_unit_id))
                if not evidence_item:
                    continue

                evidence_id = str(evidence_item.get("id", "") or "")
                if evidence_id:
                    linked_evidence_ids.append(evidence_id)
                document_title = str(evidence_item.get("document_title", "") or "")
                if document_title:
                    supporting_documents.append(document_title)
                evidence_item.setdefault("path_ids", []).append(str(path.get("id", "") or ""))

            path["evidence_ids"] = list(dict.fromkeys(linked_evidence_ids))
            path["evidence_count"] = len(path["evidence_ids"])
            path["supporting_documents"] = list(dict.fromkeys(supporting_documents))

        for item in evidence:
            item["path_ids"] = list(dict.fromkeys([
                str(path_id) for path_id in item.get("path_ids", []) if str(path_id)
            ]))

    def _build_evidence_record(
        self,
        text_unit_id: str,
        supporting_entities: list[str],
        supporting_relationships: list[str],
    ) -> dict[str, Any] | None:
        text_row = self.text_unit_by_id.get(text_unit_id)
        if text_row is None:
            return None

        raw_text = str(text_row.get("text", "") or "")
        document_ids = self._normalize_array(text_row.get("document_ids"))
        document_id = document_ids[0] if document_ids else ""
        document_row = self.document_by_id.get(document_id)
        document_title = str(document_row.get("title", "")) if document_row is not None else ""
        raw_content = str(document_row.get("raw_content", "")) if document_row is not None else raw_text

        location = self._locate_text(raw_content, raw_text)
        source_map = load_source_map(document_title) if document_title else None
        source_map_locator = resolve_locator_from_source_map(
            source_map=source_map,
            char_start=location.get("char_start"),
            char_end=location.get("char_end"),
            snippet=location.get("snippet", ""),
        )
        graph_entity_refs = list(dict.fromkeys([entity for entity in supporting_entities if entity]))

        return {
            "id": f"evidence-{text_unit_id}",
            "text_unit_id": text_unit_id,
            "document_id": document_id,
            "document_title": document_title or "未知文档",
            "snippet": location["snippet"],
            "raw_text": raw_text,
            "location": {
                "page": source_map_locator.get("page"),
                "paragraph_index": source_map_locator.get("paragraph_index") or location["paragraph_index"],
                "char_start": location["char_start"],
                "char_end": location["char_end"],
                "bbox": source_map_locator.get("bbox"),
            },
            "source_map_available": bool(source_map),
            "graph_refs": {
                "entity_titles": graph_entity_refs,
                "relationship_ids": list(dict.fromkeys([item for item in supporting_relationships if item])),
            },
            "supporting_path_ids": [],
        }

    @staticmethod
    def _locate_text(raw_content: str, text_unit_text: str, context_chars: int = 120) -> dict[str, Any]:
        raw_content = raw_content or ""
        text_unit_text = text_unit_text or ""
        normalized_text = text_unit_text.strip()

        if normalized_text and normalized_text in raw_content:
            start = raw_content.index(normalized_text)
            end = start + len(normalized_text)
            snippet_start = max(0, start - context_chars)
            snippet_end = min(len(raw_content), end + context_chars)
            snippet = raw_content[snippet_start:snippet_end]
            paragraph_index = raw_content[:start].count("\n\n") + 1
            return {
                "snippet": snippet,
                "char_start": start,
                "char_end": end,
                "paragraph_index": paragraph_index,
            }

        paragraphs = [part for part in raw_content.split("\n\n") if part.strip()]
        best_idx = None
        best_score = -1
        for idx, paragraph in enumerate(paragraphs, start=1):
            score = len(set(paragraph) & set(normalized_text))
            if score > best_score:
                best_score = score
                best_idx = idx

        snippet = normalized_text[: context_chars * 2] if normalized_text else raw_content[: context_chars * 2]
        return {
            "snippet": snippet,
            "char_start": None,
            "char_end": None,
            "paragraph_index": best_idx,
        }

    def _build_subgraph(
        self,
        seed_entities: dict[str, list[str]],
        paths: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        max_nodes: int = 30,
    ) -> dict[str, list[dict[str, Any]]]:
        selected_nodes = set(seed_entities.get("all_entities", []))
        path_edge_ids = set()
        evidence_edge_ids = set()
        node_path_ids: dict[str, set[str]] = {}
        node_evidence_ids: dict[str, set[str]] = {}
        edge_path_ids: dict[str, set[str]] = {}
        edge_evidence_ids: dict[str, set[str]] = {}

        for path in paths:
            selected_nodes.update(path.get("nodes", []))
            path_id = str(path.get("id", ""))
            for node in path.get("nodes", []):
                if path_id:
                    node_path_ids.setdefault(node, set()).add(path_id)
            for edge in path.get("edges", []):
                if edge.get("id"):
                    path_edge_ids.add(edge["id"])
                    evidence_edge_ids.add(edge["id"])
                    if path_id:
                        edge_path_ids.setdefault(edge["id"], set()).add(path_id)

        for item in evidence:
            evidence_id = str(item.get("id", ""))
            for entity_title in item.get("graph_refs", {}).get("entity_titles", []):
                selected_nodes.add(entity_title)
                if evidence_id:
                    node_evidence_ids.setdefault(entity_title, set()).add(evidence_id)
            for rel_id in item.get("graph_refs", {}).get("relationship_ids", []):
                evidence_edge_ids.add(rel_id)
                if evidence_id:
                    edge_evidence_ids.setdefault(rel_id, set()).add(evidence_id)
                rel_row = self.relationship_by_id.get(rel_id)
                if rel_row is not None:
                    source = str(rel_row.get("source", ""))
                    target = str(rel_row.get("target", ""))
                    selected_nodes.add(source)
                    selected_nodes.add(target)
                    if evidence_id:
                        node_evidence_ids.setdefault(source, set()).add(evidence_id)
                        node_evidence_ids.setdefault(target, set()).add(evidence_id)

        # Expand one hop around selected nodes until max_nodes.
        frontier = list(selected_nodes)
        for node in frontier:
            if len(selected_nodes) >= max_nodes:
                break
            if node not in self.graph:
                continue
            neighbors = sorted(
                self.graph.neighbors(node),
                key=lambda item: (
                    -self._safe_float(self.graph.get_edge_data(node, item, default={}).get("weight"), 0.0),
                    -self.graph.degree(item),
                    item,
                ),
            )
            for neighbor in neighbors:
                selected_nodes.add(neighbor)
                if len(selected_nodes) >= max_nodes:
                    break

        nodes = []
        for node in sorted(selected_nodes):
            if node not in self.graph:
                continue
            attrs = self.graph.nodes[node]
            nodes.append({
                "id": str(attrs.get("id", node)),
                "label": node,
                "title": node,
                "type": str(attrs.get("type", "ENTITY")),
                "description": str(attrs.get("description", "") or ""),
                "degree": self.graph.degree(node),
                "is_query_entity": node in seed_entities.get("query_entities", []),
                "is_answer_entity": node in seed_entities.get("answer_entities", []),
                "is_seed": node in seed_entities.get("all_entities", []),
                "is_path_node": any(node in path.get("nodes", []) for path in paths),
                "path_ids": sorted(node_path_ids.get(node, set())),
                "evidence_ids": sorted(node_evidence_ids.get(node, set())),
            })

        node_titles = {node["label"] for node in nodes}
        edges = []
        for source, target, attrs in self.graph.edges(data=True):
            if source not in node_titles or target not in node_titles:
                continue
            edge_id = str(attrs.get("id", ""))
            edges.append({
                "id": edge_id,
                "source": source,
                "target": target,
                "description": str(attrs.get("description", "") or ""),
                "weight": self._safe_float(attrs.get("weight"), 0.0),
                "text_unit_ids": self._normalize_array(attrs.get("text_unit_ids")),
                "is_path_edge": edge_id in path_edge_ids,
                "is_evidence_edge": edge_id in evidence_edge_ids,
                "path_ids": sorted(edge_path_ids.get(edge_id, set())),
                "evidence_ids": sorted(edge_evidence_ids.get(edge_id, set())),
            })

        edges.sort(key=lambda item: (-item["is_path_edge"], -item["weight"], item["source"], item["target"]))
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "path_node_count": len([node for node in nodes if node.get("is_path_node")]),
                "path_edge_count": len([edge for edge in edges if edge.get("is_path_edge")]),
                "evidence_edge_count": len([edge for edge in edges if edge.get("is_evidence_edge")]),
            },
        }

    @staticmethod
    def _build_summary(
        seed_entities: dict[str, list[str]],
        paths: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        subgraph: dict[str, Any],
    ) -> dict[str, Any]:
        stats = subgraph.get("stats", {}) or {}
        source_map_hits = sum(1 for item in evidence if item.get("source_map_available"))
        evidence_count = len(evidence)

        return {
            "query_focus": seed_entities.get("query_entities", []),
            "answer_focus": seed_entities.get("answer_entities", []),
            "path_count": len(paths),
            "evidence_count": evidence_count,
            "subgraph_node_count": int(stats.get("node_count", 0) or 0),
            "subgraph_edge_count": int(stats.get("edge_count", 0) or 0),
            "source_map_coverage_ratio": round(source_map_hits / evidence_count, 3) if evidence_count else 0.0,
            "source_documents": list(dict.fromkeys([
                str(item.get("document_title", "") or "")
                for item in evidence
                if str(item.get("document_title", "") or "")
            ])),
        }

    def build_explanation(
        self,
        query: str,
        response: str = "",
        max_nodes: int = 30,
        max_paths: int = 3,
        max_evidence: int = 6,
    ) -> dict[str, Any]:
        seed_entities = self.select_seed_entities(query=query, response=response)
        paths = self._find_paths(
            query_entities=seed_entities.get("query_entities", []),
            answer_entities=seed_entities.get("answer_entities", []),
            max_paths=max_paths,
        )
        evidence = self._collect_evidence(seed_entities=seed_entities, paths=paths, max_evidence=max_evidence)
        self._link_paths_and_evidence(paths=paths, evidence=evidence)
        subgraph = self._build_subgraph(
            seed_entities=seed_entities,
            paths=paths,
            evidence=evidence,
            max_nodes=max_nodes,
        )

        limitations = []
        if not evidence:
            limitations.append("当前回答未找到稳定的支撑证据片段。")
        if not paths:
            limitations.append("当前问题未形成稳定的图谱关键路径，子图仅展示相关邻域。")
        limitations.append("当前索引仅支持文档/片段/字符区间级溯源；页码与版面坐标需要PDF版面解析元数据。")
        if any(item.get("source_map_available") for item in evidence):
            limitations = [item for item in limitations if "页码与版面坐标" not in item]
            limitations.append("部分证据已通过 PDF source map 提供页码/段落/bbox；其余文档仍只有片段级定位。")

        reasoning_steps = self._build_reasoning_steps(
            response=response,
            paths=paths,
            evidence=evidence,
        )
        for path in paths:
            path_relationship_ids = {edge.get("id") for edge in path.get("edges", []) if edge.get("id")}
            path_entity_ids = set(path.get("nodes", []))
            supporting_evidence_ids = []
            for item in evidence:
                evidence_relationship_ids = set(item.get("graph_refs", {}).get("relationship_ids", []))
                evidence_entity_titles = set(item.get("graph_refs", {}).get("entity_titles", []))
                if path_relationship_ids & evidence_relationship_ids or path_entity_ids & evidence_entity_titles:
                    supporting_evidence_ids.append(str(item.get("id", "")))
            path["supporting_evidence_ids"] = [item for item in supporting_evidence_ids if item]

        for item in evidence:
            evidence_relationship_ids = set(item.get("graph_refs", {}).get("relationship_ids", []))
            evidence_entity_titles = set(item.get("graph_refs", {}).get("entity_titles", []))
            supporting_path_ids = []
            for path in paths:
                path_relationship_ids = {edge.get("id") for edge in path.get("edges", []) if edge.get("id")}
                path_entity_ids = set(path.get("nodes", []))
                if path_relationship_ids & evidence_relationship_ids or path_entity_ids & evidence_entity_titles:
                    supporting_path_ids.append(str(path.get("id", "")))
            item["supporting_path_ids"] = [path_id for path_id in supporting_path_ids if path_id]
            item["locator_text"] = self._format_location_text(item.get("location", {}))

        subgraph_summary = {
            "node_count": subgraph.get("stats", {}).get("node_count", len(subgraph.get("nodes", []))),
            "edge_count": subgraph.get("stats", {}).get("edge_count", len(subgraph.get("edges", []))),
            "path_count": len(paths),
            "evidence_count": len(evidence),
            "path_edge_count": subgraph.get("stats", {}).get("path_edge_count", len([edge for edge in subgraph.get("edges", []) if edge.get("is_path_edge")])),
            "evidence_edge_count": subgraph.get("stats", {}).get("evidence_edge_count", len([edge for edge in subgraph.get("edges", []) if edge.get("is_evidence_edge")])),
            "path_node_count": subgraph.get("stats", {}).get("path_node_count", len([node for node in subgraph.get("nodes", []) if node.get("is_path_node")])),
            "document_count": len({item.get("document_title") for item in evidence if item.get("document_title")}),
            "provenance_coverage": (
                "page_bbox_partial"
                if any(item.get("source_map_available") for item in evidence)
                else ("snippet_only" if evidence else "graph_only")
            ),
        }
        summary = self._build_summary(
            seed_entities=seed_entities,
            paths=paths,
            evidence=evidence,
            subgraph=subgraph,
        )
        summary["provenance_coverage"] = subgraph_summary.get("provenance_coverage", "graph_only")
        summary["document_count"] = subgraph_summary.get("document_count", 0)

        return {
            "schema_version": 2,
            "query_id": hashlib.md5(f"{query}\n{response}".encode("utf-8")).hexdigest(),
            "query": query,
            "answer": response,
            "seed_entities": seed_entities,
            "reasoning_steps": reasoning_steps,
            "paths": paths,
            "evidence": evidence,
            "subgraph": subgraph,
            "subgraph_summary": subgraph_summary,
            "summary": summary,
            "provenance_map": {
                "documents": [
                    {
                        "id": str(item.get("id", "")),
                        "title": str(item.get("title", "")),
                    }
                    for _, item in self.document_df.iterrows()
                ],
            },
            "limitations": limitations,
        }


def build_answer_explanation(
    input_dir: str,
    query: str,
    response: str = "",
    max_nodes: int = 30,
    max_paths: int = 3,
    max_evidence: int = 6,
) -> dict[str, Any]:
    engine = GraphExplanationEngine(input_dir)
    return engine.build_explanation(
        query=query,
        response=response,
        max_nodes=max_nodes,
        max_paths=max_paths,
        max_evidence=max_evidence,
    )


def render_explanation_html(explanation: dict[str, Any]) -> str:
    """Render a simple HTML summary for the Gradio demo."""
    reasoning_steps = explanation.get("reasoning_steps", [])
    paths = explanation.get("paths", [])
    evidence = explanation.get("evidence", [])
    limitations = explanation.get("limitations", [])
    subgraph_summary = explanation.get("subgraph_summary", {})

    step_items = []
    for idx, step in enumerate(reasoning_steps, start=1):
        claim = html.escape(str(step.get("claim", "") or ""))
        entities = "、".join(step.get("claim_entities", [])[:6]) or "未显式命中实体"
        path_refs = " / ".join(
            html.escape(path.get("explanation", ""))
            for path in step.get("supporting_paths", [])[:2]
        ) or "未匹配到稳定路径"
        evidence_refs = "；".join(
            html.escape(
                f"{item.get('document_title', '未知文档')} @ "
                f"{'页码 ' + str(item.get('location', {}).get('page')) if item.get('location', {}).get('page') else '段落 ' + str(item.get('location', {}).get('paragraph_index') or '未定位')}"
            )
            for item in step.get("supporting_evidence", [])[:2]
        ) or "未匹配到稳定证据"
        step_items.append(
            f"""
            <div style='padding:12px;border:1px solid #1d4ed8;border-radius:10px;background:#0b1120;margin-bottom:10px;'>
              <div style='color:#bfdbfe;font-weight:600;'>步骤 {idx} · 置信度 {step.get('confidence', 0):.0%}</div>
              <div style='margin-top:6px;color:#eff6ff;white-space:pre-wrap;'>{claim}</div>
              <div style='margin-top:8px;color:#93c5fd;font-size:12px;'>实体：{html.escape(entities)}</div>
              <div style='margin-top:6px;color:#cbd5e1;font-size:12px;'>路径：{path_refs}</div>
              <div style='margin-top:6px;color:#cbd5e1;font-size:12px;'>证据：{evidence_refs}</div>
            </div>
            """
        )

    path_items = []
    for idx, path in enumerate(paths, start=1):
        edge_desc = "".join(
            f"<li><b>{html.escape(str(edge['source']))} → {html.escape(str(edge['target']))}</b>：{html.escape(str(edge.get('description', '') or '关系已命中'))}</li>"
            for edge in path.get("edges", [])
        )
        supporting_evidence_ids = "、".join(path.get("supporting_evidence_ids", [])) or "无"
        path_items.append(
            f"""
            <div style='padding:12px;border:1px solid #2f3b52;border-radius:10px;background:#111827;margin-bottom:10px;'>
              <div style='color:#93c5fd;font-weight:600;'>路径 {idx} · 分数 {path.get('score', 0)}</div>
              <div style='margin-top:6px;color:#e5e7eb;'>{html.escape(str(path.get('explanation', '') or ''))}</div>
              <div style='margin-top:6px;color:#94a3b8;font-size:12px;'>路径ID：{html.escape(str(path.get('id', '') or ''))} · 关联证据：{html.escape(supporting_evidence_ids)}</div>
              <ul style='margin:8px 0 0 18px;color:#cbd5e1;'>{edge_desc}</ul>
            </div>
            """
        )

    evidence_items = []
    for item in evidence:
        loc_text = item.get("locator_text") or GraphExplanationEngine._format_location_text(item.get("location", {}))
        supporting_path_ids = "、".join(item.get("supporting_path_ids", [])) or "无"
        evidence_items.append(
            f"""
            <div style='padding:12px;border:1px solid #334155;border-radius:10px;background:#0f172a;margin-bottom:10px;'>
              <div style='color:#f8fafc;font-weight:600;'>{html.escape(str(item.get('document_title', '未知文档') or '未知文档'))}</div>
              <div style='color:#94a3b8;font-size:12px;margin-top:2px;'>text_unit_id: {html.escape(str(item.get('text_unit_id', '') or ''))} · {html.escape(loc_text)}</div>
              <div style='color:#64748b;font-size:12px;margin-top:4px;'>支撑路径：{html.escape(supporting_path_ids)}</div>
              <div style='margin-top:8px;color:#e2e8f0;white-space:pre-wrap;'>{html.escape(str(item.get('snippet', '') or ''))}</div>
            </div>
            """
        )

    limitation_html = "".join(f"<li>{item}</li>" for item in limitations)
    if not path_items:
        path_items = ["<div style='color:#94a3b8;'>暂无稳定关键路径。</div>"]
    if not evidence_items:
        evidence_items = ["<div style='color:#94a3b8;'>暂无稳定证据。</div>"]
    if not step_items:
        step_items = ["<div style='color:#94a3b8;'>当前回答暂未拆解出稳定的推理步骤。</div>"]

    return f"""
    <div style='background:#020617;border-radius:12px;padding:16px;'>
      <h3 style='color:#f8fafc;margin:0 0 12px 0;'>🧠 推理路径与溯源证据</h3>
      <div style='margin-bottom:12px;padding:10px 12px;border:1px solid #334155;border-radius:10px;background:#0f172a;color:#cbd5e1;font-size:13px;'>
        子图摘要：节点 {subgraph_summary.get('node_count', 0)} · 边 {subgraph_summary.get('edge_count', 0)} · 路径节点 {subgraph_summary.get('path_node_count', 0)} · 路径 {subgraph_summary.get('path_count', 0)} · 证据 {subgraph_summary.get('evidence_count', 0)} · 文档 {subgraph_summary.get('document_count', 0)} · 溯源覆盖 {html.escape(str(subgraph_summary.get('provenance_coverage', 'unknown')))}
      </div>
      <div style='display:grid;grid-template-columns:1fr;gap:14px;'>
        <div>
          <div style='color:#cbd5e1;font-weight:600;margin-bottom:8px;'>答案拆解</div>
          {''.join(step_items)}
        </div>
        <div>
          <div style='color:#cbd5e1;font-weight:600;margin-bottom:8px;'>关键推理路径</div>
          {''.join(path_items)}
        </div>
        <div>
          <div style='color:#cbd5e1;font-weight:600;margin-bottom:8px;'>支撑证据</div>
          {''.join(evidence_items)}
        </div>
        <div>
          <div style='color:#cbd5e1;font-weight:600;margin-bottom:4px;'>当前限制</div>
          <ul style='margin:0 0 0 18px;color:#94a3b8;'>{limitation_html}</ul>
        </div>
      </div>
    </div>
    """


def render_explanation_text(explanation: dict[str, Any]) -> str:
    """Render a plain-text summary for CLI/demo output."""
    lines = []
    lines.append(f"问题: {explanation.get('query', '')}")
    answer = str(explanation.get("answer", "") or "").strip()
    if answer:
        lines.append(f"答案: {answer}")

    summary = explanation.get("subgraph_summary", {})
    if summary:
        lines.append(
            "子图摘要: "
            f"节点 {summary.get('node_count', 0)} | "
            f"边 {summary.get('edge_count', 0)} | "
            f"路径 {summary.get('path_count', 0)} | "
            f"证据 {summary.get('evidence_count', 0)}"
        )

    steps = explanation.get("reasoning_steps", []) or []
    if steps:
        lines.append("")
        lines.append("答案拆解:")
        for idx, step in enumerate(steps, start=1):
            lines.append(f"  {idx}. {step.get('claim', '')}")
            if step.get("claim_entities"):
                lines.append(f"     实体: {'、'.join(step.get('claim_entities', []))}")
            for path in step.get("supporting_paths", [])[:2]:
                lines.append(
                    f"     路径: {path.get('explanation', '')} "
                    f"(id={path.get('id', '')}, score={path.get('score', 0)})"
                )
            for item in step.get("supporting_evidence", [])[:2]:
                loc = item.get("location", {})
                location_parts = []
                if loc.get("page") is not None:
                    location_parts.append(f"页码 {loc.get('page')}")
                if loc.get("paragraph_index") is not None:
                    location_parts.append(f"段落 {loc.get('paragraph_index')}")
                if loc.get("bbox"):
                    location_parts.append(f"bbox {loc.get('bbox')}")
                location_text = " / ".join(location_parts) or "未定位"
                lines.append(f"     证据: {item.get('document_title', '未知文档')} | {location_text}")

    paths = explanation.get("paths", []) or []
    if paths:
        lines.append("")
        lines.append("关键路径:")
        for idx, path in enumerate(paths, start=1):
            lines.append(
                f"  {idx}. {path.get('explanation', '')} "
                f"(id={path.get('id', '')}, score={path.get('score', 0)}, evidence={','.join(path.get('supporting_evidence_ids', [])) or '无'})"
            )

    evidence = explanation.get("evidence", []) or []
    if evidence:
        lines.append("")
        lines.append("支撑证据:")
        for idx, item in enumerate(evidence, start=1):
            loc = item.get("location", {})
            location_text = item.get("locator_text") or GraphExplanationEngine._format_location_text(loc)
            lines.append(
                f"  {idx}. {item.get('document_title', '未知文档')} | {item.get('text_unit_id', '')} | {location_text} | paths={','.join(item.get('supporting_path_ids', [])) or '无'}"
            )
            snippet = str(item.get("snippet", "") or "").replace("\n", " ").strip()
            if snippet:
                lines.append(f"     {snippet[:220]}")

    limitations = explanation.get("limitations", []) or []
    if limitations:
        lines.append("")
        lines.append("限制:")
        for item in limitations:
            lines.append(f"  - {item}")

    edges = explanation.get("subgraph", {}).get("edges", []) or []
    if edges:
        lines.append("")
        lines.append("子图边:")
        for edge in edges[:12]:
            tags = []
            if edge.get("is_path_edge"):
                tags.append("PATH")
            if edge.get("is_evidence_edge"):
                tags.append("EVIDENCE")
            suffix = f" [{'|'.join(tags)}]" if tags else ""
            lines.append(
                f"  - {edge.get('source', '')} -> {edge.get('target', '')}{suffix} | path_ids={','.join(edge.get('path_ids', [])) or '无'} | evidence_ids={','.join(edge.get('evidence_ids', [])) or '无'}"
            )

    return "\n".join(lines)
