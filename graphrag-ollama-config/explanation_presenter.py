"""
Presentation helpers for explanation payloads.
"""

from __future__ import annotations

from typing import Any


def _format_location(location: dict[str, Any]) -> str:
    if not isinstance(location, dict):
        return "未定位"

    paragraph = location.get("paragraph_index")
    page = location.get("page")
    bbox = location.get("bbox")
    char_start = location.get("char_start")
    char_end = location.get("char_end")

    page_text = ""
    if isinstance(page, list):
        page_text = f"页码 {', '.join(str(item) for item in page)}"
    elif page is not None:
        page_text = f"页码 {page}"

    char_text = ""
    if char_start is not None and char_end is not None:
        char_text = f"字符 {char_start}-{char_end}"

    bbox_text = f"bbox {bbox}" if bbox else ""
    return " / ".join(
        item for item in [
            page_text,
            f"段落 {paragraph}" if paragraph else "",
            char_text,
            bbox_text,
        ] if item
    ) or "未定位"


def render_explanation_html(explanation: dict[str, Any]) -> str:
    paths = explanation.get("paths", [])
    evidence = explanation.get("evidence", [])
    limitations = explanation.get("limitations", [])
    summary = explanation.get("summary", {})
    subgraph = explanation.get("subgraph", {})
    stats = subgraph.get("stats", {})

    metrics = [
        ("关键路径", summary.get("path_count", len(paths))),
        ("证据片段", summary.get("evidence_count", len(evidence))),
        ("子图节点", summary.get("subgraph_node_count", stats.get("node_count", 0))),
        ("子图边", summary.get("subgraph_edge_count", stats.get("edge_count", 0))),
        ("PDF定位覆盖", f"{int(round(float(summary.get('source_map_coverage_ratio', 0.0)) * 100))}%"),
    ]
    metric_html = "".join(
        f"""
        <div style='padding:12px;border:1px solid #334155;border-radius:10px;background:#0f172a;'>
          <div style='color:#94a3b8;font-size:12px;'>{label}</div>
          <div style='color:#f8fafc;font-size:20px;font-weight:700;margin-top:4px;'>{value}</div>
        </div>
        """
        for label, value in metrics
    )

    path_items = []
    for idx, path in enumerate(paths, start=1):
        edge_desc = "".join(
            f"<li><b>{edge['source']} → {edge['target']}</b>：{edge.get('description', '') or '关系已命中'}</li>"
            for edge in path.get("edges", [])
        )
        docs = "、".join(path.get("supporting_documents", [])) or "未关联文档"
        evidence_text = "、".join(path.get("evidence_ids", [])) or "无"
        path_items.append(
            f"""
            <div style='padding:12px;border:1px solid #2f3b52;border-radius:10px;background:#111827;margin-bottom:10px;'>
              <div style='color:#93c5fd;font-weight:600;'>路径 {idx} · 分数 {path.get('score', 0)}</div>
              <div style='margin-top:6px;color:#e5e7eb;'>{path.get('explanation', '')}</div>
              <div style='margin-top:8px;color:#94a3b8;font-size:12px;'>支撑文档：{docs}</div>
              <div style='margin-top:4px;color:#94a3b8;font-size:12px;'>证据ID：{evidence_text}</div>
              <ul style='margin:8px 0 0 18px;color:#cbd5e1;'>{edge_desc}</ul>
            </div>
            """
        )

    evidence_items = []
    for item in evidence:
        loc_text = _format_location(item.get("location", {}))
        path_text = "、".join(item.get("path_ids", [])) or "未绑定路径"
        evidence_items.append(
            f"""
            <div style='padding:12px;border:1px solid #334155;border-radius:10px;background:#0f172a;margin-bottom:10px;'>
              <div style='color:#f8fafc;font-weight:600;'>{item.get('document_title', '未知文档')}</div>
              <div style='color:#94a3b8;font-size:12px;margin-top:2px;'>text_unit_id: {item.get('text_unit_id', '')} · {loc_text}</div>
              <div style='color:#94a3b8;font-size:12px;margin-top:2px;'>关联路径：{path_text}</div>
              <div style='margin-top:8px;color:#e2e8f0;white-space:pre-wrap;'>{item.get('snippet', '')}</div>
            </div>
            """
        )

    limitation_html = "".join(f"<li>{item}</li>" for item in limitations)
    if not path_items:
        path_items = ["<div style='color:#94a3b8;'>暂无稳定关键路径。</div>"]
    if not evidence_items:
        evidence_items = ["<div style='color:#94a3b8;'>暂无稳定证据。</div>"]

    return f"""
    <div style='background:#020617;border-radius:12px;padding:16px;'>
      <h3 style='color:#f8fafc;margin:0 0 12px 0;'>🧠 推理路径与溯源证据</h3>
      <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:14px;'>
        {metric_html}
      </div>
      <div style='display:grid;grid-template-columns:1fr;gap:14px;'>
        <div>
          <div style='color:#cbd5e1;font-weight:600;margin-bottom:8px;'>关键推理路径</div>
          {''.join(path_items)}
        </div>
        <div>
          <div style='color:#cbd5e1;font-weight:600;margin-bottom:8px;'>支撑证据</div>
          {''.join(evidence_items)}
        </div>
        <div>
          <div style='color:#cbd5e1;font-weight:600;margin-bottom:8px;'>子图摘要</div>
          <div style='padding:12px;border:1px solid #334155;border-radius:10px;background:#0f172a;color:#e2e8f0;'>
            种子实体：{", ".join(summary.get("query_focus", [])) or "无"}<br>
            回答实体：{", ".join(summary.get("answer_focus", [])) or "无"}<br>
            主要证据文档：{", ".join(summary.get("source_documents", [])) or "无"}
          </div>
        </div>
        <div>
          <div style='color:#cbd5e1;font-weight:600;margin-bottom:4px;'>当前限制</div>
          <ul style='margin:0 0 0 18px;color:#94a3b8;'>{limitation_html}</ul>
        </div>
      </div>
    </div>
    """


def render_explanation_text(explanation: dict[str, Any], max_subgraph_items: int = 12) -> str:
    summary = explanation.get("summary", {})
    subgraph = explanation.get("subgraph", {})
    stats = subgraph.get("stats", {})
    paths = explanation.get("paths", [])
    evidence = explanation.get("evidence", [])
    limitations = explanation.get("limitations", [])

    lines = [
        "=" * 72,
        "Explanation Package",
        "=" * 72,
        f"问题: {explanation.get('query', '')}",
        f"答案: {explanation.get('answer', '')}",
        f"路径数: {summary.get('path_count', len(paths))}",
        f"证据数: {summary.get('evidence_count', len(evidence))}",
        f"子图: {summary.get('subgraph_node_count', stats.get('node_count', 0))} 节点 / {summary.get('subgraph_edge_count', stats.get('edge_count', 0))} 边",
        f"PDF定位覆盖: {int(round(float(summary.get('source_map_coverage_ratio', 0.0)) * 100))}%",
        "-" * 72,
        "关键路径:",
    ]

    if paths:
        for idx, path in enumerate(paths, start=1):
            lines.append(f"[路径 {idx}] {path.get('explanation', '')} (score={path.get('score', 0)})")
            lines.append(f"  文档: {', '.join(path.get('supporting_documents', [])) or '无'}")
            for edge in path.get("edges", []):
                lines.append(f"  - {edge.get('source')} -> {edge.get('target')} | {edge.get('description', '') or '关系已命中'}")
    else:
        lines.append("暂无稳定关键路径。")

    lines.extend(["-" * 72, "支撑证据:"])
    if evidence:
        for idx, item in enumerate(evidence, start=1):
            lines.append(f"[证据 {idx}] {item.get('document_title', '未知文档')} | {item.get('text_unit_id', '')}")
            lines.append(f"  定位: {_format_location(item.get('location', {}))}")
            lines.append(f"  路径: {', '.join(item.get('path_ids', [])) or '未绑定路径'}")
            lines.append(f"  摘录: {str(item.get('snippet', '')).replace(chr(10), ' ')}")
    else:
        lines.append("暂无稳定证据。")

    lines.extend(["-" * 72, "子图节点:"])
    nodes = subgraph.get("nodes", [])
    if nodes:
        for node in nodes[:max_subgraph_items]:
            flags = []
            if node.get("is_query_entity"):
                flags.append("Q")
            if node.get("is_answer_entity"):
                flags.append("A")
            if node.get("is_path_node"):
                flags.append("P")
            flag_text = f" [{'|'.join(flags)}]" if flags else ""
            lines.append(f"- {node.get('label', node.get('title', ''))}{flag_text}")
        if len(nodes) > max_subgraph_items:
            lines.append(f"... 其余 {len(nodes) - max_subgraph_items} 个节点省略")
    else:
        lines.append("暂无子图节点。")

    lines.extend(["-" * 72, "子图边:"])
    edges = subgraph.get("edges", [])
    if edges:
        for edge in edges[:max_subgraph_items]:
            flags = []
            if edge.get("is_path_edge"):
                flags.append("PATH")
            if edge.get("is_evidence_edge"):
                flags.append("EVIDENCE")
            flag_text = f" [{'|'.join(flags)}]" if flags else ""
            lines.append(f"- {edge.get('source')} -> {edge.get('target')}{flag_text}")
        if len(edges) > max_subgraph_items:
            lines.append(f"... 其余 {len(edges) - max_subgraph_items} 条边省略")
    else:
        lines.append("暂无子图边。")

    if limitations:
        lines.extend(["-" * 72, "限制:"])
        for item in limitations:
            lines.append(f"- {item}")

    lines.append("=" * 72)
    return "\n".join(lines)
