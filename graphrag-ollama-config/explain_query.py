"""
CLI demo for answer explainability output.

Usage:
    ../.venv/bin/python explain_query.py \
      --query "政务服务事项办理中，材料预审与正式受理有什么区别？" \
      --answer "材料预审主要用于提前发现缺项..."
"""

from __future__ import annotations

import argparse
import json
import os

from explanation_engine import build_answer_explanation, render_explanation_text


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = os.path.join(base_dir, "output")

    parser = argparse.ArgumentParser(description="Print answer explainability bundle.")
    parser.add_argument("--input-dir", default=default_input_dir, help="GraphRAG artifacts directory")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--answer", default="", help="Answer text to explain")
    parser.add_argument("--max-nodes", type=int, default=35)
    parser.add_argument("--max-paths", type=int, default=3)
    parser.add_argument("--max-evidence", type=int, default=6)
    parser.add_argument("--json", action="store_true", help="Print full JSON payload after text summary")
    args = parser.parse_args()

    result = build_answer_explanation(
        input_dir=args.input_dir,
        query=args.query,
        response=args.answer,
        max_nodes=args.max_nodes,
        max_paths=args.max_paths,
        max_evidence=args.max_evidence,
    )

    print(render_explanation_text(result))
    if args.json:
        print("\nJSON:")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
