"""
Smoke test for explanation output.

Run from the graphrag-ollama-config directory:
    ../.venv/bin/python test_explanation_demo.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explanation_engine import build_answer_explanation, render_explanation_text


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "output")

    if not os.path.exists(os.path.join(input_dir, "create_final_nodes.parquet")):
        print(f"❌ 未找到索引输出目录: {input_dir}")
        return

    query = "政务服务事项办理中，材料预审与正式受理有什么区别？"
    answer = (
        "材料预审主要用于提前发现缺项、错项和格式问题，"
        "减少正式提交后的反复补正；正式受理则表示材料已满足基本条件，"
        "正式进入法定流程和时限管理。"
    )

    result = build_answer_explanation(
        input_dir=input_dir,
        query=query,
        response=answer,
        max_nodes=30,
        max_paths=3,
        max_evidence=6,
    )

    print("=" * 72)
    print("🧠 Explanation Demo")
    print("=" * 72)
    print(render_explanation_text(result))
    print("-" * 72)
    print("JSON预览:")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:3000])
    print("=" * 72)


if __name__ == "__main__":
    main()
