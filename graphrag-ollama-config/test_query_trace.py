"""
Offline smoke test for the query trace builder.
"""

import os

from query_trace import build_query_trace


def run_demo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "output")

    query = "政务服务事项办理中，材料预审与正式受理有什么区别？"
    response = (
        "材料预审的主要作用是帮助申请人提前发现缺项、错项和格式问题，"
        "减少正式提交后的反复补正；正式受理则表示材料已经满足基本条件，"
        "进入法定流程和时限管理。"
    )

    trace = build_query_trace(
        query=query,
        response=response,
        input_dir=input_dir,
        query_type="demo",
    )

    print("=" * 70)
    print("Trace Markdown Preview")
    print("=" * 70)
    print(trace["markdown"])
    print("=" * 70)

    assert trace["subgraph"]["node_count"] > 0, "Expected non-empty subgraph"
    assert len(trace["evidence"]) > 0, "Expected at least one evidence item"


if __name__ == "__main__":
    run_demo()
