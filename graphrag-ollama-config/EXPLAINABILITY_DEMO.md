# VeritasGraph 可解释问答 Demo 说明

## 已新增能力

当用户提问后，系统现在会同时输出：

1. **答案**
2. **答案拆解步骤**
3. **最相关子图**
4. **关键推理路径**
5. **支撑证据**
6. **结构化 JSON 解释结果**

其中：

- 子图里会高亮问题实体、回答实体、关键路径、证据边
- 回答会拆成若干步骤，每一步绑定对应路径和证据
- 证据会回溯到 `document -> text_unit -> snippet`
- 若文档来自 **PDF 上传并生成了 source map**，解释面板会尽量展示：
  - 页码
  - 段落
  - bbox 坐标
- 当前基础溯源粒度是：
  - 文档名
  - 段落号
  - 字符区间
- 只有带 source map 的 PDF 才能稳定补充页码和版面坐标

---

## 主要文件

- `app.py`
  - Gradio 主界面
  - “解释与溯源”Tab 展示 HTML 和 JSON
- `explanation_engine.py`
  - 解释结果构建核心
  - 负责生成：
    - `seed_entities`
    - `reasoning_steps`
    - `paths`
    - `evidence`
    - `subgraph`
    - `subgraph_summary`
    - `limitations`
- `graph_visualizer.py`
  - explanation graph 高亮渲染
- `explain_query.py`
  - 命令行 demo
  - 可直接打印答案、步骤、路径、子图摘要和证据
- `test_explanation_demo.py`
  - 命令行 smoke test
- `pdf_source_map.py`
  - PDF 页码 / 段落 / bbox sidecar 提取与解析
- `test_pdf_source_map.py`
  - source map 定位测试

---

## 启动 Demo

在 `VeritasGraph` 目录下运行：

```bash
./.venv/bin/python graphrag-ollama-config/app.py --port 7860
```

打开：

```text
http://127.0.0.1:7860
```

建议测试问题：

- `政务服务事项办理中，材料预审与正式受理有什么区别？`
- `网格化治理中，街道、社区和网格员如何协同处置问题？`
- `国有企业合规管理通常包括哪些关键环节？`

如需启用 PDF 页码 / bbox：

```bash
./.venv/bin/pip install pdfplumber
```

---

## 命令行测试

```bash
cd graphrag-ollama-config
../.venv/bin/python test_explanation_demo.py
```

会打印：

- 答案拆解步骤
- 路径数量
- 证据数量
- 子图节点/边数量
- 关键路径
- 证据摘要
- JSON 预览

也可以直接打印单次解释包：

```bash
cd graphrag-ollama-config
../.venv/bin/python explain_query.py \
  --query "政务服务事项办理中，材料预审与正式受理有什么区别？" \
  --answer "材料预审主要用于提前发现缺项、错项和格式问题，减少正式提交后的反复补正；正式受理则表示材料已满足基本条件，正式进入法定流程和时限管理。"
```

真实 PDF smoke test：

```bash
cd graphrag-ollama-config
../.venv/bin/python test_real_pdf_ingest.py "../VeritasGraph - A Sovereign GraphRAG Framework for Enterprise-Grade AI with Verifiable Attribution.pdf"
```

会验证：

- PDF 文本抽取
- source map sidecar 生成
- 页码 / 段落 / bbox 解析

---

## 当前实现策略

### 1. 关键路径

基于图谱中的实体和关系，使用加权最短路生成 1~3 条解释性路径。

> 注意：这是“解释性链路”，不是形式化证明。

### 2. 证据溯源

优先从路径上的关系边拿 `text_unit_ids`，再回溯到：

- `create_final_text_units.parquet`
- `create_final_documents.parquet`

最终生成：

- 文档标题
- 原文摘录
- text unit id
- 图谱引用

### 3. 答案拆解

回答会先拆成 1~4 个步骤，每个步骤绑定：

- `claim`
- `supporting_paths`
- `supporting_evidence`
- `referenced_entities`
- `confidence`

这层结构适合后续前端高亮、审计导出和测试断言。

### 4. 子图

子图来源于：

- query entities
- answer entities
- path nodes / path edges
- evidence refs

并限制节点规模，避免 UI 过重。

---

## 已知限制

1. 当前 `reasoning_search` 模块在仓库中缺失，所以 UI 已自动隐藏不可用入口。
2. 当前 ingestion 主要仍是 txt/文本级；只有带 source map 的 PDF 才能稳定展示页码与 bbox。
3. 段落号仍然存在近似定位的情况。
4. 关键路径和答案拆解目前是工程解释链路，不等同于严格因果推理。

---

## 推荐下一步

### P1

- 增加“导出 JSON”按钮
- 增加证据去重/排序策略
- 为 answer step 增加更稳的句子级对齐

### P2

- 引入更稳的 PDF ingestion 元数据
- 支持点击证据后跳原文定位
- 支持 bbox 叠加高亮

### P3

- 独立前端重构
- 更强交互式路径筛选与回放
- 查询日志与审计留痕
