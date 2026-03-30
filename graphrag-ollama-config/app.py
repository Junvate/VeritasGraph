import gradio as gr
import os
import asyncio
import time
from urllib.parse import urlparse

import httpx
import pandas as pd
import tiktoken
from dotenv import load_dotenv

# Graph visualization
from graph_visualizer import (
    create_graph_html_for_query,
    create_explanation_graph_html,
    get_graph_stats,
    load_graph_data,
    extract_entities_from_response
)
from explanation_engine import (
    build_answer_explanation,
    render_explanation_html,
)

from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# Import OpenAI-compatible API configuration from separate module
from openai_config import get_api_type, get_llm_config, get_embedding_config

# Import Instant Knowledge ingest module
from ingest import (
    ingest_url,
    ingest_text_content,
    ingest_pdf_file,
    trigger_graphrag_index,
    trigger_graphrag_index_async,
    trigger_graphrag_index_with_progress,
    get_indexing_status,
    list_input_files, 
    delete_input_file,
    get_file_preview,
    check_dependencies
)

# Import PageIndex-inspired Reasoning Search (for 99%+ accuracy)
try:
    from reasoning_search import (
        enhanced_search,
        reasoning_local_search,
        reasoning_global_search,
        hybrid_reasoning_search,
        ReasoningSearchEngine
    )
    REASONING_SEARCH_AVAILABLE = True
except ImportError:
    REASONING_SEARCH_AVAILABLE = False
    print("提示：reasoning_search 模块不可用，将使用标准检索模式。")

# Load .env from the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'))
join = os.path.join
LAUNCH_THEME = gr.themes.Base()

PRESET_MAPPING = {
    "默认": {
        "community_level": 2,
        "response_type": "多段文字"
    },
    "详细": {
        "community_level": 4,
        "response_type": "多页报告"
    },
    "快速": {
        "community_level": 1,
        "response_type": "单段摘要"
    },
    "要点": {
        "community_level": 2,
        "response_type": "3-7条要点列表"
    },
    "全面": {
        "community_level": 5,
        "response_type": "多页报告"
    },
    "概览": {
        "community_level": 1,
        "response_type": "单页概览"
    },
    "聚焦": {
        "community_level": 3,
        "response_type": "多段文字"
    }
}

QUERY_TYPE_CHOICES = [
    ("全局检索", "global"),
    ("局部检索", "local"),
]

if REASONING_SEARCH_AVAILABLE:
    QUERY_TYPE_CHOICES.extend([
        ("推理检索", "reasoning"),
        ("混合检索", "hybrid"),
    ])

async def global_search(query, input_dir, community_level=2, temperature=0.5, response_type="Multiple Paragraphs"):
        llm_config = get_llm_config()

        llm = ChatOpenAI(
            api_key=llm_config["api_key"],
            api_base=llm_config["api_base"],
            model=llm_config["model"],
            api_type=llm_config["api_type"],
            max_retries=llm_config["max_retries"],
        )

        token_encoder = tiktoken.get_encoding("cl100k_base")

        COMMUNITY_REPORT_TABLE = "create_final_community_reports"
        ENTITY_TABLE = "create_final_nodes"
        ENTITY_EMBEDDING_TABLE = "create_final_entities"
        
        entity_df = pd.read_parquet(join(input_dir, f"{ENTITY_TABLE}.parquet"))
        report_df = pd.read_parquet(join(input_dir, f"{COMMUNITY_REPORT_TABLE}.parquet"))
        entity_embedding_df = pd.read_parquet(join(input_dir, f"{ENTITY_EMBEDDING_TABLE}.parquet"))

        reports = read_indexer_reports(report_df, entity_df, community_level)
        entities = read_indexer_entities(entity_df, entity_embedding_df, community_level)

        context_builder = GlobalCommunityContext(
            community_reports=reports,
            entities=entities,
            token_encoder=token_encoder,
        )

        context_builder_params = {
            "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
            "shuffle_data": True,
            "include_community_rank": True,
            "min_community_rank": 0,
            "community_rank_name": "rank",
            "include_community_weight": True,
            "community_weight_name": "occurrence weight",
            "normalize_community_weight": True,
            "max_tokens": 4000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
            "context_name": "Reports",
        }

        map_llm_params = {
            "max_tokens": 1000,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        reduce_llm_params = {
            "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
            "temperature": temperature,
        }

        search_engine = GlobalSearch(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            max_data_tokens=5000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
            map_llm_params=map_llm_params,
            reduce_llm_params=reduce_llm_params,
            allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
            json_mode=True,  # set this to False if your LLM model does not support JSON mode.
            context_builder_params=context_builder_params,
            concurrent_coroutines=1,
            response_type=response_type,  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )

        result = await search_engine.asearch(query)
        return result.response

def prepare_local_search(input_dir, community_level=2, temperature=0.5):
    LANCEDB_URI = f"{input_dir}/lancedb"

    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    RELATIONSHIP_TABLE = "create_final_relationships"
    COVARIATE_TABLE = "create_final_covariates"
    TEXT_UNIT_TABLE = "create_final_text_units"

    entity_df = pd.read_parquet(join(input_dir, f"{ENTITY_TABLE}.parquet"))
    entity_embedding_df = pd.read_parquet(join(input_dir, f"{ENTITY_EMBEDDING_TABLE}.parquet"))

    entities = read_indexer_entities(entity_df, entity_embedding_df, community_level)

    # load description embeddings to an in-memory lancedb vectorstore
    # to connect to a remote db, specify url and port values.
    description_embedding_store = LanceDBVectorStore(
        collection_name="entity_description_embeddings",
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    entity_description_embeddings = store_entity_semantic_embeddings(
        entities=entities, vectorstore=description_embedding_store
    )

    relationship_df = pd.read_parquet(join(input_dir, f"{RELATIONSHIP_TABLE}.parquet"))
    relationships = read_indexer_relationships(relationship_df)

    # covariate_df = pd.read_parquet(join(input_dir, f"{COVARIATE_TABLE}.parquet"))
    # claims = read_indexer_covariates(covariate_df)
    # covariates = {"claims": claims}

    report_df = pd.read_parquet(join(input_dir, f"{COMMUNITY_REPORT_TABLE}.parquet"))
    reports = read_indexer_reports(report_df, entity_df, community_level)

    text_unit_df = pd.read_parquet(join(input_dir, f"{TEXT_UNIT_TABLE}.parquet"))
    text_units = read_indexer_text_units(text_unit_df)

    llm_config = get_llm_config()
    embedding_config = get_embedding_config()

    llm = ChatOpenAI(
        api_key=llm_config["api_key"],
        api_base=llm_config["api_base"],
        model=llm_config["model"],
        api_type=llm_config["api_type"],
        max_retries=llm_config["max_retries"],
    )

    token_encoder = tiktoken.get_encoding("cl100k_base")

    text_embedder = OpenAIEmbedding(
        api_key=embedding_config["api_key"],
        api_base=embedding_config["api_base"],
        api_type=embedding_config["api_type"],
        model=embedding_config["model"],
        deployment_name=embedding_config["deployment_name"],
        max_retries=embedding_config["max_retries"],
    )

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
        "max_tokens": 5000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    }

    llm_params = {
        "max_tokens": 1500,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
        "temperature": temperature,
    }

    return llm, context_builder, token_encoder, llm_params, local_context_params

async def local_search(query, input_dir, community_level=2, temperature=0.5, response_type="Multiple Paragraphs"):
    (
        llm, 
        context_builder, 
        token_encoder, 
        llm_params, 
        local_context_params
    ) = prepare_local_search(input_dir, community_level, temperature)

    search_engine = LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type=response_type,  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    result = await search_engine.asearch(query)
    return result.response

async def local_question_generate(question_history, input_dir, community_level=2, temperature=0.5):
    (
        llm, 
        context_builder, 
        token_encoder, 
        llm_params, 
        local_context_params
    ) = prepare_local_search(input_dir, community_level, temperature)

    question_generator = LocalQuestionGen(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
    )

    # Ensure question_history is a list of strings (not nested lists)
    # If empty, provide a default starting question
    if not question_history:
        question_history = ["What are the main topics in this dataset?"]
    
    # Flatten any nested lists and ensure all items are strings
    flat_history = []
    for item in question_history:
        if isinstance(item, list):
            flat_history.extend([str(x) for x in item])
        else:
            flat_history.append(str(item))
    
    result = await question_generator.agenerate(
        question_history=flat_history, context_data=None, question_count=5
    )
    return result.response


async def chat_graphrag_once(
        query, 
        history,
        selected_folder,
        query_type,
        temperature,
        preset,
        show_graph=True
    ):
    input_dir = resolve_output_path(selected_folder)
    if not input_dir:
        empty_graph = "<div style='padding: 20px; color: #888;'>未找到可用索引目录，请先执行索引。</div>"
        empty_expl = "<div style='padding: 20px; color: #888;'>暂无可解释结果。</div>"
        return "未找到可用索引目录，请先完成索引。", empty_graph, empty_expl, {}

    community_level = PRESET_MAPPING[preset]["community_level"]
    response_type = PRESET_MAPPING[preset]["response_type"]

    response = None
    query_entities = []
    
    if query == "/generate":
        question_history = [msg["content"] for msg in history if msg.get("role") == "user"]
        response = await local_question_generate(
            question_history, input_dir, community_level, temperature
        )
    elif query_type == "reasoning" and REASONING_SEARCH_AVAILABLE:
        # PageIndex-inspired reasoning-based search (highest accuracy)
        result = await enhanced_search(
            query=query,
            input_dir=input_dir,
            query_type="auto",
            community_level=community_level,
            temperature=temperature,
            response_type=response_type
        )
        response = result["response"]
        # Add confidence and verification info
        confidence = result.get("confidence", 0)
        verified = result.get("verified", False)
        if confidence > 0:
            response += f"\n\n---\n📊 **置信度：** {confidence:.0%} | ✅ **已校验：** {verified}"
        query_entities = [word for word in query.split() if len(word) > 3]
    elif query_type == "hybrid" and REASONING_SEARCH_AVAILABLE:
        # Hybrid reasoning search (combined local + global)
        result = await hybrid_reasoning_search(
            query=query,
            input_dir=input_dir,
            community_level=community_level,
            temperature=temperature,
            response_type=response_type
        )
        response = result["response"]
        confidence = result.get("confidence", 0)
        if confidence > 0:
            response += f"\n\n---\n📊 **置信度：** {confidence:.0%} | 🔄 **策略：** 混合检索"
        query_entities = [word for word in query.split() if len(word) > 3]
    elif query_type == "global":
        response = await global_search(
            query, input_dir, community_level, temperature, response_type
        )
        # Extract key terms from query for graph visualization
        query_entities = [word for word in query.split() if len(word) > 3]
    elif query_type == "local":
        response = await local_search(
            query, input_dir, community_level, temperature, response_type
        )
        query_entities = [word for word in query.split() if len(word) > 3]
    else:
        response = "抱歉，我暂时无法执行检索。"
    
    explanation_data = {}
    explanation_html = "<div style='padding: 20px; color: #888;'>暂无可解释结果。</div>"

    if query != "/generate":
        try:
            explanation_data = build_answer_explanation(
                input_dir=input_dir,
                query=query,
                response=response,
                max_nodes=35,
                max_paths=3,
                max_evidence=6,
            )
            explanation_html = render_explanation_html(explanation_data)
        except Exception as e:
            explanation_html = f"<div style='padding: 20px; color: #888;'>解释结果暂不可用：{str(e)}</div>"
            explanation_data = {
                "query": query,
                "answer": response,
                "limitations": [f"解释结果生成失败：{str(e)}"],
            }

    # Generate graph visualization if enabled
    graph_html = ""
    if show_graph and query != "/generate":
        try:
            if explanation_data and explanation_data.get("subgraph", {}).get("nodes"):
                graph_html = create_explanation_graph_html(explanation_data)
            else:
                # Fallback to generic query graph
                entity_df, _, _ = load_graph_data(input_dir)
                response_entities = extract_entities_from_response(response, entity_df)
                all_entities = list(set(query_entities + response_entities))
                graph_html = create_graph_html_for_query(
                    input_dir,
                    query_entities=all_entities[:10],
                    max_nodes=40
                )
        except Exception as e:
            graph_html = f"<div style='padding: 20px; color: #888;'>图谱可视化暂不可用：{str(e)}</div>"
    
    return response, graph_html, explanation_html, explanation_data


def chat_graphrag_stream(
        query,
        history,
        selected_folder,
        query_type,
        temperature,
        preset,
        show_graph=True
    ):
    history = history or []
    if not query or not str(query).strip():
        yield "", history, gr.update(), gr.update(), gr.update()
        return

    query = str(query).strip()
    thinking_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "正在检索，请稍候…"}
    ]
    yield (
        "",
        thinking_history,
        "<div style='padding: 20px; color: #888;'>正在生成回答与图谱，请稍候…</div>",
        "<div style='padding: 20px; color: #888;'>正在整理推理路径与溯源证据…</div>",
        {}
    )

    response, graph_html, explanation_html, explanation_data = asyncio.run(
        chat_graphrag_once(
            query,
            history,
            selected_folder,
            query_type,
            temperature,
            preset,
            show_graph
        )
    )

    final_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": ""}
    ]

    step = 24
    for idx in range(step, len(response) + step, step):
        final_history[-1]["content"] = response[:idx]
        yield (
            "",
            final_history,
            graph_html if idx >= len(response) else "<div style='padding: 20px; color: #888;'>图谱生成完成后显示…</div>",
            explanation_html if idx >= len(response) else "<div style='padding: 20px; color: #888;'>推理路径与证据整理完成后显示…</div>",
            explanation_data if idx >= len(response) else {}
        )
        time.sleep(0.015)

def resolve_output_path(selected_folder=None):
    """Resolve GraphRAG output path across multiple layout variants."""
    output_dir = join(script_dir, "output")
    if not os.path.exists(output_dir):
        return None

    direct_parquet = join(output_dir, "create_final_nodes.parquet")
    direct_artifacts = join(output_dir, "artifacts", "create_final_nodes.parquet")
    if os.path.exists(direct_parquet):
        return output_dir
    if os.path.exists(direct_artifacts):
        return join(output_dir, "artifacts")

    if selected_folder and selected_folder not in {"", "未找到输出目录"}:
        candidate_artifacts = join(output_dir, selected_folder, "artifacts", "create_final_nodes.parquet")
        candidate_direct = join(output_dir, selected_folder, "create_final_nodes.parquet")
        if os.path.exists(candidate_artifacts):
            return join(output_dir, selected_folder, "artifacts")
        if os.path.exists(candidate_direct):
            return join(output_dir, selected_folder)

    for folder in sorted(os.listdir(output_dir), reverse=True):
        folder_path = join(output_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if os.path.exists(join(folder_path, "artifacts", "create_final_nodes.parquet")):
            return join(folder_path, "artifacts")
        if os.path.exists(join(folder_path, "create_final_nodes.parquet")):
            return folder_path

    return None

def list_output_folders():
    """List available output folders for GraphRAG queries.
    
    Supports direct parquet output, artifacts output, and timestamped folders.
    """
    output_dir = join(script_dir, "output")
    if not os.path.exists(output_dir):
        return []
    
    if os.path.exists(join(output_dir, "create_final_nodes.parquet")):
        return ["output"]

    if os.path.exists(join(output_dir, "artifacts", "create_final_nodes.parquet")):
        return ["output"]

    folders = []
    for f in os.listdir(output_dir):
        folder_path = join(output_dir, f)
        if not os.path.isdir(folder_path):
            continue
        if os.path.exists(join(folder_path, "artifacts", "create_final_nodes.parquet")) or os.path.exists(join(folder_path, "create_final_nodes.parquet")):
            folders.append(f)
    return sorted(folders, reverse=True)

def create_gradio_interface():
    # 中文示例提示词
    SAMPLE_PROMPTS = [
        "政务服务事项办理中，材料预审与正式受理有什么区别？",
        "网格化治理中，街道、社区和网格员如何协同处置问题？",
        "国有企业合规管理通常包括哪些关键环节？",
        "突发事件信息报送为什么强调时效性和分级响应？",
        "纪检监督中的廉政风险排查通常怎么开展？",
        "/generate",
    ]
    
    custom_css = """
    .gradio-container,
    .contain {
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    #component-0 {
        height: 100%;
    }

    #main-container {
        display: flex;
        gap: 18px;
        align-items: flex-start;
        flex-wrap: nowrap;
        width: 100%;
    }

    #left-column,
    #middle-column,
    #right-column,
    #query-row,
    #query-input,
    #chatbot,
    .gr-block,
    .gr-group,
    .gr-form,
    .gr-box,
    .gradio-container .gr-row > * {
        min-width: 0 !important;
        box-sizing: border-box;
    }

    #left-column {
        flex: 0 0 240px;
        min-width: 210px;
        max-width: 240px;
        width: 240px;
        position: sticky;
        top: 12px;
        align-self: flex-start;
        height: fit-content;
    }

    #middle-column {
        min-width: 0;
        flex: 1 1 0;
    }

    #right-column {
        flex: 0 0 460px;
        min-width: 400px;
        max-width: 460px;
        width: 460px;
        position: sticky;
        top: 12px;
        align-self: flex-start;
    }

    #chat-shell,
    #workspace-shell {
        border: 1px solid #dbe3f0;
        border-radius: 16px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }

    #chat-shell {
        padding: 14px;
        margin-bottom: 16px;
    }

    #workspace-shell {
        padding: 10px 12px 14px 12px;
        max-height: calc(100vh - 24px);
        overflow: auto;
    }

    #workspace-panels {
        gap: 14px;
        align-items: stretch;
    }

    #graph-panel,
    #explain-panel,
    #ingest-shell {
        border: 1px solid #dbe3f0;
        border-radius: 14px;
        background: #ffffff;
        padding: 12px;
    }

    #graph-panel {
        min-height: 640px;
    }

    #explain-panel {
        min-height: 640px;
    }

    #chatbot {
        flex-grow: 1;
        overflow: auto;
        min-height: 700px;
    }

    #query-panel,
    #query-input,
    #query-input textarea,
    #query-input input,
    .gradio-container textarea,
    .gradio-container input {
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
    }

    #query-panel {
        width: 100%;
        margin-top: 0;
        margin-bottom: 12px;
    }

    #query-actions {
        align-items: stretch;
        gap: 10px;
    }

    #query-submit {
        min-width: 148px;
        border: none !important;
        border-radius: 14px !important;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%) !important;
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.12);
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }

    #query-submit:hover {
        transform: translateY(-1px);
        filter: brightness(1.04);
        box-shadow: 0 18px 34px rgba(15, 23, 42, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.16);
    }

    #query-submit:active {
        transform: translateY(0);
        filter: brightness(0.98);
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.16), inset 0 2px 6px rgba(0, 0, 0, 0.16);
    }

    #query-submit:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(125, 211, 252, 0.45), 0 16px 30px rgba(15, 23, 42, 0.18);
    }

    #left-column .gradio-dropdown,
    #left-column .gradio-radio,
    #left-column .gradio-checkbox,
    #left-column .gradio-slider {
        margin-bottom: 8px;
    }

    .sample-prompts {
        margin-top: 10px;
    }

    .sample-prompts button {
        margin: 2px;
        font-size: 12px;
        white-space: normal !important;
        word-break: break-word;
    }

    .prose table,
    .gradio-container table {
        display: block;
        width: 100%;
        overflow-x: auto;
    }

    iframe,
    #graph-display {
        max-width: 100%;
    }

    #graph-display iframe {
        min-height: 560px;
    }

    #explanation-display {
        min-height: 560px;
    }

    @media (max-width: 900px) {
        #main-container {
            flex-direction: column;
        }

        #left-column,
        #middle-column,
        #right-column {
            width: 100%;
            min-width: 0;
            flex: 1 1 auto;
        }

        #left-column {
            position: static;
            max-width: none;
        }

        #right-column {
            position: static;
            max-width: none;
        }

        #query-panel {
            width: 100%;
        }

        #chatbot {
            min-height: 520px;
        }

        #workspace-panels {
            flex-direction: column;
        }
    }
    """
    with gr.Blocks(title="VeritasGraph 中文演示") as demo:
        gr.Markdown("""
        # 🔍 VeritasGraph 中文版演示
        **企业级知识图谱 RAG，支持可验证溯源**
        
        📊 **当前示例主题：** 政务服务、基层治理、国企合规、应急报送、纪检监督、党建组织运行

        你可以直接点击下方中文示例问题，或输入自己的问题开始检索。
        """)
        
        with gr.Row(elem_id="main-container"):
            with gr.Column(scale=1, elem_id="left-column"):
                output_folders = list_output_folders()
                output_folder = output_folders[0] if output_folders else "未找到输出目录"
                gr.Markdown("### 检索配置")
                selected_folder = gr.Dropdown(
                    label="选择输出目录",
                    choices=output_folders if output_folders else ["未找到输出目录"],
                    value=output_folder,
                    interactive=True,
                    allow_custom_value=True
                )

                query_type = gr.Radio(
                    QUERY_TYPE_CHOICES,
                    label="检索方式",
                    value="reasoning" if REASONING_SEARCH_AVAILABLE else "global",
                    info="🎯 推理检索：更适合复杂问题 ｜ 🔄 混合检索：综合局部与全局 ｜ 🌐 全局检索：适合总体总结 ｜ 📍 局部检索：适合实体细节"
                )

                temperature = gr.Slider(
                    label="温度",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=float(0.5)
                )

                preset = gr.Radio(
                    ["默认", "详细", "快速", "要点", "全面", "概览", "聚焦"],
                    label="回答风格",
                    value="默认",
                    info="控制回答的详细程度与呈现方式"
                )
                
                # Graph visualization toggle
                show_graph = gr.Checkbox(
                    label="🔗 显示图谱可视化",
                    value=True,
                    info="每次提问后展示相关知识图谱"
                )

            with gr.Column(scale=4, elem_id="middle-column"):
                with gr.Column(elem_id="chat-shell"):
                    with gr.Column(elem_id="query-panel"):
                        with gr.Row(elem_id="query-actions"):
                            query = gr.Textbox(
                                label="输入问题",
                                placeholder="请在这里输入问题，或点击下方政务治理示例问题……",
                                elem_id="query-input",
                                lines=3,
                                max_lines=5,
                                scale=6
                            )
                            query_btn = gr.Button("开始检索", variant="primary", elem_id="query-submit", scale=1)

                    gr.Markdown("**📝 中文示例问题（点击即可使用）：**", elem_classes=["sample-prompts"])
                    with gr.Row():
                        example_btns = []
                        example_prompts = [
                            ("🏛️ 审批流程", "政务服务事项办理中，材料预审与正式受理有什么区别？"),
                            ("🏘️ 基层协同", "网格化治理中，街道、社区和网格员如何协同处置问题？"),
                            ("🏢 国企合规", "国有企业合规管理通常包括哪些关键环节？"),
                            ("💡 生成追问", "/generate"),
                        ]

                    with gr.Row():
                        for label, prompt in example_prompts[:2]:
                            btn = gr.Button(label, size="sm", variant="secondary")
                            example_btns.append((btn, prompt))
                    with gr.Row():
                        for label, prompt in example_prompts[2:]:
                            btn = gr.Button(label, size="sm", variant="secondary")
                            example_btns.append((btn, prompt))

                    chatbot = gr.Chatbot(
                        label="对话记录",
                        elem_id="chatbot",
                        height=720,
                        value=[{"role": "assistant", "content": """👋 欢迎使用 **VeritasGraph 中文版**！

我可以帮助你探索当前知识图谱中的政务服务、基层治理、国企合规、应急处置、纪检监督和党建组织运行等中文示例内容。

**📊 你可以这样问：**
- “政务服务事项办理中，预审和正式受理有什么区别？”
- “基层网格治理的问题上报链路是什么？”
- “国有企业合规管理如何形成闭环？”
- “突发事件信息报送为什么要分级分层？”

**使用建议：**
- 🎯 **推理检索**：适合复杂、多跳问题
- 🔄 **混合检索**：结合局部与全局信息
- 🌐 **全局检索**：适合做总体总结
- 📍 **局部检索**：适合查找具体实体
- 输入 `/generate` 可自动生成后续问题
- 勾选“显示图谱可视化”可查看关联图谱

现在就开始提问吧！"""}]
                    )

            with gr.Column(scale=3, elem_id="right-column"):
                with gr.Column(elem_id="workspace-shell"):
                    with gr.Column(elem_id="workspace-panels"):
                        with gr.Column(elem_id="graph-panel"):
                            gr.Markdown("""
                            ### 交互式知识图谱
                            每次提问后，图谱会自动更新，展示回答中涉及的实体、证据边与关键路径。
                            
                            **图例说明：** 
                            - 🔴 **红色节点** = 问题实体
                            - 🟢 **绿色节点** = 回答实体
                            - 🟠 **橙色节点/边** = 关键推理路径
                            - 🔵 **蓝色边** = 证据边
                            - **节点大小** = 重要程度（连接数）
                            - **悬停** 查看详情 ｜ **拖拽** 调整布局 ｜ **滚轮** 缩放
                            """)
                            explore_btn = gr.Button("🔍 查看完整图谱", variant="secondary", size="sm")
                            graph_display = gr.HTML(
                                value="<div style='padding: 40px; text-align: center; color: #888; background: #0a0a0a; border-radius: 8px; min-height: 500px;'><h3>🔗 知识图谱</h3><p>请先发起一次提问，以查看相关子图可视化</p></div>",
                                elem_id="graph-display"
                            )

                        with gr.Column(elem_id="explain-panel"):
                            gr.Markdown("""
                            ### 回答解释包
                            这里会展示：
                            - **关键推理路径**：答案相关实体如何通过图谱关系连起来
                            - **支撑证据**：对应原始文本片段
                            - **结构化JSON**：便于复用、导出、接前端
                            """)
                            explanation_display = gr.HTML(
                                value="<div style='padding: 20px; color: #888;'>请先发起一次提问，以查看关键路径与溯源证据。</div>",
                                elem_id="explanation-display"
                            )
                            explanation_json = gr.JSON(
                                label="结构化解释结果",
                                value={}
                            )

                    with gr.Accordion("⚡ 即时入库", open=False, elem_id="ingest-shell"):
                            deps = check_dependencies()
                            dep_status = []
                            if deps['youtube']:
                                dep_status.append("✅ YouTube")
                            else:
                                dep_status.append("❌ YouTube（安装：pip install youtube-transcript-api yt-dlp）")
                            if deps['web']:
                                dep_status.append("✅ 网页文章")
                            else:
                                dep_status.append("❌ 网页文章（安装：pip install trafilatura）")
                            if deps['pdf']:
                                dep_status.append("✅ PDF 溯源")
                            else:
                                dep_status.append("❌ PDF 溯源（安装：pip install pdfplumber）")

                            gr.Markdown(f"""
                            ### ⚡ 即时知识入库
                            **粘贴 YouTube 链接或网页文章链接**，即可快速加入知识图谱。

                            **支持来源：**
                            - 📺 **YouTube 视频**：自动提取字幕（含自动字幕）
                            - 📰 **网页文章**：提取博客、新闻、文档正文内容

                            **依赖状态：** {' | '.join(dep_status)}
                            """)

                            with gr.Row():
                                ingest_url_input = gr.Textbox(
                                    label="🔗 来源链接",
                                    placeholder="请粘贴 YouTube 或文章链接……（例如：https://youtube.com/watch?v=...）",
                                    scale=4
                                )
                                ingest_btn = gr.Button("⚡ 链接入库", variant="primary", scale=1)

                            gr.Markdown("---")
                            gr.Markdown("""
                            ### 📕 上传 PDF
                            **上传 PDF 后会同时生成文本索引文件和 source map sidecar。**
                            若环境已安装 `pdfplumber`，解释面板会尽量展示页码、段落和 bbox 坐标。
                            """)
                            with gr.Row():
                                pdf_input = gr.File(
                                    label="上传 PDF 文件",
                                    file_types=[".pdf"],
                                    type="filepath",
                                    scale=4
                                )
                                pdf_ingest_btn = gr.Button("📕 PDF 入库", variant="primary", scale=1)

                            gr.Markdown("---")
                            gr.Markdown("""
                            ### 📝 粘贴文本内容
                            **可直接复制粘贴文本**，来源可以是文件、文档、PDF 提取结果等。
                            """)

                            with gr.Row():
                                text_title_input = gr.Textbox(
                                    label="📌 标题",
                                    placeholder="请输入这段内容的标题……",
                                    scale=2
                                )

                            text_content_input = gr.Textbox(
                                label="📄 文本内容",
                                placeholder="请在这里粘贴文本内容……（至少 50 个字符）",
                                lines=8,
                                max_lines=20
                            )

                            with gr.Row():
                                text_ingest_btn = gr.Button("📝 添加到知识库", variant="primary")
                                text_clear_btn = gr.Button("🗑️ 清空", variant="secondary")

                            ingest_status = gr.Markdown(
                                value="*请粘贴链接或文本内容，然后点击对应按钮加入知识库。*"
                            )

                            with gr.Row():
                                index_btn = gr.Button("📊 全量建索引", variant="primary")
                                update_index_btn = gr.Button("🔄 增量更新", variant="secondary")
                                check_status_btn = gr.Button("📋 查看状态", variant="secondary")
                                refresh_files_btn = gr.Button("🔃 刷新文件", variant="secondary")

                            gr.Markdown("---")
                            gr.Markdown("### 📁 输入文件")

                            def format_file_list():
                                files = list_input_files()
                                if not files:
                                    return "*输入目录里还没有文件，请先在上方添加内容。*"
                                rows = ["| 文件名 | 大小 | 修改时间 |\n|------|------|----------|"]
                                for f in files[:20]:
                                    size_kb = f['size'] / 1024
                                    rows.append(f"| {f['name']} | {size_kb:.1f} KB | {f['modified']} |")
                                if len(files) > 20:
                                    rows.append(f"\n*……另外还有 {len(files) - 20} 个文件*")
                                return '\n'.join(rows)

                            file_list_display = gr.Markdown(value=format_file_list())

                            with gr.Accordion("📄 文件预览", open=False):
                                file_select = gr.Dropdown(
                                    label="选择要预览的文件",
                                    choices=[f['name'] for f in list_input_files()],
                                    interactive=True
                                )
                                file_preview = gr.Textbox(
                                    label="内容预览",
                                    lines=10,
                                    max_lines=20,
                                    interactive=False
                                )
                                delete_file_btn = gr.Button("🗑️ 删除选中文件", variant="stop")

                            def handle_ingest(url):
                                if not url or not url.strip():
                                    return "⚠️ 请先输入链接。"
                                success, message, filepath = ingest_url(url)
                                return message

                            ingest_btn.click(
                                fn=handle_ingest,
                                inputs=[ingest_url_input],
                                outputs=[ingest_status]
                            ).then(
                                fn=format_file_list,
                                outputs=[file_list_display]
                            ).then(
                                fn=lambda: gr.update(choices=[f['name'] for f in list_input_files()]),
                                outputs=[file_select]
                            )

                            def handle_pdf_ingest(pdf_path):
                                if not pdf_path:
                                    return "⚠️ 请先上传 PDF 文件。"
                                success, message, filepath = ingest_pdf_file(pdf_path)
                                return message

                            pdf_ingest_btn.click(
                                fn=handle_pdf_ingest,
                                inputs=[pdf_input],
                                outputs=[ingest_status]
                            ).then(
                                fn=format_file_list,
                                outputs=[file_list_display]
                            ).then(
                                fn=lambda: gr.update(choices=[f['name'] for f in list_input_files()]),
                                outputs=[file_select]
                            ).then(
                                fn=lambda: None,
                                outputs=[pdf_input]
                            )

                            def handle_text_ingest(title, content):
                                success, message, filepath = ingest_text_content(title, content)
                                return message

                            text_ingest_btn.click(
                                fn=handle_text_ingest,
                                inputs=[text_title_input, text_content_input],
                                outputs=[ingest_status]
                            ).then(
                                fn=format_file_list,
                                outputs=[file_list_display]
                            ).then(
                                fn=lambda: gr.update(choices=[f['name'] for f in list_input_files()]),
                                outputs=[file_select]
                            ).then(
                                fn=lambda: ("", ""),
                                outputs=[text_title_input, text_content_input]
                            )

                            text_clear_btn.click(
                                fn=lambda: ("", ""),
                                outputs=[text_title_input, text_content_input]
                            )

                            def handle_full_index():
                                for progress_msg in trigger_graphrag_index_with_progress(update_mode=False):
                                    yield progress_msg

                            index_btn.click(
                                fn=handle_full_index,
                                outputs=[ingest_status]
                            )

                            def handle_update_index():
                                for progress_msg in trigger_graphrag_index_with_progress(update_mode=True):
                                    yield progress_msg

                            update_index_btn.click(
                                fn=handle_update_index,
                                outputs=[ingest_status]
                            )

                            def handle_check_status():
                                status, is_complete = get_indexing_status()
                                return status

                            check_status_btn.click(
                                fn=handle_check_status,
                                outputs=[ingest_status]
                            )

                            refresh_files_btn.click(
                                fn=format_file_list,
                                outputs=[file_list_display]
                            ).then(
                                fn=lambda: gr.update(choices=[f['name'] for f in list_input_files()]),
                                outputs=[file_select]
                            )

                            def handle_preview(filename):
                                if filename:
                                    return get_file_preview(filename, max_chars=2000)
                                return ""

                            file_select.change(
                                fn=handle_preview,
                                inputs=[file_select],
                                outputs=[file_preview]
                            )

                            def handle_delete(filename):
                                if filename:
                                    success, msg = delete_input_file(filename)
                                    return msg, format_file_list(), gr.update(choices=[f['name'] for f in list_input_files()], value=None), ""
                                return "⚠️ 尚未选择文件", format_file_list(), gr.update(choices=[f['name'] for f in list_input_files()]), ""

                            delete_file_btn.click(
                                fn=handle_delete,
                                inputs=[file_select],
                                outputs=[ingest_status, file_list_display, file_select, file_preview]
                            )
                
        # Connect example buttons to fill in the query
        for btn, prompt in example_btns:
            btn.click(lambda p=prompt: p, outputs=[query])

        # Query submission with graph visualization
        query.submit(
            fn=chat_graphrag_stream, 
            inputs=[
                query, 
                chatbot,
                selected_folder,
                query_type,
                temperature,
                preset,
                show_graph
            ], 
            outputs=[query, chatbot, graph_display, explanation_display, explanation_json]
        )
        query_btn.click(
            fn=chat_graphrag_stream, 
            inputs=[
                query, 
                chatbot,
                selected_folder,
                query_type,
                temperature,
                preset,
                show_graph
            ], 
            outputs=[query, chatbot, graph_display, explanation_display, explanation_json]
        )
        
        # Standalone graph explorer function
        def explore_full_graph(selected_folder, max_nodes=50):
            """显示完整知识图谱（按连接度优先）。"""
            input_dir = resolve_output_path(selected_folder)
            if not input_dir:
                return "<div style='padding: 20px; color: #888;'>未找到可用索引目录，请先执行索引。</div>"
            return create_graph_html_for_query(input_dir, query_entities=[], max_nodes=max_nodes)
        
        explore_btn.click(
            fn=explore_full_graph,
            inputs=[selected_folder],
            outputs=[graph_display]
        )

    demo.veritas_launch_css = custom_css
    return demo.queue()


def _extend_no_proxy(host: str) -> None:
    no_proxy_hosts = {"127.0.0.1", "localhost", "::1"}
    if host:
        no_proxy_hosts.add(host)

    for env_name in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(env_name, "")
        merged = {item.strip() for item in existing.split(",") if item.strip()}
        merged.update(no_proxy_hosts)
        os.environ[env_name] = ",".join(sorted(merged))


def _is_local_startup_probe(url: str) -> bool:
    parsed = urlparse(str(url))
    return (
        parsed.path.endswith("/startup-events")
        and parsed.hostname in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}
    )


def launch_gradio_app(demo: gr.Blocks, host: str, port: int, share: bool) -> tuple:
    _extend_no_proxy(host)

    original_httpx_get = httpx.get

    def patched_httpx_get(url, *args, **kwargs):
        if _is_local_startup_probe(url):
            verify = kwargs.pop("verify", True)
            with httpx.Client(trust_env=False, verify=verify) as client:
                return client.get(url, *args, **kwargs)
        return original_httpx_get(url, *args, **kwargs)

    httpx.get = patched_httpx_get
    try:
        return demo.launch(
            server_port=port,
            server_name=host,
            share=share,
            allowed_paths=[GRAPH_CACHE_DIR],
            theme=LAUNCH_THEME,
            css=getattr(demo, "veritas_launch_css", None),
        )
    finally:
        httpx.get = original_httpx_get


demo = create_gradio_interface()
app = demo.app

# Path to graph cache directory for file serving
GRAPH_CACHE_DIR = os.path.join(os.path.dirname(__file__), "graph_cache")
os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VeritasGraph 中文演示")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (use 0.0.0.0 for external access)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 VeritasGraph 中文演示服务")
    print("="*60)
    if args.share:
        print("📡 正在创建可分享的公网链接……")
    print(f"🌐 本地地址：http://{args.host}:{args.port}")
    print("="*60 + "\n")
    
    launch_gradio_app(
        demo=demo,
        host=args.host,
        port=args.port,
        share=args.share,
    )
