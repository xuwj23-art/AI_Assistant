"""
搜索 API 路由

用户输入关键词 → 在线抓取论文 → 生成向量 → 训练主题 → 返回结果
这是整个系统的入口端点。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

router = APIRouter(
    prefix="/api",
    tags=["搜索"]
)


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., min_length=1, max_length=200, description="搜索关键词")
    max_results: int = Field(50, ge=10, le=500, description="抓取论文数量")
    sources: Optional[List[str]] = Field(None, description="数据源列表，默认 ['arxiv']")


class SearchResponse(BaseModel):
    """搜索响应"""
    query: str
    paper_count: int
    topic_count: int
    csv_path: str
    topics_csv_path: str
    message: str
    source_stats: Optional[Dict[str, int]] = None
    routing: Optional[dict] = Field(None, description="智能来源路由结果（多源时返回）")


class SearchStatusResponse(BaseModel):
    """搜索状态响应"""
    has_data: bool
    current_query: Optional[str] = None
    paper_count: int = 0
    topic_count: int = 0


# 全局状态：当前已加载的查询
_current_state = {
    "query": None,
    "paper_count": 0,
    "topic_count": 0,
    "csv_path": None,
    "topics_csv_path": None,
    "embeddings_path": None,
    "model_path": None,
    "is_processing": False,
    "source_stats": None,
}


def get_current_state() -> dict:
    """获取当前数据状态（供其他模块使用）"""
    return _current_state


@router.get(
    "/search/status",
    response_model=SearchStatusResponse,
    summary="获取当前数据状态",
    description="检查是否已有抓取的数据"
)
def get_search_status():
    """获取当前搜索/数据状态"""
    from ..auto_fetch import get_latest_data_paths

    # 先检查内存状态，但同时验证文件是否真实存在
    if _current_state["query"]:
        csv_p = _current_state.get("topics_csv_path") or _current_state.get("csv_path")
        if csv_p:
            from pathlib import Path as _Path
            if not _Path(str(csv_p)).exists():
                # 文件已被删除，清空内存状态避免"幽灵数据"
                _current_state.update({
                    "query": None, "paper_count": 0, "topic_count": 0,
                    "csv_path": None, "topics_csv_path": None,
                    "embeddings_path": None, "model_path": None,
                    "source_stats": None,
                })
                from ..auto_fetch import set_active_paths
                set_active_paths(None)
                _reset_dependent_services()
                # 跳过，继续走磁盘扫描逻辑
            else:
                return SearchStatusResponse(
                    has_data=True,
                    current_query=_current_state["query"],
                    paper_count=_current_state["paper_count"],
                    topic_count=_current_state["topic_count"],
                )
        else:
            return SearchStatusResponse(
                has_data=True,
                current_query=_current_state["query"],
                paper_count=_current_state["paper_count"],
                topic_count=_current_state["topic_count"],
            )

    # 再检查磁盘上是否有历史数据
    paths = get_latest_data_paths()
    if paths and paths.get("topics_csv"):
        topics_csv = paths["topics_csv"]
        # 快速读取论文数/主题数，填充侧边栏信息
        paper_count = 0
        topic_count = 0
        query_name = "(historical data)"
        try:
            import pandas as _pd
            from pathlib import Path as _Path
            csv_p = _Path(str(topics_csv))
            df_mini = _pd.read_csv(csv_p, usecols=["topic_id"])
            paper_count = len(df_mini)
            topic_count = int(df_mini["topic_id"][df_mini["topic_id"] != -1].nunique())
            # 从文件名还原关键词：arxiv_<safe_query>_with_topics.csv
            stem = csv_p.stem  # "arxiv_BERT_with_topics"
            query_name = stem.removeprefix("arxiv_").removesuffix("_with_topics").replace("_", " ")
        except Exception:
            pass

        # 自动激活该路径（确保 TopicService/ChatService 重载时加载正确文件）
        from ..auto_fetch import set_active_paths
        set_active_paths({
            "csv": str(topics_csv),
            "topics_csv": str(topics_csv),
            "embeddings": str(paths["embeddings"]) if paths.get("embeddings") else None,
            "model": str(paths["model"]) if paths.get("model") else None,
        })
        _current_state.update({
            "query": query_name,
            "paper_count": paper_count,
            "topic_count": topic_count,
        })

        return SearchStatusResponse(
            has_data=True,
            current_query=query_name,
            paper_count=paper_count,
            topic_count=topic_count,
        )

    return SearchStatusResponse(has_data=False)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="搜索并抓取论文",
    description=(
        "用户输入关键词，系统在线抓取论文、生成向量、训练主题模型。\n"
        "这是整个系统的入口，首次使用时必须调用。"
    )
)
def search_papers(request: SearchRequest):
    """
    执行完整的搜索流水线：
    1. 在线抓取论文
    2. 生成向量
    3. 训练主题模型
    4. 返回结果摘要
    """
    global _current_state

    if _current_state["is_processing"]:
        raise HTTPException(
            status_code=409,
            detail="Processing in progress, please wait for the current task to finish"
        )

    _current_state["is_processing"] = True

    try:
        from ..auto_fetch import run_pipeline

        result = run_pipeline(
            query=request.query,
            max_results=request.max_results,
            sources=request.sources,
        )

        source_stats = result.get("source_stats", None)

        # 更新全局状态
        _current_state.update({
            "query": request.query,
            "paper_count": result["paper_count"],
            "topic_count": result["topic_count"],
            "csv_path": result["csv_path"],
            "topics_csv_path": result["topics_csv_path"],
            "embeddings_path": result["embeddings_path"],
            "model_path": result["model_path"],
            "is_processing": False,
            "source_stats": source_stats,
        })

        # 同步写入活跃路径覆盖，让 TopicService/ChatService 重新初始化时加载正确文件
        from ..auto_fetch import set_active_paths
        set_active_paths({
            "csv": result["topics_csv_path"],
            "topics_csv": result["topics_csv_path"],
            "embeddings": result["embeddings_path"],
            "model": result["model_path"],
            "hierarchy": result.get("hierarchy_path"),
        })

        # 重置依赖此数据的服务（让它们重新加载）
        _reset_dependent_services()

        # 构建来源统计消息
        source_msg = ""
        if source_stats:
            parts = [f"{name}: {count}" for name, count in source_stats.items()]
            source_msg = f" (sources: {', '.join(parts)})"

        return SearchResponse(
            query=request.query,
            paper_count=result["paper_count"],
            topic_count=result["topic_count"],
            csv_path=result["topics_csv_path"],
            topics_csv_path=result["topics_csv_path"],
            message=f"Fetched {result['paper_count']} papers, found {result['topic_count']} research topics{source_msg}",
            source_stats=source_stats,
            routing=result.get("routing"),
        )

    except Exception as e:
        _current_state["is_processing"] = False
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get(
    "/search/source-stats",
    summary="获取各数据源论文统计",
    description="返回当前数据中各来源的论文数量统计"
)
def get_source_stats():
    """获取各数据源的论文数量统计"""
    # 先检查内存中的统计
    if _current_state.get("source_stats"):
        return {"source_stats": _current_state["source_stats"]}

    # 从磁盘数据中统计
    from ..auto_fetch import get_latest_data_paths
    import pandas as pd

    paths = get_latest_data_paths()
    if paths and paths.get("topics_csv") and paths["topics_csv"].exists():
        try:
            df = pd.read_csv(paths["topics_csv"])
            if "source" in df.columns:
                stats = df["source"].value_counts().to_dict()
                return {"source_stats": stats}
        except Exception:
            pass

    return {"source_stats": {}}


def _reset_dependent_services():
    """重置依赖数据的服务，让它们在下次请求时重新加载"""
    from .topic_routes import reset_topic_service
    reset_topic_service()

    from .chat_service import reset_chat_service
    reset_chat_service()

    # 主题质心缓存（基于 csv/emb 的 mtime 也会自动失效，这里多一道保险）
    try:
        from ..nlp.topic_similarity import reset_centroid_cache
        reset_centroid_cache()
    except Exception:
        pass


# ========== 历史记录端点 ==========

class HistoryDataset(BaseModel):
    """单条历史数据集摘要"""
    query: str
    safe_query: str
    paper_count: int
    topic_count: int
    modified: float
    modified_str: str
    topics_csv: str
    embeddings: Optional[str]
    has_embeddings: bool


class HistorySwitchRequest(BaseModel):
    """切换历史数据集请求"""
    safe_query: str = Field(..., description="要切换到的数据集 safe_query（文件名标识）")


@router.get(
    "/history",
    summary="获取历史数据集列表",
    description="扫描本地已有的所有搜索历史，返回摘要（按时间倒序）",
)
def get_history():
    """返回所有历史数据集，供前端展示切换列表"""
    from ..auto_fetch import list_all_datasets
    datasets = list_all_datasets()
    return {"datasets": datasets, "total": len(datasets)}


@router.post(
    "/history/switch",
    summary="切换激活数据集",
    description="将系统切换到指定的历史数据集（CSV + 向量），无需重新搜索",
)
def switch_history(request: HistorySwitchRequest):
    """
    切换到历史数据集：
    1. 找到对应的 topics_csv 和 embeddings 文件
    2. 更新 _current_state（让各服务下次请求时自动重载）
    3. 重置 TopicService / ChatService
    """
    global _current_state

    from pathlib import Path as _Path
    from ..auto_fetch import list_all_datasets

    # 在列表中找到目标数据集
    all_ds = list_all_datasets()
    target = next((d for d in all_ds if d["safe_query"] == request.safe_query), None)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {request.safe_query} — please check the keyword spelling",
        )

    topics_csv = _Path(target["topics_csv"])
    if not topics_csv.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {topics_csv}",
        )

    # 更新全局状态（服务会在下次请求时重新加载这些路径）
    _current_state.update({
        "query": target["query"],
        "paper_count": target["paper_count"],
        "topic_count": target["topic_count"],
        "csv_path": target["topics_csv"],
        "topics_csv_path": target["topics_csv"],
        "embeddings_path": target["embeddings"],
        "model_path": None,     # 不加载历史模型，从 CSV 读关键词即可
        "is_processing": False,
        "source_stats": None,
    })

    # 写入活跃路径覆盖，确保 TopicService/ChatService 初始化时加载正确文件
    from ..auto_fetch import set_active_paths
    set_active_paths({
        "csv": target["topics_csv"],
        "topics_csv": target["topics_csv"],
        "embeddings": target["embeddings"],
        "model": None,
        "hierarchy": target.get("hierarchy"),
    })

    # 重置依赖服务，让它们重新按新路径初始化
    _reset_dependent_services()

    print(f"[History] Switched to: {target['query']} ({target['paper_count']} papers, {target['topic_count']} topics)")
    return {
        "switched_to": target["query"],
        "paper_count": target["paper_count"],
        "topic_count": target["topic_count"],
        "modified_str": target["modified_str"],
        "message": f"Switched to \"{target['query']}\" ({target['paper_count']} papers, {target['topic_count']} topics)",
    }
