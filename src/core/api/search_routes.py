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

    # 先检查内存状态
    if _current_state["query"]:
        return SearchStatusResponse(
            has_data=True,
            current_query=_current_state["query"],
            paper_count=_current_state["paper_count"],
            topic_count=_current_state["topic_count"],
        )

    # 再检查磁盘上是否有历史数据
    paths = get_latest_data_paths()
    if paths and paths.get("topics_csv"):
        return SearchStatusResponse(
            has_data=True,
            current_query="(历史数据)",
            paper_count=0,
            topic_count=0,
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
            detail="正在处理中，请等待当前任务完成"
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

        # 重置依赖此数据的服务（让它们重新加载）
        _reset_dependent_services()

        # 构建来源统计消息
        source_msg = ""
        if source_stats:
            parts = [f"{name}: {count}篇" for name, count in source_stats.items()]
            source_msg = f"（来源: {', '.join(parts)}）"

        return SearchResponse(
            query=request.query,
            paper_count=result["paper_count"],
            topic_count=result["topic_count"],
            csv_path=result["topics_csv_path"],
            topics_csv_path=result["topics_csv_path"],
            message=f"成功抓取 {result['paper_count']} 篇论文，发现 {result['topic_count']} 个研究主题{source_msg}",
            source_stats=source_stats,
        )

    except Exception as e:
        _current_state["is_processing"] = False
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


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
    # 重置 TopicService
    from .topic_routes import reset_topic_service
    reset_topic_service()

    # 重置 ChatService
    from .chat_service import reset_chat_service
    reset_chat_service()
