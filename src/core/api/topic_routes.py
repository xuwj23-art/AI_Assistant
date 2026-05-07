"""
主题管理 API 路由 (Stage 3 升级版)

新增端点:
  GET /api/topics/sunburst   — Plotly Sunburst 图数据
  GET /api/topics/trends     — 研究趋势时间轴数据
  GET /api/topics/{id}/papers — 带分页/排序的论文列表（升级）
"""
from fastapi import APIRouter, HTTPException, Query, Path, Depends
from typing import List, Optional

from .models import (
    TopicResponse,
    TopicListResponse,
    PaperListResponse,
    PaperListPagedResponse,
    SunburstResponse,
    TrendResponse,
    TrendTopicSeries,
    ErrorResponse,
)
from .topic_service import TopicService


# 创建路由器
router = APIRouter(
    prefix="/api",
    tags=["主题管理"]
)


# ========== 依赖注入 ==========

_topic_service: TopicService = None


def reset_topic_service():
    """重置主题服务（数据更新后调用，下次请求时重新初始化）"""
    global _topic_service
    _topic_service = None
    print("[TopicService] 已重置，下次请求时将重新加载数据")


def get_topic_service() -> TopicService:
    """
    获取主题服务实例（延迟初始化）

    使用 auto_fetch 查找最新数据文件，不再硬编码路径。
    如果没有数据，抛出异常提示用户先搜索。
    """
    global _topic_service

    if _topic_service is None:
        from pathlib import Path
        from ..auto_fetch import get_latest_data_paths

        paths = get_latest_data_paths()

        if paths is None or paths.get("topics_csv") is None:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="尚无主题数据。请先在搜索框中输入关键词进行搜索。"
            )

        data_path = paths["topics_csv"]
        if not data_path.exists():
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="尚无主题数据。请先在搜索框中输入关键词进行搜索。"
            )

        # 模型路径
        model_path = paths.get("model")
        if model_path and not model_path.exists():
            model_path = None

        _topic_service = TopicService(
            str(data_path),
            str(model_path) if model_path else None
        )
        print(f"[API] 初始化 TopicService: {data_path}")

    return _topic_service


# ========== 基础端点 ==========

@router.get(
    "/topics",
    response_model=TopicListResponse,
    summary="获取主题列表",
    description="获取所有主题及其统计信息"
)
def get_topics(
    service: TopicService = Depends(get_topic_service)
):
    """获取所有主题列表（按论文数量降序）"""
    try:
        topics = service.get_all_topics()
        return TopicListResponse(total=len(topics), topics=topics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取主题列表失败: {str(e)}")


@router.get(
    "/topics/stats",
    summary="获取主题统计信息",
    description="返回主题总数、论文总数等统计数据"
)
def get_topic_stats(
    service: TopicService = Depends(get_topic_service)
):
    """获取主题统计信息"""
    try:
        return service.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


# ========== Stage 3 新增端点 ==========

@router.get(
    "/topics/sunburst",
    response_model=SunburstResponse,
    summary="获取 Sunburst 图数据",
    description=(
        "返回 Plotly Sunburst 图所需的平铺格式数据。"
        "labels/parents/values/ids 四个数组可直接传入 plotly.graph_objects.Sunburst()。"
    )
)
def get_sunburst_data(
    service: TopicService = Depends(get_topic_service)
):
    """
    获取 Plotly Sunburst 旭日图数据

    返回格式（Plotly 直接使用）:
    {
      "labels":    ["All Topics", "NLP",        "Transformer", "CV",         ...],
      "parents":   ["",           "All Topics", "NLP",         "All Topics", ...],
      "values":    [0,            80,           40,            30,           ...],
      "ids":       ["root",       "topic_2",    "topic_7",     "topic_1",    ...],
      "topic_ids": [-1,           2,            7,             1,            ...]
    }
    """
    try:
        topics = service.get_all_topics()

        if not topics:
            return SunburstResponse(
                labels=["All Topics"],
                parents=[""],
                values=[0],
                ids=["root"],
                topic_ids=[-1]
            )

        labels = ["All Topics"]
        parents = [""]
        ids = ["root"]
        topic_ids = [-1]
        child_values = []

        for topic in topics:
            node_id = f"topic_{topic.topic_id}"
            labels.append(f"Topic {topic.topic_id}: {topic.topic_name}")
            parents.append("All Topics")
            child_values.append(topic.paper_count)
            ids.append(node_id)
            topic_ids.append(topic.topic_id)

        # 根节点 value = 子节点之和（branchvalues="total" 要求）
        values = [sum(child_values)] + child_values

        return SunburstResponse(
            labels=labels,
            parents=parents,
            values=values,
            ids=ids,
            topic_ids=topic_ids
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成 Sunburst 数据失败: {str(e)}")


@router.get(
    "/topics/trends",
    response_model=TrendResponse,
    summary="获取研究趋势数据",
    description=(
        "按年份统计各主题的论文数量，并返回增长最快的主题列表。"
        "可用于绘制折线图，展示各研究方向的热度变化。"
    )
)
def get_trends(
    top_n: int = Query(10, ge=1, le=30, description="返回主题数量上限"),
    service: TopicService = Depends(get_topic_service)
):
    """
    获取研究趋势时间轴数据

    返回格式:
    {
      "years": [2018, 2019, 2020, 2021, 2022, 2023],
      "topics": [
        {"name": "Transformer", "counts": [5, 12, 28, 45, 78, 120]},
        ...
      ],
      "trending": [
        {"name": "Transformer", "growth_rate": 2.5},
        ...
      ]
    }
    """
    try:
        from core.nlp.trend_analysis import (
            compute_trends_from_dataframe,
            get_trending_topics,
            trend_df_to_api_format,
        )

        df = service._load_data()
        trend_df, topic_names = compute_trends_from_dataframe(df)

        if trend_df.empty:
            return TrendResponse(years=[], topics=[], trending=[])

        # 只保留论文数最多的 top_n 个主题（避免图表过于拥挤）
        topic_totals = trend_df.sum().sort_values(ascending=False)
        top_topics = topic_totals.head(top_n).index.tolist()
        trend_df_filtered = trend_df[top_topics]

        # 转换为 API 格式
        api_data = trend_df_to_api_format(trend_df_filtered)

        # 计算增长最快的主题
        trending_raw = get_trending_topics(trend_df_filtered, top_n=5)
        trending = [
            {"name": name, "growth_rate": rate}
            for name, rate in trending_raw
        ]

        return TrendResponse(
            years=api_data["years"],
            topics=[
                TrendTopicSeries(name=t["name"], counts=t["counts"])
                for t in api_data["topics"]
            ],
            trending=trending
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取趋势数据失败: {str(e)}")


# ========== 主题详情端点 ==========

@router.get(
    "/topics/{topic_id}",
    response_model=TopicResponse,
    summary="获取主题详情",
    description="根据ID获取单个主题的详细信息",
    responses={
        404: {"model": ErrorResponse, "description": "主题不存在"}
    }
)
def get_topic(
    topic_id: int = Path(..., description="主题ID"),
    service: TopicService = Depends(get_topic_service)
):
    """获取主题详情"""
    topic = service.get_topic_by_id(topic_id)

    if topic is None:
        raise HTTPException(
            status_code=404,
            detail=f"Topic not found: {topic_id}"
        )

    return topic


@router.get(
    "/topics/{topic_id}/papers",
    response_model=PaperListPagedResponse,
    summary="获取主题下的论文（带分页/排序）",
    description=(
        "获取某个主题下的论文列表，支持分页和排序。\n\n"
        "sort_by 参数:\n"
        "- `date`: 按发表时间降序（最新优先）\n"
        "- `relevance`: 按主题相关性（默认，即原始顺序）"
    ),
    responses={
        404: {"model": ErrorResponse, "description": "主题不存在"}
    }
)
def get_topic_papers(
    topic_id: int = Path(..., description="主题ID"),
    page: int = Query(1, ge=1, description="页码（从1开始）"),
    page_size: int = Query(20, ge=5, le=100, description="每页论文数"),
    sort_by: str = Query("relevance", description="排序方式: relevance | date"),
    service: TopicService = Depends(get_topic_service)
):
    """
    获取主题下的论文（带分页和排序）

    参数:
        topic_id: 主题ID
        page: 页码（从1开始）
        page_size: 每页数量（5-100）
        sort_by: 排序方式（relevance=相关性 | date=时间降序）
    """
    # 先检查主题是否存在
    topic = service.get_topic_by_id(topic_id)
    if topic is None:
        raise HTTPException(
            status_code=404,
            detail=f"Topic not found: {topic_id}"
        )

    # 获取该主题全部论文（用于分页）
    all_papers = service.get_papers_by_topic(topic_id, limit=10000)

    # 排序
    if sort_by == "date":
        all_papers = sorted(
            all_papers,
            key=lambda p: p.published if p.published else __import__("datetime").datetime.min,
            reverse=True
        )
    # sort_by == "relevance" 保持原始顺序

    # 分页
    total = len(all_papers)
    start = (page - 1) * page_size
    end = start + page_size
    page_papers = all_papers[start:end]

    return PaperListPagedResponse(
        topic_name=topic.topic_name,
        total=total,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        papers=page_papers
    )


@router.get(
    "/topics/{topic_id}/similar",
    summary="获取相关主题推荐",
    description="基于主题关键词相似度，推荐与当前主题最相关的其他主题",
    responses={
        404: {"model": ErrorResponse, "description": "主题不存在"}
    }
)
def get_similar_topics(
    topic_id: int = Path(..., description="主题ID"),
    top_n: int = Query(5, ge=1, le=20, description="返回相关主题数量"),
    service: TopicService = Depends(get_topic_service)
):
    """
    获取相关主题推荐

    基于主题关键词的词汇重叠度计算相似性（轻量级方案，无需向量计算）。
    """
    # 检查目标主题是否存在
    target_topic = service.get_topic_by_id(topic_id)
    if target_topic is None:
        raise HTTPException(
            status_code=404,
            detail=f"Topic not found: {topic_id}"
        )

    try:
        all_topics = service.get_all_topics()

        # 目标主题的关键词集合
        target_words = {kw.word.lower() for kw in target_topic.keywords}

        similar = []
        for t in all_topics:
            if t.topic_id == topic_id:
                continue
            t_words = {kw.word.lower() for kw in t.keywords}
            # Jaccard 相似度
            if not target_words and not t_words:
                score = 0.0
            else:
                intersection = len(target_words & t_words)
                union = len(target_words | t_words)
                score = intersection / union if union > 0 else 0.0

            similar.append({
                "topic_id": t.topic_id,
                "topic_name": t.topic_name,
                "paper_count": t.paper_count,
                "similarity": round(score, 4)
            })

        # 按相似度降序
        similar.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "topic_id": topic_id,
            "topic_name": target_topic.topic_name,
            "similar_topics": similar[:top_n]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取相关主题失败: {str(e)}")
