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
                detail="No topic data yet. Please search for a keyword first."
            )

        data_path = paths["topics_csv"]
        if not data_path.exists():
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="No topic data yet. Please search for a keyword first."
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
        raise HTTPException(status_code=500, detail=f"Failed to retrieve topic list: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")


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
    获取 Plotly Sunburst 旭日图数据（多级层级版）

    优先策略:
      从 data/processed/arxiv_<query>_hierarchy.json 读取预计算的多级层级树
      （由 run_pipeline 在训练 BERTopic 后自动生成）。

    回退策略:
      如果层级 JSON 不存在（旧数据或计算失败），退化为 2 层平铺结构。

    返回格式（Plotly Sunburst 直接使用）:
      labels/parents/values/ids/topic_ids 平铺列表
      topic_ids: -1=根节点, -2=内部聚类节点, >=0=叶子主题
    """
    try:
        # ── 优先：读取预计算的层级 JSON ──────────────────────────
        import json as _json
        from ..auto_fetch import get_latest_data_paths

        hier_data = None
        paths = get_latest_data_paths()
        hier_path = paths.get("hierarchy") if paths else None
        if hier_path and hier_path.exists():
            try:
                with open(hier_path, "r", encoding="utf-8") as _f:
                    hier_data = _json.load(_f)
                # 基本校验：标签数量 + values 约束（parent >= sum of children）
                if not (hier_data.get("labels") and len(hier_data["labels"]) > 1):
                    hier_data = None
                else:
                    _ids_set = set(hier_data["ids"])
                    _parent_bad = any(
                        p != "" and p not in _ids_set
                        for p in hier_data["parents"]
                    )
                    from collections import defaultdict as _dd
                    _child_sum: dict = _dd(int)
                    for p, v in zip(hier_data["parents"], hier_data["values"]):
                        if p:
                            _child_sum[p] += v
                    _val_bad = any(
                        v < _child_sum.get(nid, 0)
                        for nid, v in zip(hier_data["ids"], hier_data["values"])
                    )
                    if _parent_bad or _val_bad:
                        print(f"[Sunburst] 层级 JSON 校验失败（parent_bad={_parent_bad}, val_bad={_val_bad}），尝试重算")
                        hier_data = None
                        if hier_path:
                            try:
                                hier_path.unlink()
                            except Exception:
                                pass
            except Exception as _e:
                print(f"[Sunburst] 层级 JSON 读取失败（{_e}），回退 2 层")
                hier_data = None

        # 若 JSON 缺失或失效，尝试从当前 BERTopic 模型重新计算
        if hier_data is None:
            try:
                _model = service._load_model()
                # 历史切换时 service.model_path=None；退而使用磁盘上保存的最新模型
                if _model is None:
                    from ..auto_fetch import MODELS_DIR
                    _default_model_path = MODELS_DIR / "bertopic_model"
                    if _default_model_path.exists():
                        from core.nlp.topic_modeling import TopicModeler
                        _model = TopicModeler.load_model(_default_model_path, verbose=False)
                _df = service._load_data()
                if _model is not None and _df is not None:
                    from ..auto_fetch import _compute_hierarchy_sunburst, PROCESSED_DATA_DIR
                    import re as _re
                    csv_path = paths.get("topics_csv") if paths else None
                    safe_q = ""
                    if csv_path:
                        m = _re.search(r"arxiv_(.+)_with_topics", str(csv_path))
                        safe_q = m.group(1) if m else ""
                    labels_map = {t.topic_id: (t.topic_label or t.topic_name) for t in service.get_all_topics()}
                    hier_data = _compute_hierarchy_sunburst(_model, _df, labels_map)
                    if safe_q:
                        _out = PROCESSED_DATA_DIR / f"arxiv_{safe_q}_hierarchy.json"
                        with open(_out, "w", encoding="utf-8") as _f:
                            _json.dump(hier_data, _f, ensure_ascii=False)
                        print(f"[Sunburst] 重算完成，已保存至 {_out.name}")
            except Exception as _e2:
                print(f"[Sunburst] 重算失败（{_e2}），回退 2 层")
                hier_data = None

        if hier_data:
            return SunburstResponse(
                labels=hier_data["labels"],
                parents=hier_data["parents"],
                values=hier_data["values"],
                ids=hier_data["ids"],
                topic_ids=hier_data["topic_ids"],
            )

        # ── 回退：2 层平铺结构 ────────────────────────────────────
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
            display_name = topic.topic_label or topic.topic_name
            labels.append(f"Topic {topic.topic_id}: {display_name}")
            parents.append("root")
            child_values.append(topic.paper_count)
            ids.append(f"topic_{topic.topic_id}")
            topic_ids.append(topic.topic_id)

        values = [sum(child_values)] + child_values
        return SunburstResponse(
            labels=labels,
            parents=parents,
            values=values,
            ids=ids,
            topic_ids=topic_ids
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate topic data: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Failed to retrieve trend data: {str(e)}")


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
        from datetime import datetime as _dt

        def _date_key(p):
            # 统一转为 timezone-naive，避免 aware vs. naive 比较异常
            if p.published is None:
                return _dt.min
            try:
                return p.published.replace(tzinfo=None)
            except Exception:
                return _dt.min

        all_papers = sorted(all_papers, key=_date_key, reverse=True)
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
    description=(
        "基于主题向量质心的余弦相似度推荐相关主题；"
        "若向量不可用，回退到 Jaccard 关键词相似度。"
    ),
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

    优先策略: 主题向量质心 + 余弦相似度（语义层面的"相关"）
    回退策略: 关键词 Jaccard 重叠（数据/向量缺失时）
    """
    target_topic = service.get_topic_by_id(topic_id)
    if target_topic is None:
        raise HTTPException(
            status_code=404,
            detail=f"Topic not found: {topic_id}"
        )

    try:
        all_topics = service.get_all_topics()
        topic_meta = {t.topic_id: t for t in all_topics}

        # ===== 优先：余弦相似度 =====
        method = "cosine"
        similar: list[dict] = []
        try:
            from ..auto_fetch import get_latest_data_paths
            from ..nlp.topic_similarity import get_topic_centroids, cosine_similar_topics

            paths = get_latest_data_paths()
            if paths:
                csv_path = paths.get("topics_csv")
                emb_path = paths.get("embeddings")
                centroids = (
                    get_topic_centroids(csv_path, emb_path)
                    if (csv_path and emb_path) else {}
                )
                if centroids and topic_id in centroids:
                    cosine_pairs = cosine_similar_topics(topic_id, centroids, top_n=top_n)
                    for tid, score in cosine_pairs:
                        meta = topic_meta.get(tid)
                        if meta is None:
                            continue
                        similar.append({
                            "topic_id": tid,
                            "topic_name": meta.topic_label or meta.topic_name,
                            "paper_count": meta.paper_count,
                            "similarity": round(float(score), 4),
                        })
        except Exception as e:
            print(f"[similar] 余弦计算失败，回退 Jaccard: {e}")
            similar = []

        # ===== 回退：Jaccard =====
        if not similar:
            method = "jaccard"
            target_words = {kw.word.lower() for kw in target_topic.keywords}
            for t in all_topics:
                if t.topic_id == topic_id:
                    continue
                t_words = {kw.word.lower() for kw in t.keywords}
                if not target_words and not t_words:
                    score = 0.0
                else:
                    inter = len(target_words & t_words)
                    union = len(target_words | t_words)
                    score = inter / union if union > 0 else 0.0
                similar.append({
                    "topic_id": t.topic_id,
                    "topic_name": t.topic_label or t.topic_name,
                    "paper_count": t.paper_count,
                    "similarity": round(score, 4),
                })
            similar.sort(key=lambda x: x["similarity"], reverse=True)
            similar = similar[:top_n]

        return {
            "topic_id": topic_id,
            "topic_name": target_topic.topic_label or target_topic.topic_name,
            "method": method,
            "similar_topics": similar,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve related topics: {str(e)}")
