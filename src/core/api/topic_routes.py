"""
主题管理 API 路由

定义主题相关的 API 端点
"""
from fastapi import APIRouter, HTTPException, Query, Path, Depends
from typing import List
from .models import TopicResponse, TopicListResponse, PaperListResponse, ErrorResponse
from .topic_service import TopicService


# 创建路由器
router = APIRouter(
    prefix="/api",
    tags=["主题管理"]
)


# 依赖注入：获取主题服务实例
_topic_service: TopicService = None


def get_topic_service() -> TopicService:
    """
    获取主题服务实例（依赖注入）
    """
    global _topic_service
    
    if _topic_service is None:
        from pathlib import Path
        # 构建数据文件路径
        project_root = Path(__file__).parent.parent.parent.parent
        
        # 优先查找带主题标签的数据文件
        data_candidates = [
            project_root / "data" / "processed" / "arxiv_Transformer_with_topics.csv",
            project_root / "data" / "raw" / "arxiv_Transformer.csv",
        ]
        
        data_path = None
        for candidate in data_candidates:
            if candidate.exists():
                data_path = candidate
                break
        
        if data_path is None:
            data_path = data_candidates[0]  # 使用默认路径
        
        # 模型路径
        model_path = project_root / "models" / "bertopic_model"
        if not model_path.exists():
            model_path = None
        
        _topic_service = TopicService(
            str(data_path),
            str(model_path) if model_path else None
        )
        print(f"[API] 初始化 TopicService: {data_path}")
    
    return _topic_service


@router.get(
    "/topics",
    response_model=TopicListResponse,
    summary="获取主题列表",
    description="获取所有主题及其统计信息"
)
def get_topics(
    service: TopicService = Depends(get_topic_service)
):
    """
    获取主题列表
    
    返回:
        - 所有主题的列表和总数
    """
    try:
        topics = service.get_all_topics()
        
        return TopicListResponse(
            total=len(topics),
            topics=topics
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取主题列表失败: {str(e)}"
        )


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
    """
    获取主题详情
    
    参数:
        - topic_id: 主题的ID
    
    返回:
        - 主题详细信息
    
    错误:
        - 404: 主题不存在
    """
    topic = service.get_topic_by_id(topic_id)
    
    if topic is None:
        raise HTTPException(
            status_code=404,
            detail=f"Topic not found: {topic_id}"
        )
    
    return topic


@router.get(
    "/topics/{topic_id}/papers",
    response_model=PaperListResponse,
    summary="获取主题下的论文",
    description="获取某个主题下的所有论文",
    responses={
        404: {"model": ErrorResponse, "description": "主题不存在"}
    }
)
def get_topic_papers(
    topic_id: int = Path(..., description="主题ID"),
    limit: int = Query(50, ge=1, le=100, description="返回数量"),
    service: TopicService = Depends(get_topic_service)
):
    """
    获取主题下的论文
    
    参数:
        - topic_id: 主题ID
        - limit: 最多返回数量（1-100）
    
    返回:
        - 该主题下的论文列表
    
    错误:
        - 404: 主题不存在
    """
    # 先检查主题是否存在
    topic = service.get_topic_by_id(topic_id)
    if topic is None:
        raise HTTPException(
            status_code=404,
            detail=f"Topic not found: {topic_id}"
        )
    
    # 获取论文
    papers = service.get_papers_by_topic(topic_id, limit=limit)
    
    return PaperListResponse(
        total=len(papers),
        papers=papers
    )
