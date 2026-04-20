"""
RAG API 路由

定义基于RAG的对话接口
"""
from fastapi import APIRouter, HTTPException, Query, Body, Depends
from typing import Optional, List
from pydantic import BaseModel, Field

from .models import ErrorResponse
from .chat_service import get_chat_service, ChatService


# 请求/响应模型
class ChatInitRequest(BaseModel):
    """对话初始化请求"""
    session_id: Optional[str] = Field(None, description="会话ID（不指定则自动生成）")
    topic_id: Optional[int] = Field(None, description="主题ID（可选）")


class ChatInitResponse(BaseModel):
    """对话初始化响应"""
    session_id: str = Field(..., description="会话ID")
    topic_id: Optional[int] = Field(None, description="主题ID")
    topic_name: Optional[str] = Field(None, description="主题名称")
    message: str = Field(..., description="欢迎消息")


class ChatMessageRequest(BaseModel):
    """发送消息请求"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., min_length=1, description="用户消息")


class SourceInfo(BaseModel):
    """来源信息"""
    id: str = Field(..., description="论文ID")
    title: str = Field(..., description="论文标题")
    authors: List[str] = Field(..., description="作者")
    published: Optional[str] = Field(None, description="发布日期")
    pdf_url: Optional[str] = Field(None, description="PDF链接")
    relevance: Optional[float] = Field(None, description="相关性得分")
    abstract: Optional[str] = Field(None, description="原始摘要")
    ai_summary: Optional[str] = Field(None, description="AI生成的摘要总结")


class ChatMessageResponse(BaseModel):
    """发送消息响应"""
    answer: str = Field(..., description="回答内容")
    sources: List[SourceInfo] = Field(..., description="参考来源")
    question: str = Field(..., description="原始问题")
    summary_enabled: bool = Field(False, description="是否启用AI摘要")


class HistoryItem(BaseModel):
    """历史记录项"""
    role: str = Field(..., description="角色（user/assistant）")
    content: str = Field(..., description="内容")
    timestamp: str = Field(..., description="时间戳")


class ChatHistoryResponse(BaseModel):
    """对话历史响应"""
    session_id: str = Field(..., description="会话ID")
    history: List[HistoryItem] = Field(..., description="历史记录")


# 创建路由器
router = APIRouter(
    prefix="/api/chat",
    tags=["RAG对话"]
)


@router.post(
    "/init",
    response_model=ChatInitResponse,
    summary="初始化对话",
    description="创建一个新的对话会话"
)
def init_chat(
        request: ChatInitRequest = Body(...),
        service: ChatService = Depends(get_chat_service)
):
    """
    初始化对话会话

    参数:
        - session_id: 可选，指定会话ID
        - topic_id: 可选，限定主题范围

    返回:
        - 会话信息
    """
    result = service.init_session(
        session_id=request.session_id,
        topic_id=request.topic_id
    )

    return result


@router.post(
    "/message",
    response_model=ChatMessageResponse,
    summary="发送消息",
    description="向RAG系统发送问题并获取答案",
    responses={
        404: {"model": ErrorResponse, "description": "会话不存在"}
    }
)
def send_message(
        request: ChatMessageRequest = Body(...),
        service: ChatService = Depends(get_chat_service)
):
    """
    发送消息并获取回答

    参数:
        - session_id: 会话ID
        - message: 用户问题

    返回:
        - 回答内容和参考来源
    """
    # 检查会话是否存在
    session_info = service.get_session_info(request.session_id)
    if session_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )

    # 发送消息
    result = service.send_message(
        session_id=request.session_id,
        message=request.message
    )

    return result


@router.get(
    "/history",
    response_model=ChatHistoryResponse,
    summary="获取对话历史",
    description="获取指定会话的对话历史",
    responses={
        404: {"model": ErrorResponse, "description": "会话不存在"}
    }
)
def get_history(
        session_id: str = Query(..., description="会话ID"),
        service: ChatService = Depends(get_chat_service)
):
    """
    获取对话历史

    参数:
        - session_id: 会话ID

    返回:
        - 对话历史记录
    """
    # 检查会话是否存在
    session_info = service.get_session_info(session_id)
    if session_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )

    # 获取历史
    result = service.get_history(session_id)

    return result


@router.get(
    "/sessions/{session_id}",
    response_model=ChatInitResponse,
    summary="获取会话信息",
    description="获取指定会话的详细信息",
    responses={
        404: {"model": ErrorResponse, "description": "会话不存在"}
    }
)
def get_session(
        session_id: str,
        service: ChatService = Depends(get_chat_service)
):
    """
    获取会话信息

    参数:
        - session_id: 会话ID

    返回:
        - 会话详细信息
    """
    session_info = service.get_session_info(session_id)
    if session_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )

    return {
        "session_id": session_info["session_id"],
        "topic_id": session_info.get("topic_id"),
        "topic_name": None,  # 可以从topic_service获取
        "message": f"会话已存在，共有 {session_info['message_count']} 条消息"
    }