from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, json_schema


class PaperResponse(BaseModel):
    """
    单论文响应模型(API返回给客户端的论文格式)
    """
    id: str = Field(..., description="论文ID")
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="论文标题",
    )
    abstract: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="论文摘要",
    )
    published: datetime = Field(..., description="发布日期")
    authors: List[str] = Field(..., description="作者列表")
    pdf_url: Optional[HttpUrl] = Field(None, description="PDF 下载链接")
    categories: List[str] = Field(default_factory=list, description="论文分类")

    class Config:
        json_schema_extra = {
            "example":{
                "id":"2301.00001",
                "title":"Example Paper on BERT",
                "authors":["Alice Smith","Bob Johnson"],
                "abstract":"This paper discusses advanced techniques in natural language processing using BERT models.",
                "published": "2023-01-01T00:00:00",
                "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
                "categories": ["cs.CL", "cs.AI"]
            }
        }
class PaperListResponse(BaseModel):
    """
    论文列表的相应模型，包含论文数组喝元数据
    """
    total: int = Field(...,description="论文总数")
    papers: List[PaperResponse] = Field(...,description="论文列表")
    class Config:
        json_schema_extra = {
            "example":{
                "total": 2,
                "papers":[
                    {
                        "id": "2301.00001",
                        "title": "Paper 1",
                        "authors": ["Alice"],
                        "abstract": "Abstract content for paper 1 with sufficient length for validation.",
                        "published": "2023-01-01T00:00:00",
                        "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
                        "categories": ["cs.CL"]
                    },
                    {
                        "id": "2301.00002",
                        "title": "Paper 2",
                        "authors": ["Bob"],
                        "abstract": "Abstract content for paper 2 with sufficient length for validation.",
                        "published": "2023-01-02T00:00:00",
                        "pdf_url": "https://arxiv.org/pdf/2301.00002.pdf",
                        "categories": ["cs.AI"]
                    }
                ]
            }
        }

class SearchRequest(BaseModel):
    """
    搜索请求模型

    客户端发送的搜索参数
    """
    query: str = Field(...,min_length=1,description="搜索关键词")
    max_results: int = Field(10,ge=1,le=100,description="最多返回数量")
    fields: List[str] = Field(
        default_factory=lambda: ["title", "abstract"],
        description="搜索字段"
    )

    class Config:
        json_schema_extra = {
            "example":{
                "query": "BERT",
                "max_results":10,
                "fields":["title", "abstract"]
            }
        }

class ErrorResponse(BaseModel):
    """
    错误响应模型(API返回错误格式)
    """
    error: str = Field(...,description="错误类型")
    message: str = Field(...,description="错误详情")

    class Config:
        json_schema_extra = {
            "example":{
                "error":"NotFound",
                "message":"Paper not found"
            }
        }


class TopicKeyword(BaseModel):
    """
    主题关键词模型
    """
    word: str = Field(..., description="关键词")
    score: float = Field(..., ge=0.0, le=1.0, description="关键词得分")

    class Config:
        json_schema_extra = {
            "example": {
                "word": "transformer",
                "score": 0.85
            }
        }


class TopicResponse(BaseModel):
    """
    主题响应模型
    """
    topic_id: int = Field(..., description="主题ID")
    topic_name: str = Field(..., description="主题名称")
    keywords: List[TopicKeyword] = Field(..., description="关键词列表")
    paper_count: int = Field(..., ge=0, description="论文数量")

    class Config:
        json_schema_extra = {
            "example": {
                "topic_id": 0,
                "topic_name": "transformer_attention_model",
                "keywords": [
                    {"word": "transformer", "score": 0.85},
                    {"word": "attention", "score": 0.78},
                    {"word": "model", "score": 0.72}
                ],
                "paper_count": 45
            }
        }


class TopicListResponse(BaseModel):
    """
    主题列表响应模型
    """
    total: int = Field(..., description="主题总数")
    topics: List[TopicResponse] = Field(..., description="主题列表")

    class Config:
        json_schema_extra = {
            "example": {
                "total": 2,
                "topics": [
                    {
                        "topic_id": 0,
                        "topic_name": "transformer_attention_model",
                        "keywords": [
                            {"word": "transformer", "score": 0.85},
                            {"word": "attention", "score": 0.78}
                        ],
                        "paper_count": 45
                    },
                    {
                        "topic_id": 1,
                        "topic_name": "neural_network_learning",
                        "keywords": [
                            {"word": "neural", "score": 0.82},
                            {"word": "network", "score": 0.75}
                        ],
                        "paper_count": 38
                    }
                ]
            }
        }