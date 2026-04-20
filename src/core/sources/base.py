"""
多数据源统一接口基类

定义跨数据源的统一论文模型和抽象接口
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List
from datetime import datetime


class Paper(BaseModel):
    """
    统一论文数据模型（跨数据源）
    
    用于整合来自 arXiv、OpenAlex、Semantic Scholar 等不同来源的论文数据
    """
    # ===== 核心标识字段（用于去重） =====
    doi: Optional[str] = Field(None, description="DOI 标识符（最可靠的去重依据）")
    arxiv_id: Optional[str] = Field(None, description="arXiv ID（预印本标识）")
    openalex_id: Optional[str] = Field(None, description="OpenAlex ID")
    
    # ===== 基础信息 =====
    title: str = Field(..., description="论文标题")
    abstract: str = Field(..., description="论文摘要")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    published: Optional[datetime] = Field(None, description="发表日期")
    updated: Optional[datetime] = Field(None, description="更新日期")
    
    # ===== 来源标识 =====
    source: str = Field(..., description="数据来源：arxiv | openalex | semantic_scholar")
    
    # ===== 可选字段 =====
    url: Optional[HttpUrl] = Field(None, description="论文页面 URL")
    pdf_url: Optional[HttpUrl] = Field(None, description="PDF 下载链接")
    categories: List[str] = Field(default_factory=list, description="学科分类")
    citations_count: Optional[int] = Field(None, description="引用次数（OpenAlex 提供）")
    venue: Optional[str] = Field(None, description="发表期刊/会议")
    
    class Config:
        json_schema_extra = {
            "example": {
                "doi": "10.48550/arXiv.1706.03762",
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "abstract": "The dominant sequence transduction models...",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "published": "2017-06-12T00:00:00",
                "source": "arxiv",
                "url": "https://arxiv.org/abs/1706.03762",
                "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
                "categories": ["cs.CL", "cs.LG"],
                "citations_count": 50000
            }
        }
    
    def get_unique_id(self) -> str:
        """
        获取论文的唯一标识符（用于去重）
        
        优先级：DOI > arXiv ID > OpenAlex ID > 标题哈希
        """
        if self.doi:
            return f"doi:{self.doi.lower().strip()}"
        elif self.arxiv_id:
            return f"arxiv:{self.arxiv_id.lower().strip()}"
        elif self.openalex_id:
            return f"openalex:{self.openalex_id.lower().strip()}"
        else:
            # 使用标题哈希作为最后手段
            import hashlib
            title_hash = hashlib.md5(self.title.lower().encode()).hexdigest()[:12]
            return f"title:{title_hash}"


class PaperSource(ABC):
    """
    论文数据源抽象基类
    
    所有数据源（arXiv、OpenAlex 等）都需要实现此接口
    """
    
    @abstractmethod
    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """
        搜索论文
        
        参数:
            query: 搜索关键词
            max_results: 最大返回数量
        
        返回:
            统一的 Paper 对象列表
        """
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """
        返回数据源名称
        
        返回:
            数据源标识符（如 "arxiv", "openalex"）
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.get_source_name()}>"
