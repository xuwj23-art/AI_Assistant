"""
arXiv 数据模型定义

使用 Pydantic BaseModel 定义标准化的数据结构。
这相当于 C 语言中的 struct,但带有类型检查和自动验证功能。
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, HttpUrl, Field


class ArxivPaper(BaseModel):
    """
    表示一篇从 arXiv 抓取到的论文的标准结构。

    为什么使用 Pydantic?
    1. 自动类型验证: 如果传入的数据类型不对,会立即报错
    2. 数据序列化: 可以轻松转换为 JSON/dict
    3. IDE 支持: 提供良好的代码补全和类型提示
    4. 文档生成: FastAPI 会用这些模型自动生成 API 文档
    """

    # 论文唯一标识符 (例如: http://arxiv.org/abs/2203.05794v1)
    id: str = Field(..., description="arXiv entry ID (URL)")

    # 论文标题
    title: str = Field(..., min_length=1, description="Paper title")

    # 论文摘要 (通常是我们做 embedding 和聚类的主要文本)
    abstract: str = Field(..., min_length=1, description="Paper abstract")

    # 作者列表
    authors: List[str] = Field(default_factory=list, description="List of author names")

    # 发布时间
    published: datetime = Field(..., description="Original publication date")

    # 最后更新时间
    updated: datetime = Field(..., description="Last update date")

    # 论文分类 (例如: ['cs.CL', 'cs.AI'])
    categories: List[str] = Field(
        default_factory=list, description="arXiv category tags"
    )

    # 论文链接
    url: HttpUrl = Field(..., description="Full arXiv URL")

    # 可选字段: 主题 ID (BERTopic 聚类后会填充这个字段)
    topic_id: Optional[int] = Field(None, description="Topic cluster ID from BERTopic")

    # 可选字段: 主题名称
    topic_name: Optional[str] = Field(None, description="Topic cluster name")

    class Config:
        """Pydantic 配置"""

        # 允许使用示例数据 (用于 API 文档)
        json_schema_extra = {
            "example": {
                "id": "http://arxiv.org/abs/2203.05794v1",
                "title": "BERTopic: Neural topic modeling with a class-based TF-IDF procedure",
                "abstract": "Topic modeling is a technique for discovering...",
                "authors": ["Maarten Grootendorst"],
                "published": "2022-03-11T09:00:00Z",
                "updated": "2022-03-11T09:00:00Z",
                "categories": ["cs.CL", "cs.LG"],
                "url": "http://arxiv.org/abs/2203.05794v1",
            }
        }

