"""
arXiv 数据源适配器

将现有的 arXiv 客户端包装为统一的 PaperSource 接口
"""
from __future__ import annotations
from typing import List
import arxiv
from .base import Paper, PaperSource


class ArxivSource(PaperSource):
    """
    arXiv 论文数据源
    
    基于官方 arxiv Python SDK 实现
    """
    
    def __init__(
        self,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending
    ):
        """
        初始化 arXiv 数据源
        
        参数:
            sort_by: 排序字段（Relevance | SubmittedDate | LastUpdatedDate）
            sort_order: 排序顺序（Ascending | Descending）
        """
        self.sort_by = sort_by
        self.sort_order = sort_order
    
    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """
        搜索 arXiv 论文
        
        参数:
            query: 搜索关键词
            max_results: 最大返回数量
        
        返回:
            统一的 Paper 对象列表
        """
        # 创建 arXiv 搜索对象
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=self.sort_by,
            sort_order=self.sort_order
        )
        
        papers = []
        
        try:
            for result in search.results():
                # 提取 arXiv ID（从 entry_id URL 中提取）
                arxiv_id = self._extract_arxiv_id(result.entry_id)
                
                # 清理标题和摘要（去除换行符）
                clean_title = result.title.strip().replace("\n", " ")
                clean_abstract = result.summary.strip().replace("\n", " ")
                
                # 构建 PDF URL
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None
                
                # 转换为统一 Paper 模型
                paper = Paper(
                    arxiv_id=arxiv_id,
                    doi=result.doi if result.doi else None,
                    title=clean_title,
                    abstract=clean_abstract,
                    authors=[author.name for author in result.authors],
                    published=result.published,
                    updated=result.updated,
                    source="arxiv",
                    url=result.entry_id,
                    pdf_url=pdf_url,
                    categories=list(result.categories) if result.categories else []
                )
                
                papers.append(paper)
        
        except Exception as e:
            print(f"[ArxivSource] 搜索失败: {e}")
            raise
        
        return papers
    
    def get_source_name(self) -> str:
        """返回数据源名称"""
        return "arxiv"
    
    @staticmethod
    def _extract_arxiv_id(entry_id: str) -> str:
        """
        从 arXiv entry_id URL 中提取 arXiv ID
        
        示例:
            http://arxiv.org/abs/1706.03762v5 -> 1706.03762
        """
        if "arxiv.org/abs/" in entry_id:
            arxiv_id = entry_id.split("arxiv.org/abs/")[-1]
            # 去除版本号（如 v1, v2）
            if "v" in arxiv_id:
                arxiv_id = arxiv_id.split("v")[0]
            return arxiv_id
        return entry_id
