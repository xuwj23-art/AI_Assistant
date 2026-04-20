"""
OpenAlex API 客户端

OpenAlex 是一个完全免费、无速率限制的学术论文数据库
API 文档: https://docs.openalex.org/
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx
from ..sources.base import Paper, PaperSource


class OpenAlexSource(PaperSource):
    """
    OpenAlex 论文数据源
    
    特点:
    - 完全免费，无 API 密钥要求
    - 无速率限制
    - 覆盖所有学科领域
    - 提供引用次数、DOI 等丰富元数据
    """
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, email: Optional[str] = None, timeout: int = 30):
        """
        初始化 OpenAlex 数据源
        
        参数:
            email: 可选的邮箱（用于礼貌池，获得更快响应）
            timeout: 请求超时时间（秒）
        """
        self.email = email
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
    
    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """
        搜索 OpenAlex 论文
        
        参数:
            query: 搜索关键词
            max_results: 最大返回数量
        
        返回:
            统一的 Paper 对象列表
        """
        papers = []
        per_page = min(max_results, 200)  # OpenAlex 单页最多 200 条
        page = 1
        
        while len(papers) < max_results:
            try:
                # 构建请求参数
                params = {
                    "search": query,
                    "per_page": per_page,
                    "page": page,
                    "select": "id,doi,title,authorships,abstract_inverted_index,publication_year,publication_date,primary_topic,cited_by_count,primary_location"
                }
                
                # 如果提供了邮箱，加入礼貌池
                if self.email:
                    params["mailto"] = self.email
                
                # 发送请求
                response = self.client.get(
                    f"{self.BASE_URL}/works",
                    params=params
                )
                response.raise_for_status()
                
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    break
                
                # 解析每篇论文
                for work in results:
                    paper = self._parse_work(work)
                    if paper:
                        papers.append(paper)
                    
                    if len(papers) >= max_results:
                        break
                
                page += 1
                
            except httpx.HTTPError as e:
                print(f"[OpenAlexSource] HTTP 错误: {e}")
                break
            except Exception as e:
                print(f"[OpenAlexSource] 解析错误: {e}")
                break
        
        return papers[:max_results]
    
    def _parse_work(self, work: Dict[str, Any]) -> Optional[Paper]:
        """
        解析 OpenAlex work 对象为统一 Paper 模型
        
        参数:
            work: OpenAlex API 返回的 work 对象
        
        返回:
            Paper 对象，解析失败返回 None
        """
        try:
            # 提取 OpenAlex ID（去除 URL 前缀）
            openalex_id = work.get("id", "").replace("https://openalex.org/", "")
            
            # 标题
            title = work.get("title", "").strip()
            if not title:
                return None
            
            # 摘要（需要从倒排索引重建）
            abstract = self._reconstruct_abstract(
                work.get("abstract_inverted_index")
            )
            if not abstract:
                # 如果没有摘要，跳过该论文
                return None
            
            # 作者列表
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                author_name = author.get("display_name")
                if author_name:
                    authors.append(author_name)
            
            # DOI
            doi = work.get("doi")
            if doi and doi.startswith("https://doi.org/"):
                doi = doi.replace("https://doi.org/", "")
            
            # 发表日期
            pub_date_str = work.get("publication_date")
            published = None
            if pub_date_str:
                try:
                    published = datetime.fromisoformat(pub_date_str)
                except:
                    pass
            
            # 学科分类
            categories = []
            primary_topic = work.get("primary_topic")
            if primary_topic:
                topic_name = primary_topic.get("display_name")
                if topic_name:
                    categories.append(topic_name)
            
            # 引用次数
            citations_count = work.get("cited_by_count", 0)
            
            # URL 和 PDF
            url = f"https://openalex.org/{openalex_id}"
            pdf_url = None
            
            # 尝试从 primary_location 获取 PDF
            primary_location = work.get("primary_location")
            if primary_location:
                pdf_url = primary_location.get("pdf_url")
                if not pdf_url:
                    # 如果是 arXiv 论文，构建 PDF URL
                    landing_page = primary_location.get("landing_page_url", "")
                    if "arxiv.org/abs/" in landing_page:
                        arxiv_id = landing_page.split("arxiv.org/abs/")[-1]
                        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # 发表期刊/会议
            venue = None
            if primary_location:
                source = primary_location.get("source")
                if source:
                    venue = source.get("display_name")
            
            # 构建 Paper 对象
            paper = Paper(
                openalex_id=openalex_id,
                doi=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                published=published,
                source="openalex",
                url=url,
                pdf_url=pdf_url,
                categories=categories,
                citations_count=citations_count,
                venue=venue
            )
            
            return paper
        
        except Exception as e:
            print(f"[OpenAlexSource] 解析 work 失败: {e}")
            return None
    
    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
        """
        从 OpenAlex 的倒排索引重建摘要原文
        
        OpenAlex 的摘要格式示例:
        {
            "The": [0, 15],
            "dominant": [1],
            "sequence": [2],
            ...
        }
        
        参数:
            inverted_index: 倒排索引字典
        
        返回:
            重建的摘要文本
        """
        if not inverted_index:
            return ""
        
        try:
            # 创建位置到单词的映射
            position_to_word = {}
            for word, positions in inverted_index.items():
                for pos in positions:
                    position_to_word[pos] = word
            
            # 按位置排序并拼接
            sorted_positions = sorted(position_to_word.keys())
            words = [position_to_word[pos] for pos in sorted_positions]
            
            return " ".join(words)
        
        except Exception as e:
            print(f"[OpenAlexSource] 重建摘要失败: {e}")
            return ""
    
    def get_source_name(self) -> str:
        """返回数据源名称"""
        return "openalex"
    
    def __del__(self):
        """清理 HTTP 客户端"""
        if hasattr(self, 'client'):
            self.client.close()
