"""
多数据源聚合器

实现并行抓取、去重、数量控制等核心逻辑
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Optional
import re
from .base import Paper, PaperSource


class MultiSourceAggregator:
    """
    多数据源聚合器
    
    功能:
    - 并行调用多个数据源（ThreadPoolExecutor）
    - 三级去重：DOI > arXiv ID > 标题相似度
    - 智能数量控制：过采样 + 迭代补充
    """
    
    def __init__(
        self,
        sources: List[PaperSource],
        max_workers: int = 3
    ):
        """
        初始化聚合器
        
        参数:
            sources: 数据源列表（如 [ArxivSource(), OpenAlexSource()]）
            max_workers: 最大并行线程数
        """
        self.sources = sources
        self.max_workers = max_workers
    
    def search_all(
        self,
        query: str,
        max_results: int = 200,
        oversample_ratio: float = 1.5,
        min_tolerance: float = 0.9,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Paper]:
        """
        多源并行搜索 + 去重

        参数:
            query: 搜索关键词
            max_results: 目标论文数量
            oversample_ratio: 过采样倍数（应对去重损失，默认 1.5）
            min_tolerance: 最小容忍度（0.9 = 至少达到 90% 目标数量）
            weights: 各数据源抓取权重 {"arxiv": 0.7, "openalex": 0.3}
                若为 None，则各源平均分配（保持原有行为）

        返回:
            去重后的 Paper 列表（最多 max_results 篇）
        """
        target = max_results
        total_budget = int(target * oversample_ratio)
        # 给每个 source 分配本轮请求数量
        per_source_map = self._allocate_budget(total_budget, weights)

        print(f"\n{'='*60}")
        print(f"[多源抓取] 目标: {target} 篇论文")
        print(f"[多源抓取] 数据源: {[s.get_source_name() for s in self.sources]}")
        if weights:
            print(f"[多源抓取] 智能权重: {weights}")
        print(f"[多源抓取] 各源请求: {per_source_map}（过采样倍数 {oversample_ratio}）")
        print(f"{'='*60}\n")

        # 第一轮：并行抓取（按权重分配每源数量）
        all_papers = self._fetch_parallel(query, per_source_map)
        
        print(f"\n[去重前] 总计: {len(all_papers)} 篇")
        
        # 去重
        unique_papers = self._deduplicate(all_papers)
        
        dedup_rate = 1 - len(unique_papers) / len(all_papers) if all_papers else 0
        print(f"[去重后] 剩余: {len(unique_papers)} 篇（重复率: {dedup_rate:.1%}）")
        
        # 检查是否需要补充
        min_acceptable = int(target * min_tolerance)
        if len(unique_papers) < min_acceptable:
            shortage = target - len(unique_papers)
            print(f"\n[警告] 不足目标数量，缺少 {shortage} 篇")
            print(f"[提示] 可尝试增加 oversample_ratio 或添加更多数据源")
        
        # 截断到目标数量
        result = unique_papers[:target]
        
        print(f"\n[最终结果] 返回: {len(result)} 篇论文")
        print(f"{'='*60}\n")
        
        return result
    
    def _allocate_budget(
        self,
        total_budget: int,
        weights: Optional[Dict[str, float]],
    ) -> Dict[str, int]:
        """
        按权重把总抓取预算分配给各数据源。

        - weights=None 或对应源缺失：均分
        - 任一源最少分到 1 篇，避免被完全跳过（用户已勾选则尊重）
        """
        n = len(self.sources)
        if n == 0:
            return {}

        # 默认均分
        if not weights:
            base = max(1, total_budget // n)
            return {s.get_source_name(): base for s in self.sources}

        # 按权重分配
        allocation: Dict[str, int] = {}
        used = 0
        for src in self.sources:
            name = src.get_source_name()
            w = weights.get(name, 1.0 / n)
            quota = max(1, int(round(total_budget * w)))
            allocation[name] = quota
            used += quota

        # 微调：避免明显超额（最大源吸收差额）
        if used > total_budget * 1.2 and allocation:
            # 等比缩放
            scale = (total_budget * 1.1) / used
            allocation = {k: max(1, int(v * scale)) for k, v in allocation.items()}
        return allocation

    def _fetch_parallel(
        self,
        query: str,
        per_source_map: Dict[str, int],
    ) -> List[Paper]:
        """
        并行抓取所有数据源

        参数:
            query: 搜索关键词
            per_source_map: 每个源的请求数量 {"arxiv": 60, "openalex": 30}

        返回:
            所有数据源的论文合集
        """
        all_papers = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_source = {
                executor.submit(
                    source.search,
                    query,
                    per_source_map.get(source.get_source_name(), 50),
                ): source
                for source in self.sources
            }

            for future in as_completed(future_to_source):
                source = future_to_source[future]
                source_name = source.get_source_name()
                try:
                    papers = future.result(timeout=120)
                    print(f"  [OK] {source_name:15s}: {len(papers):3d} 篇")
                    all_papers.extend(papers)
                except Exception as e:
                    print(f"  [FAIL] {source_name:15s}: 失败 - {e}")

        return all_papers
    
    def _deduplicate(self, papers: List[Paper]) -> List[Paper]:
        """
        三级去重策略
        
        优先级:
        1. DOI 精确匹配（最可靠）
        2. arXiv ID 匹配（针对预印本）
        3. 标题归一化匹配（去标点、小写、去空格）
        
        参数:
            papers: 待去重的论文列表
        
        返回:
            去重后的论文列表
        """
        seen_doi: Set[str] = set()
        seen_arxiv: Set[str] = set()
        seen_titles: Set[str] = set()
        unique_papers = []
        
        for paper in papers:
            is_duplicate = False
            
            # 级别1：DOI 去重
            if paper.doi:
                doi_key = paper.doi.lower().strip()
                if doi_key in seen_doi:
                    is_duplicate = True
                else:
                    seen_doi.add(doi_key)
            
            # 级别2：arXiv ID 去重
            if not is_duplicate and paper.arxiv_id:
                arxiv_key = paper.arxiv_id.lower().strip()
                if arxiv_key in seen_arxiv:
                    is_duplicate = True
                else:
                    seen_arxiv.add(arxiv_key)
            
            # 级别3：标题归一化去重
            if not is_duplicate:
                title_key = self._normalize_title(paper.title)
                if title_key in seen_titles:
                    is_duplicate = True
                else:
                    seen_titles.add(title_key)
            
            # 如果不是重复，加入结果
            if not is_duplicate:
                unique_papers.append(paper)
        
        return unique_papers
    
    @staticmethod
    def _normalize_title(title: str) -> str:
        """
        标题归一化（用于去重）
        
        步骤:
        1. 转小写
        2. 去除标点符号
        3. 去除多余空格
        4. 去除常见停用词（a, an, the 等）
        
        参数:
            title: 原始标题
        
        返回:
            归一化后的标题
        """
        # 转小写
        title = title.lower()
        
        # 去除标点符号（保留字母、数字、空格）
        title = re.sub(r'[^\w\s]', '', title)
        
        # 去除多余空格
        title = re.sub(r'\s+', ' ', title).strip()
        
        # 去除常见停用词
        stopwords = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from'}
        words = [w for w in title.split() if w not in stopwords]
        
        return ' '.join(words)
    
    def get_source_stats(self, papers: List[Paper]) -> Dict[str, int]:
        """
        统计各数据源的论文数量
        
        参数:
            papers: 论文列表
        
        返回:
            数据源统计字典 {"arxiv": 50, "openalex": 150}
        """
        stats = {}
        for paper in papers:
            source = paper.source
            stats[source] = stats.get(source, 0) + 1
        return stats
