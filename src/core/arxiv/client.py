"""
arXiv API 客户端

负责从 arXiv 抓取论文数据并转换为标准化的 ArxivPaper 模型。

设计理念:
- 单一职责: 这个模块只负责"获取数据",不做数据分析或存储
- 类型安全: 使用类型注解,便于 IDE 检查和代码维护
- 容错性: 处理可能的异常情况 (如网络错误、数据缺失)
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import List

import arxiv

from .models import ArxivPaper


def fetch_arxiv_papers(
    query: str,
    max_results: int = 200,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
) -> List[ArxivPaper]:
    """
    根据关键词从 arXiv 抓取论文列表,并转换为 ArxivPaper 模型列表。

    参数说明:
    ----------
    query : str
        检索关键词,例如 "large language models"
        支持 arXiv 的高级搜索语法,如: "cat:cs.CL AND abs:transformer"

    max_results : int, 默认 200
        最多返回多少篇论文
        注意: arXiv API 建议单次请求不要太大,避免超时

    sort_by : arxiv.SortCriterion
        排序字段,默认按提交日期排序
        其他选项: Relevance (相关性), LastUpdatedDate

    sort_order : arxiv.SortOrder
        排序顺序,默认降序 (最新的在前)

    返回值:
    -------
    List[ArxivPaper]
        论文列表,每个元素都是经过验证的 ArxivPaper 对象

    异常处理:
    ---------
    - 如果网络出错,arxiv 包会抛出异常
    - 如果数据格式异常,Pydantic 会在构造 ArxivPaper 时抛出 ValidationError

    教学要点:
    ---------
    Q: 为什么不用 async/await?
    A: 当前阶段我们优先保持代码简单。arXiv API 的 `arxiv` 包本身是同步的,
       如果未来需要并发抓取多个关键词,可以在外层用 asyncio + ThreadPoolExecutor。

    Q: 如何处理 Rate Limit?
    A: arXiv 官方建议每 3 秒不超过 1 次请求。当前我们单次拉取 100-200 篇,
       只会发出 1-2 次 HTTP 请求,不会触发限流。如果未来需要频繁调用,
       可以在循环中加 time.sleep(3)。
    """

    # 构造搜索对象 (这里还没有发出网络请求)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    papers: List[ArxivPaper] = []

    # 迭代搜索结果 (这里才真正发出网络请求并解析 XML)
    for result in search.results():
        try:
            # 清洗文本: 去掉多余换行符,避免后续处理问题
            clean_title = result.title.strip().replace("\n", " ")
            clean_abstract = result.summary.strip().replace("\n", " ")

            # 构造 ArxivPaper 对象
            # Pydantic 会自动验证每个字段的类型和约束
            paper = ArxivPaper(
                id=result.entry_id,
                title=clean_title,
                abstract=clean_abstract,
                authors=[author.name for author in result.authors],
                # 有些论文可能缺少时间字段,用当前时间兜底
                published=result.published or datetime.utcnow(),
                updated=result.updated or result.published or datetime.utcnow(),
                categories=list(result.categories),
                url=result.entry_id,
            )
            papers.append(paper)

        except Exception as e:
            # 如果某篇论文解析失败,打印警告但继续处理其他论文
            print(f"[WARNING] Failed to parse paper {result.entry_id}: {e}")
            continue

    return papers


def fetch_arxiv_papers_with_retry(
    query: str,
    max_results: int = 200,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> List[ArxivPaper]:
    """
    带重试机制的 arXiv 抓取函数 (生产环境推荐)。

    如果遇到网络错误,会自动重试最多 max_retries 次。
    """
    for attempt in range(max_retries):
        try:
            return fetch_arxiv_papers(query=query, max_results=max_results)
        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"[WARNING] Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                print(f"[ERROR] All {max_retries} attempts failed. Giving up.")
                raise

    return []  # Should never reach here

