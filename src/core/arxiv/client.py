from __future__ import annotations
import arxiv
from .model import ArxivPaper
from datetime import datetime
from typing import List

def fetch_arxiv_papers(
    query: str,
    max_results: int = 200,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending
) -> List[ArxivPaper]:
    """
    根据关键词从 arXiv 抓取论文列表，并转换为 ArxivPaper 模型列表。
    
    参数:
        query: 检索关键词
        max_results: 最多返回多少篇论文
        sort_by: 排序字段，默认为按提交日期
            Relevance = <SortCriterion.Relevance: 'relevance'>
            LastUpdatedDate = <SortCriterion.LastUpdatedDate: 'lastUpdatedDate'>
            SubmittedDate = <SortCriterion.SubmittedDate: 'submittedDate'>
        sort_order: 排序顺序，默认最新在前
            Ascending = <SortOrder.Ascending: 'ascending'>
            Descending = <SortOrder.Descending: 'descending'>
    返回:
        ArxivPaper 对象列表
    """

    # 创建搜索对象
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    # 官方API search格式
    # Search(
    #     query: str = '',
    #     id_list: list[str] | None = None,
    #     max_results: int | None = None,
    #     sort_by: SortCriterion = <SortCriterion.Relevance: 'relevance'>,
    #     sort_order: SortOrder = <SortOrder.Descending: 'descending'>
    # )

    # : List[ArxivPaper]：是注解
    papers: List[ArxivPaper] = []

    for result in search.results():
        clean_title = result.title.strip().replace("\n","")
        clean_abstract = result.summary.strip().replace("\n","")

        paper = ArxivPaper(
            id = result.entry_id,
            title = clean_title,
            abstract = clean_abstract,
            # 返回的result的authors列表，里面的作者都是一个类对象，而不是直接的字符串，
            # 所以要将实例中的name提出存入ArxivPaper的authors(字符串列表)
            authors = [author.name for author in result.authors],
            url = result.entry_id,
            #categories在返回的result中本身就是字符串列表了，可以直接传入，加上list作为保险
            categories=list(result.categories),
            updated = result.updated,
            published=result.published,
        )

        papers.append(paper)
        # 官方API result格式
        # Result(
        #     entry_id: str,
        #     updated: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0),
        #     published: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0),
        #     title: str = '',
        #     authors: list[Result.Author] | None = None,
        #     summary: str = '',
        #     comment: str = '',
        #     journal_ref: str = '',
        #     doi: str = '',
        #     primary_category: str = '',
        #     categories: list[str] | None = None,
        #     links: list[Result.Link] | None = None,
        #     _raw: feedparser.util.FeedParserDict | None = None
        #         )

    return papers