"""
按需在线抓取流水线

用户在前端输入关键词后，触发完整的数据处理流程：
  1. 在线抓取论文（arXiv + OpenAlex）
  2. 保存 CSV
  3. 生成向量（embeddings）
  4. 训练主题模型（BERTopic）
  5. 保存带主题标签的 CSV

所有步骤通过回调函数报告进度，前端可实时展示。
"""
from __future__ import annotations
import time
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any

import pandas as pd
import numpy as np

from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from .sources.base import Paper
from .sources.arxiv_adapter import ArxivSource
from .sources.aggregator import MultiSourceAggregator


# 进度回调类型: (step, total_steps, message)
ProgressCallback = Callable[[int, int, str], None]


def _default_progress(step: int, total: int, msg: str):
    print(f"[{step}/{total}] {msg}")


def papers_to_dataframe(papers: List[Paper]) -> pd.DataFrame:
    """将 Paper 对象列表转换为 DataFrame"""
    data = []
    for paper in papers:
        data.append({
            'id': paper.get_unique_id(),
            'doi': paper.doi or '',
            'arxiv_id': paper.arxiv_id or '',
            'openalex_id': paper.openalex_id or '',
            'title': paper.title,
            'abstract': paper.abstract,
            'authors': str(paper.authors),
            'published': paper.published.isoformat() if paper.published else None,
            'updated': paper.updated.isoformat() if paper.updated else None,
            'source': paper.source,
            'url': str(paper.url) if paper.url else '',
            'pdf_url': str(paper.pdf_url) if paper.pdf_url else '',
            'categories': str(paper.categories),
            'citations_count': paper.citations_count or 0,
            'venue': paper.venue or ''
        })
    return pd.DataFrame(data)


def run_pipeline(
    query: str,
    max_results: int = 50,
    sources: Optional[List[str]] = None,
    progress: ProgressCallback = _default_progress,
) -> Dict[str, Any]:
    """
    执行完整的按需抓取流水线

    参数:
        query: 用户输入的搜索关键词
        max_results: 抓取论文数量（默认 50，平衡速度和质量）
        sources: 数据源列表，默认 ["arxiv"]
        progress: 进度回调函数

    返回:
        {
            "query": str,
            "csv_path": str,           # 原始 CSV 路径
            "topics_csv_path": str,     # 带主题标签的 CSV 路径
            "embeddings_path": str,     # 向量文件路径
            "model_path": str,          # BERTopic 模型路径
            "paper_count": int,         # 论文数量
            "topic_count": int,         # 主题数量
        }
    """
    if sources is None:
        sources = ["arxiv"]

    total_steps = 5
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query.replace(' ', '_')[:30]

    # ========== 步骤 1: 在线抓取论文 ==========
    progress(1, total_steps, f"正在从 {', '.join(sources)} 在线抓取论文...")

    source_objects = []
    if "arxiv" in sources:
        source_objects.append(ArxivSource())
    try:
        from .openalex.client import OpenAlexSource
        if "openalex" in sources:
            source_objects.append(OpenAlexSource())
    except ImportError:
        pass

    if not source_objects:
        source_objects.append(ArxivSource())

    aggregator = MultiSourceAggregator(sources=source_objects, max_workers=len(source_objects))
    papers = aggregator.search_all(query=query, max_results=max_results, oversample_ratio=1.3)

    if not papers:
        raise RuntimeError(f"未能抓取到任何论文，请检查网络连接或更换关键词: {query}")

    # 统计各来源论文数量
    source_stats = aggregator.get_source_stats(papers)
    print(f"[来源统计] {source_stats}")

    # ========== 步骤 2: 保存 CSV ==========
    progress(2, total_steps, f"已抓取 {len(papers)} 篇论文，正在保存...")

    df = papers_to_dataframe(papers)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_filename = f"papers_{safe_query}_{timestamp}.csv"
    csv_path = RAW_DATA_DIR / csv_filename
    df.to_csv(csv_path, index=False, encoding='utf-8')

    # 同时保存一份标准名称的 CSV（供其他服务使用）
    standard_csv = RAW_DATA_DIR / f"arxiv_{safe_query}.csv"
    df.to_csv(standard_csv, index=False, encoding='utf-8')

    # ========== 步骤 3: 生成向量 ==========
    progress(3, total_steps, "正在生成论文向量（embedding）...")

    from .nlp.embeddings import EmbeddingGenerator

    generator = EmbeddingGenerator(model_name="all-mpnet-base-v2", batch_size=64)
    abstracts = df['abstract'].fillna("").tolist()
    embeddings = generator.encode_texts(abstracts, show_progress=True)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = PROCESSED_DATA_DIR / f"embeddings_{safe_query}.npy"
    np.save(str(embeddings_path), embeddings)

    # 同时保存标准名称
    standard_emb = PROCESSED_DATA_DIR / "embeddings.npy"
    np.save(str(standard_emb), embeddings)

    # ========== 步骤 4: 训练主题模型 ==========
    progress(4, total_steps, "正在训练主题模型（BERTopic）...")

    from .nlp.topic_modeling import TopicModeler

    # 根据论文数量动态调整 min_topic_size
    n_papers = len(df)
    min_topic_size = max(3, min(10, n_papers // 10))

    modeler = TopicModeler(min_topic_size=min_topic_size, verbose=True)
    topics, probs = modeler.fit(abstracts, embeddings)

    # 添加主题标签到 DataFrame
    df['topic_id'] = topics
    topic_name_map = {}
    for topic_id in set(topics):
        topic_name_map[topic_id] = modeler.get_topic_name(topic_id)
    df['topic_name'] = df['topic_id'].map(topic_name_map)

    # 保存带主题标签的 CSV
    topics_csv_path = PROCESSED_DATA_DIR / f"papers_{safe_query}_with_topics.csv"
    df.to_csv(topics_csv_path, index=False, encoding='utf-8')

    # 同时保存标准名称（供 topic_service 使用）
    standard_topics_csv = PROCESSED_DATA_DIR / f"arxiv_{safe_query}_with_topics.csv"
    df.to_csv(standard_topics_csv, index=False, encoding='utf-8')

    # 保存模型
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "bertopic_model"
    modeler.save_model(model_path)

    # ========== 步骤 5: 完成 ==========
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    progress(5, total_steps, f"完成！共 {len(papers)} 篇论文，{n_topics} 个主题")

    result = {
        "query": query,
        "csv_path": str(standard_csv),
        "topics_csv_path": str(standard_topics_csv),
        "embeddings_path": str(standard_emb),
        "model_path": str(model_path),
        "paper_count": len(papers),
        "topic_count": n_topics,
        "source_stats": source_stats,
    }

    return result


def get_latest_data_paths(query: Optional[str] = None) -> Optional[Dict[str, Path]]:
    """
    查找最新的数据文件路径

    参数:
        query: 可选的查询关键词，用于精确匹配

    返回:
        {"csv": Path, "topics_csv": Path, "embeddings": Path, "model": Path}
        如果没有找到数据返回 None
    """
    # 查找带主题标签的 CSV
    if query:
        safe_query = query.replace(' ', '_')[:30]
        topics_csv = PROCESSED_DATA_DIR / f"arxiv_{safe_query}_with_topics.csv"
        raw_csv = RAW_DATA_DIR / f"arxiv_{safe_query}.csv"
        if topics_csv.exists():
            return {
                "csv": raw_csv if raw_csv.exists() else topics_csv,
                "topics_csv": topics_csv,
                "embeddings": PROCESSED_DATA_DIR / f"embeddings_{safe_query}.npy",
                "model": MODELS_DIR / "bertopic_model",
            }

    # 回退：查找任意已有的数据
    topic_csvs = list(PROCESSED_DATA_DIR.glob("*_with_topics.csv"))
    if topic_csvs:
        latest = max(topic_csvs, key=lambda p: p.stat().st_mtime)
        return {
            "csv": latest,
            "topics_csv": latest,
            "embeddings": PROCESSED_DATA_DIR / "embeddings.npy",
            "model": MODELS_DIR / "bertopic_model",
        }

    # 再回退：查找原始 CSV
    raw_csvs = list(RAW_DATA_DIR.glob("*.csv"))
    if raw_csvs:
        latest = max(raw_csvs, key=lambda p: p.stat().st_mtime)
        return {
            "csv": latest,
            "topics_csv": None,
            "embeddings": PROCESSED_DATA_DIR / "embeddings.npy",
            "model": MODELS_DIR / "bertopic_model",
        }

    return None
