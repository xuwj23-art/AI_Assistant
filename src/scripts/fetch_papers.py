"""
多源论文抓取脚本

功能:
- 支持多数据源并行抓取（arXiv + OpenAlex）
- 自动去重（DOI + arXiv ID + 标题相似度）
- ChromaDB 持久化缓存
- 导出 CSV 文件

使用示例:
    # 从 arXiv 和 OpenAlex 抓取 200 篇论文
    python -m scripts.fetch_papers --query "transformer" --max-results 200 --sources arxiv openalex
    
    # 仅从 arXiv 抓取
    python -m scripts.fetch_papers --query "deep learning" --max-results 100 --sources arxiv
    
    # 抓取并生成向量
    python -m scripts.fetch_papers --query "NLP" --max-results 150 --sources arxiv openalex --generate-embeddings
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sources.base import Paper
from core.sources.arxiv_adapter import ArxivSource
from core.openalex.client import OpenAlexSource
from core.sources.aggregator import MultiSourceAggregator
from core.storage.chroma_store import ChromaStore
from core.nlp.embeddings import EmbeddingGenerator
from core.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DB_DIR


def papers_to_dataframe(papers: list[Paper]) -> pd.DataFrame:
    """
    将 Paper 对象列表转换为 DataFrame
    
    参数:
        papers: Paper 对象列表
    
    返回:
        DataFrame
    """
    data = []
    for paper in papers:
        data.append({
            'id': paper.get_unique_id(),
            'doi': paper.doi or '',
            'arxiv_id': paper.arxiv_id or '',
            'openalex_id': paper.openalex_id or '',
            'title': paper.title,
            'abstract': paper.abstract,
            'authors': str(paper.authors),  # 转为字符串存储
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


def main():
    parser = argparse.ArgumentParser(
        description="多源论文抓取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="搜索关键词"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=200,
        help="目标论文数量（默认 200）"
    )
    
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["arxiv", "openalex"],
        default=["arxiv", "openalex"],
        help="数据源列表（默认: arxiv openalex）"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help=f"CSV 输出目录（默认: {RAW_DATA_DIR}）"
    )
    
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        help="是否生成向量（耗时较长）"
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-mpnet-base-v2",
        help="向量模型名称（默认: all-mpnet-base-v2）"
    )
    
    parser.add_argument(
        "--oversample-ratio",
        type=float,
        default=1.5,
        help="过采样倍数（默认 1.5，应对去重损失）"
    )
    
    args = parser.parse_args()
    
    # 打印配置
    print("\n" + "=" * 70)
    print("多源论文抓取工具")
    print("=" * 70)
    print(f"搜索关键词: {args.query}")
    print(f"目标数量: {args.max_results} 篇")
    print(f"数据源: {', '.join(args.sources)}")
    print(f"过采样倍数: {args.oversample_ratio}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 70 + "\n")
    
    # 初始化数据源
    sources = []
    if "arxiv" in args.sources:
        sources.append(ArxivSource())
        print("[OK] arXiv 数据源已加载")
    
    if "openalex" in args.sources:
        sources.append(OpenAlexSource())
        print("[OK] OpenAlex 数据源已加载")
    
    if not sources:
        print("[错误] 至少需要指定一个数据源")
        sys.exit(1)
    
    # 创建聚合器
    aggregator = MultiSourceAggregator(sources=sources, max_workers=len(sources))
    
    # 抓取论文
    print(f"\n开始抓取论文...")
    papers = aggregator.search_all(
        query=args.query,
        max_results=args.max_results,
        oversample_ratio=args.oversample_ratio
    )
    
    if not papers:
        print("[错误] 未抓取到任何论文")
        sys.exit(1)
    
    # 统计数据源分布
    source_stats = aggregator.get_source_stats(papers)
    print(f"\n数据源分布:")
    for source, count in source_stats.items():
        print(f"  {source:15s}: {count:3d} 篇 ({count/len(papers)*100:.1f}%)")
    
    # 转换为 DataFrame
    df = papers_to_dataframe(papers)
    
    # 保存 CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"papers_{args.query.replace(' ', '_')}_{timestamp}.csv"
    csv_path = output_dir / csv_filename
    
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] CSV 已保存: {csv_path}")
    
    # 生成向量（可选）
    if args.generate_embeddings:
        print(f"\n开始生成向量...")
        print(f"模型: {args.embedding_model}")
        
        generator = EmbeddingGenerator(
            model_name=args.embedding_model,
            batch_size=64
        )
        
        embeddings = generator.encode_texts(
            [paper.abstract for paper in papers],
            show_progress=True
        )
        
        # 保存向量
        embeddings_dir = Path(PROCESSED_DATA_DIR)
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = embeddings_dir / f"embeddings_{args.query.replace(' ', '_')}_{timestamp}.npy"
        
        generator.save_embeddings(embeddings, embeddings_path)
        print(f"[OK] 向量已保存: {embeddings_path}")
        
        # 写入 ChromaDB
        print(f"\n写入 ChromaDB 缓存...")
        chroma_store = ChromaStore(
            persist_directory=CHROMA_DB_DIR,
            collection_name="papers"
        )
        
        new_count = chroma_store.upsert_papers(papers, embeddings)
        print(f"[OK] 新增 {new_count} 篇论文到 ChromaDB")
        
        # 显示统计
        stats = chroma_store.get_stats()
        print(f"\nChromaDB 统计:")
        print(f"  总论文数: {stats['total_papers']}")
        print(f"  数据源分布: {stats['source_distribution']}")
    
    print("\n" + "=" * 70)
    print("抓取完成！")
    print("=" * 70)
    print(f"论文数量: {len(papers)}")
    print(f"CSV 文件: {csv_path}")
    if args.generate_embeddings:
        print(f"向量文件: {embeddings_path}")
        print(f"ChromaDB: {CHROMA_DB_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
