"""
阶段2功能集成测试

测试完整链路:
1. 多源抓取（arXiv + OpenAlex）
2. 去重
3. ChromaDB 缓存
4. 向量化
5. 层级聚类

使用方法:
    cd src
    python test_stage2_integration.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from core.sources.arxiv_adapter import ArxivSource
from core.openalex.client import OpenAlexSource
from core.sources.aggregator import MultiSourceAggregator
from core.storage.chroma_store import ChromaStore
from core.nlp.embeddings import EmbeddingGenerator
from core.nlp.topic_modeling import TopicModeler
from core.config import CHROMA_DB_DIR
import numpy as np


def test_multi_source_fetch():
    """测试1: 多源抓取 + 去重"""
    print("\n" + "="*70)
    print("测试1: 多源抓取 + 去重")
    print("="*70)
    
    # 初始化数据源
    arxiv_source = ArxivSource()
    openalex_source = OpenAlexSource()
    
    # 创建聚合器
    aggregator = MultiSourceAggregator(
        sources=[arxiv_source, openalex_source],
        max_workers=2
    )
    
    # 抓取论文用于测试(至少50篇以满足BERTopic要求)
    papers = aggregator.search_all(
        query="transformer attention",
        max_results=50,
        oversample_ratio=1.5
    )
    
    assert len(papers) > 0, "未抓取到论文"
    assert len(papers) <= 50, "论文数量超过限制"
    
    # 检查去重
    unique_ids = set(paper.get_unique_id() for paper in papers)
    assert len(unique_ids) == len(papers), "存在重复论文"
    
    print(f"[OK] 测试通过: 抓取 {len(papers)} 篇论文，无重复")
    
    return papers


def test_chroma_cache(papers):
    """测试2: ChromaDB 缓存"""
    print("\n" + "="*70)
    print("测试2: ChromaDB 缓存")
    print("="*70)
    
    # 生成向量
    print("生成向量...")
    generator = EmbeddingGenerator(model_name="all-mpnet-base-v2")
    embeddings = generator.encode_texts(
        [paper.abstract for paper in papers],
        show_progress=False
    )
    
    assert embeddings.shape[0] == len(papers), "向量数量不匹配"
    
    # 写入 ChromaDB
    print("写入 ChromaDB...")
    chroma_store = ChromaStore(
        persist_directory=CHROMA_DB_DIR / "test",
        collection_name="test_papers"
    )
    
    new_count = chroma_store.upsert_papers(papers, embeddings)
    
    # 验证缓存
    total_count = chroma_store.count()
    assert total_count >= len(papers), "ChromaDB 论文数量不正确"
    
    # 测试缓存命中
    first_paper_id = papers[0].get_unique_id()
    assert chroma_store.is_cached(first_paper_id), "缓存命中检查失败"
    
    print(f"[OK] 测试通过: ChromaDB 存储 {total_count} 篇论文")
    
    # 清理测试数据
    chroma_store.reset()
    print("[OK] 测试数据已清理")
    
    return embeddings


def test_hierarchical_topics(papers, embeddings):
    """测试3: 层级主题聚类"""
    print("\n" + "="*70)
    print("测试3: 层级主题聚类")
    print("="*70)
    
    # 准备文档
    documents = [paper.abstract for paper in papers]
    
    # 训练主题模型
    print("训练主题模型...")
    topic_modeler = TopicModeler(
        min_topic_size=3,  # 测试数据少，降低最小主题大小
        verbose=False
    )
    
    topics, probs = topic_modeler.fit(documents, embeddings)
    
    assert len(topics) == len(papers), "主题数量不匹配"
    
    # 测试层级主题
    print("生成层级主题...")
    hierarchical_topics = topic_modeler.get_hierarchical_topics(documents)
    
    assert hierarchical_topics is not None, "层级主题生成失败"
    assert len(hierarchical_topics) > 0, "层级主题为空"
    
    print(f"[OK] 层级主题节点数: {len(hierarchical_topics)}")
    
    # 测试 JSON 导出
    print("导出层级 JSON...")
    hierarchy_json = topic_modeler.export_hierarchy_json(documents)
    
    assert "name" in hierarchy_json, "JSON 格式错误"
    assert "children" in hierarchy_json, "JSON 缺少 children"
    
    print(f"[OK] JSON 导出成功，根节点: {hierarchy_json['name']}")
    print(f"[OK] 子主题数: {len(hierarchy_json['children'])}")
    
    # 显示主题信息
    topic_info = topic_modeler.get_topic_info()
    print(f"\n主题统计:")
    print(f"  总主题数: {len(topic_info)}")
    print(f"  前3个主题:")
    for _, row in topic_info.head(3).iterrows():
        topic_id = row['Topic']
        if topic_id != -1:
            topic_name = topic_modeler.get_topic_name(topic_id, max_words=3)
            print(f"    Topic {topic_id}: {topic_name}")
    
    print(f"\n[OK] 测试通过: 层级主题聚类成功")


def main():
    print("\n" + "="*70)
    print("阶段2功能集成测试")
    print("="*70)
    print("测试内容:")
    print("  1. 多源抓取（arXiv + OpenAlex）+ 去重")
    print("  2. ChromaDB 向量缓存")
    print("  3. BERTopic 层级主题聚类")
    print("="*70)
    
    try:
        # 测试1: 多源抓取
        papers = test_multi_source_fetch()
        
        # 测试2: ChromaDB 缓存
        embeddings = test_chroma_cache(papers)
        
        # 测试3: 层级主题
        test_hierarchical_topics(papers, embeddings)
        
        print("\n" + "="*70)
        print("[OK] 所有测试通过！")
        print("="*70)
        print("\n阶段2核心功能已验证:")
        print("  [OK] 多源并行抓取")
        print("  [OK] 三级去重（DOI + arXiv ID + 标题）")
        print("  [OK] ChromaDB 持久化缓存")
        print("  [OK] BERTopic 层级主题")
        print("  [OK] JSON 导出（前端可用）")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n[FAIL] 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] 测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
