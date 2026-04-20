"""
测试4：RAG核心功能测试
位置：D:\dataSciencePro\MscDSPro\AI_Assistant\src\test_rag_core.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os

# ===== 设置环境变量，防止DLL冲突 =====
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# ========================================

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_rag_core():
    print("=" * 60)
    print("RAG核心功能测试")
    print("=" * 60)

    try:
        # 导入RAG模块
        from core.nlp.rag import RAGService
        from core.api.models import PaperResponse

        # 1. 准备数据
        print("\n1. 准备测试数据...")
        csv_path = project_root / "data" / "raw" / "arxiv_Transformer.csv"
        embeddings_path = project_root / "data" / "processed" / "embeddings.npy"

        if not csv_path.exists():
            print(f"❌ CSV文件不存在: {csv_path}")
            # 列出可用的CSV文件
            raw_dir = project_root / "data" / "raw"
            if raw_dir.exists():
                files = list(raw_dir.glob("*.csv"))
                print(f"可用的CSV文件: {[f.name for f in files]}")
            return

        df = pd.read_csv(csv_path)
        print(f"✅ 加载论文: {len(df)}篇")

        # 显示前3篇论文的标题，确认数据
        print("\n   前3篇论文标题:")
        for i, title in enumerate(df['title'].head(3)):
            print(f"   {i+1}. {title}")

        # 2. 初始化RAG服务
        print("\n2. 初始化RAG服务...")
        rag = RAGService(
            papers_df=df,
            embeddings_path=embeddings_path if embeddings_path.exists() else None
        )
        print("✅ RAG服务初始化成功")

        # 3. 测试相似论文搜索 - 多个查询词
        print("\n3. 测试相似论文搜索...")

        test_queries = [
            "transformer attention",
            "neural network",
            "deep learning",
            "bert language model",
            "注意力机制",  # 添加中文查询
            "attention mechanism"  # 添加英文查询
        ]

        for query in test_queries:
            print(f"\n   🔍 搜索: '{query}'")

            # 搜索论文，使用较低的阈值
            results = rag.search_similar_papers(query, top_k=5, threshold=0.1)
            print(f"      找到 {len(results)} 篇论文")

            # 显示搜索结果
            for paper, score in results:
                # 截断过长的标题
                title = paper.title if len(paper.title) <= 60 else paper.title[:57] + "..."
                print(f"      📄 [{score:.3f}] {title}")

                # 显示摘要片段（调试用）
                if score > 0.3:  # 只显示高分的摘要
                    abstract_preview = paper.abstract[:100] + "..." if len(paper.abstract) > 100 else paper.abstract
                    print(f"         摘要: {abstract_preview}")

        # 4. 测试答案生成 - 详细调试
        print("\n4. 测试答案生成...")

        test_questions = [
            "什么是transformer模型？",
            "注意力机制的原理是什么？",
            "BERT模型有什么特点？"
        ]

        for question in test_questions:
            print(f"\n   ❓ 问题: {question}")

            # 4.1 先搜索论文
            print(f"      步骤1: 搜索相关论文...")
            results = rag.search_similar_papers(question, top_k=5, threshold=0.15)
            print(f"      找到 {len(results)} 篇相关论文")

            # 4.2 显示搜索结果详情
            for i, (paper, score) in enumerate(results, 1):
                print(f"         论文{i}: 得分 {score:.3f} - {paper.title[:50]}...")

            # 4.3 提取论文列表
            papers = [p for p, _ in results]

            # 4.4 生成答案
            print(f"      步骤2: 生成答案...")
            answer_result = rag.generate_answer(question, papers)

            # 4.5 显示答案
            print(f"      步骤3: 答案生成完成")
            print(f"      📝 答案: {answer_result['answer'][:200]}...")
            print(f"      📚 来源: {len(answer_result['sources'])}篇")

            # 4.6 显示来源详情
            # 4.6 显示来源详情
            for i, source in enumerate(answer_result['sources'], 1):
                print(f"         来源{i}: {source['title']}")
                # 修复：检查relevance是否为None
                relevance = source.get('relevance')
                if relevance is not None:
                    print(f"            相关性: {relevance:.3f}")
                else:
                    print(f"            相关性: 未提供")

        # 5. 测试会话管理
        print("\n5. 测试会话管理...")

        # 创建多个会话
        session1 = rag.create_session()
        session2 = rag.create_session()
        print(f"   会话1: {session1}")
        print(f"   会话2: {session2}")

        # 添加历史记录
        rag.add_to_history(session1, "user", "测试问题1")
        rag.add_to_history(session1, "assistant", "测试回答1")
        rag.add_to_history(session2, "user", "测试问题2")

        # 获取历史
        history1 = rag.get_history(session1)
        history2 = rag.get_history(session2)

        print(f"   会话1历史: {len(history1)}条")
        for msg in history1:
            print(f"       [{msg['role']}] {msg['content'][:30]}...")

        print(f"   会话2历史: {len(history2)}条")
        for msg in history2:
            print(f"       [{msg['role']}] {msg['content'][:30]}...")

        # 6. 测试阈值对比
        print("\n6. 测试不同阈值的搜索结果对比...")

        test_query = "transformer"
        thresholds = [0.5, 0.3, 0.2, 0.1]

        for threshold in thresholds:
            results = rag.search_similar_papers(test_query, top_k=3, threshold=threshold)
            print(f"   阈值 {threshold}: 找到 {len(results)} 篇论文")
            if results:
                top_score = results[0][1]
                print(f"      最高分: {top_score:.3f}")

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)

        # 7. 返回rag实例供其他测试使用
        return rag

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_specific_queries():
    """测试特定查询，找出最佳阈值"""
    print("\n" + "=" * 60)
    print("特定查询测试")
    print("=" * 60)

    try:
        from core.nlp.rag import RAGService
        import pandas as pd

        # 加载数据
        csv_path = project_root / "data" / "raw" / "arxiv_Transformer.csv"
        embeddings_path = project_root / "data" / "processed" / "embeddings.npy"

        df = pd.read_csv(csv_path)
        rag = RAGService(
            papers_df=df,
            embeddings_path=embeddings_path if embeddings_path.exists() else None
        )

        # 测试注意力机制相关查询
        attention_queries = [
            "attention mechanism",
            "self-attention",
            "transformer attention",
            "注意力机制"
        ]

        for query in attention_queries:
            print(f"\n📊 查询: '{query}'")
            results = rag.search_similar_papers(query, top_k=5, threshold=0.0)

            if results:
                print(f"   找到 {len(results)} 篇论文")
                for paper, score in results[:3]:
                    print(f"   {score:.3f} - {paper.title[:60]}...")
            else:
                print("   没有找到相关论文")

    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    # 运行主测试
    rag_instance = test_rag_core()

    # 如果需要运行额外测试，取消下面的注释
    # test_specific_queries()

    print("\n" + "💡" * 20)
    print("提示: 如果搜索返回的论文太少，可以在 rag.py 中降低 threshold 值")
    print("当前 threshold = 0.5，建议改为 0.2 或 0.15")
    print("💡" * 20)