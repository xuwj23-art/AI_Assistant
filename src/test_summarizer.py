"""
测试AI摘要功能
位置：D:\dataSciencePro\MscDSPro\AI_Assistant\test_summarizer.py
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_summarizer():
    print("=" * 60)
    print("AI摘要功能测试")
    print("=" * 60)

    try:
        from core.nlp.summarizer import create_summarizer, PaperSummarizer, LightweightSummarizer

        # 测试文本
        test_text = """
        Transformer models have revolutionized natural language processing and computer vision.
        The key innovation is the self-attention mechanism, which allows the model to weigh the importance of different parts of the input.
        This enables parallel processing and captures long-range dependencies effectively.
        BERT, GPT, and other models build upon this architecture to achieve state-of-the-art results.
        """

        print("\n1. 测试轻量级摘要器 (t5-small)...")
        summarizer = create_summarizer("light")
        summary = summarizer.summarize(test_text)
        print(f"   原文: {test_text[:100]}...")
        print(f"   摘要: {summary}")

        print("\n✅ 摘要功能测试完成！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_integration():
    """测试与RAG的集成"""
    print("\n" + "=" * 60)
    print("RAG+摘要集成测试")
    print("=" * 60)

    try:
        from core.nlp.rag import RAGService
        import pandas as pd

        # 加载数据
        csv_path = project_root / "data" / "raw" / "arxiv_Transformer.csv"
        df = pd.read_csv(csv_path)

        # 初始化带摘要的RAG服务
        rag = RAGService(
            papers_df=df,
            enable_summary=True,
            summary_model="light"
        )

        # 测试搜索和摘要
        query = "transformer attention"
        results = rag.search_similar_papers(query, top_k=2)
        papers = [p for p, _ in results]

        # 生成带摘要的回答
        response = rag.generate_answer_with_summary(query, papers, include_summary=True)

        print(f"\n问题: {query}")
        print(f"\n回答:\n{response['answer']}")
        print(f"\n来源论文:")
        for source in response['sources']:
            print(f"  - {source['title']}")
            if source.get('ai_summary'):
                print(f"    AI总结: {source['ai_summary'][:100]}...")

        print("\n✅ 集成测试完成！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_summarizer()
    test_integration()