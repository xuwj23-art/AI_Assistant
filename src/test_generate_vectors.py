"""
测试3：测试向量生成
位置：D:\dataSciencePro\MscDSPro\AI_Assistant\test_generate_vectors.py
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_generate_vectors():
    print("=" * 60)
    print("向量生成测试")
    print("=" * 60)

    try:
        # 导入必要的模块
        from core.nlp.embeddings import EmbeddingGenerator
        import pandas as pd

        # 1. 加载数据
        print("\n1. 加载论文数据...")
        csv_path = project_root / "data" / "raw" / "arxiv_Transformer.csv"
        if not csv_path.exists():
            print(f"❌ 文件不存在: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        print(f"✅ 已加载 {len(df)} 篇论文")
        print(f"   列名: {list(df.columns)}")

        # 2. 初始化生成器
        print("\n2. 初始化向量生成器...")
        generator = EmbeddingGenerator(
            model_name="all-MiniLM-L6-v2",  # 使用小型模型加速测试
            batch_size=32
        )
        print("✅ 生成器初始化成功")

        # 3. 测试少量数据
        print("\n3. 测试生成向量（前5篇）...")
        test_df = df.head(5)
        embeddings = generator.encode_papers(test_df, show_progress=True)
        print(f"✅ 向量生成成功")
        print(f"   形状: {embeddings.shape}")
        print(f"   类型: {embeddings.dtype}")

        # 4. 保存测试
        print("\n4. 测试保存向量...")
        test_output = project_root / "data" / "processed" / "test_embeddings.npy"
        generator.save_embeddings(embeddings, test_output)
        print(f"✅ 保存成功: {test_output}")

        # 5. 加载测试
        print("\n5. 测试加载向量...")
        loaded = EmbeddingGenerator.load_embeddings(test_output)
        print(f"✅ 加载成功，形状: {loaded.shape}")

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_generate_vectors()