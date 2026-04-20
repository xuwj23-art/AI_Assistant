"""
构建向量存储

为论文生成向量并创建FAISS索引
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.nlp.rag import RAGService
from core.nlp.embeddings import EmbeddingGenerator
from core.config import PROCESSED_DATA_DIR


def main():
    parser = argparse.ArgumentParser(
        description="构建向量存储（生成向量+创建FAISS索引）"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="论文CSV文件路径"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="输出目录"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all-mpnet-base-v2",
        help="Sentence Transformer模型名称"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新生成"
    )

    args = parser.parse_args()

    # 转换路径
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "embeddings.npy"
    index_path = output_dir / "faiss.index"

    # 检查是否已存在
    if not args.force:
        if embeddings_path.exists() and index_path.exists():
            print(f"向量文件已存在: {embeddings_path}")
            print(f"索引文件已存在: {index_path}")
            response = input("是否重新生成？(y/n): ")
            if response.lower() != 'y':
                print("已取消")
                return

    # 加载论文数据
    print("=" * 60)
    print("构建向量存储")
    print("=" * 60)
    print(f"CSV文件: {csv_path}")
    print(f"输出目录: {output_dir}")
    print(f"模型: {args.model}")
    print("=" * 60)

    try:
        # 加载数据
        print("\n加载论文数据...")
        df = pd.read_csv(csv_path, parse_dates=['published', 'updated'])
        print(f"已加载 {len(df)} 篇论文")

        # 创建RAG服务（会自动生成向量和索引）
        print("\n生成向量和创建索引...")
        rag = RAGService(
            papers_df=df,
            model_name=args.model
        )

        # 保存
        print("\n保存文件...")
        rag.save_embeddings(embeddings_path)
        rag.save_index(index_path)

        # 保存论文ID映射（可选）
        ids = df['id'].tolist()
        id_path = output_dir / "paper_ids.npy"
        np.save(id_path, ids)
        print(f"论文ID已保存: {id_path}")

        print("\n" + "=" * 60)
        print("构建完成！")
        print("=" * 60)
        print(f"向量维度: {rag.embeddings.shape[1]}")
        print(f"向量文件: {embeddings_path}")
        print(f"索引文件: {index_path}")
        print(f"向量数量: {len(rag.embeddings)}")

    except Exception as e:
        print(f"\n[ERROR] 构建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()