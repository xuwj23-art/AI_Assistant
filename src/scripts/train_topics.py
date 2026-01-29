from __future__ import annotations
import argparse
from pathlib import Path
import sys
# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.nlp.topic_modeling import train_topic_model_for_papers


def main():
    parser = argparse.ArgumentParser(
        description="为论文训练主题模型"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="论文 CSV 文件路径"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="向量文件路径（.npy）"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="模型保存路径"
    )
    parser.add_argument(
        "--column",
        type=str,
        default="abstract",
        help="文本列名"
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=10,
        help="最小主题大小"
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=None,
        help="目标主题数（可选）"
    )
    
    args = parser.parse_args()
    
    # 转换为 Path
    csv_path = Path(args.csv)
    embeddings_path = Path(args.embeddings)
    output_path = Path(args.output)
    
    # 验证输入
    if not csv_path.exists():
        print(f"[ERROR] CSV 文件不存在: {csv_path}")
        sys.exit(1)
    
    if not embeddings_path.exists():
        print(f"[ERROR] 向量文件不存在: {embeddings_path}")
        sys.exit(1)
    
    # 训练
    print("=" * 60)
    print("开始训练主题模型")
    print("=" * 60)
    print(f"CSV 文件: {csv_path}")
    print(f"向量文件: {embeddings_path}")
    print(f"输出路径: {output_path}")
    print(f"文本列: {args.column}")
    print(f"最小主题大小: {args.min_topic_size}")
    if args.n_topics:
        print(f"目标主题数: {args.n_topics}")
    print("=" * 60)
    
    try:
        modeler, df = train_topic_model_for_papers(
            csv_path=csv_path,
            embeddings_path=embeddings_path,
            output_path=output_path,
            text_column=args.column,
            min_topic_size=args.min_topic_size,
            n_topics=args.n_topics
        )
        
        # 显示主题信息
        print("\n" + "=" * 60)
        print("主题信息")
        print("=" * 60)
        topic_info = modeler.get_topic_info()
        print(topic_info.to_string())
        
        # 显示每个主题的关键词
        print("\n" + "=" * 60)
        print("主题关键词")
        print("=" * 60)
        for topic_id in topic_info['Topic'].head(10):
            if topic_id == -1:
                continue
            keywords = modeler.get_topic(topic_id, top_n=5)
            words = ", ".join([f"{word}({score:.3f})" for word, score in keywords])
            print(f"主题 {topic_id}: {words}")
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()