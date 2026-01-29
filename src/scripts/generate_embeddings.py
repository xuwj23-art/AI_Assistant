from __future__ import annotations
import argparse
from pathlib import Path
import sys

from core.nlp.embeddings import create_embeddings_for_papers
from core.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR

def main():
    parser = argparse.ArgumentParser(
        description="为论文摘要生成向量"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入CSV文件路径"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出向量文件路径"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all-mpnet-base-V2",
        help="Sentence Transformer 模型名称"
    )

    parser.add_argument(
        "--column",
        type=str,
        default="abstract",
        help="需要向量化的列名"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="批处理大小"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[error] 输入文件不存在：{input_path}")
        sys.exit(1)

    # 生成向量
    print("=" * 60)
    print("开始生成向量")
    print("=" * 60)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"模型: {args.model}")
    print(f"列名: {args.column}")
    print(f"批处理大小: {args.batch_size}")
    print("=" * 60)

    try:
        embeddings = create_embeddings_for_papers(
            csv_path=input_path,
            output_path=output_path,
            model_name=args.model,
            text_column=args.column,
            batch_size=args.batch_size
        )
        
        print("\n" + "=" * 60)
        print("向量生成完成！")
        print("=" * 60)
        print(f"向量形状: {embeddings.shape}")
        print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"\n[ERROR] 生成向量失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
