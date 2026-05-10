"""
测试2：检查数据文件
位置：D:\dataSciencePro\MscDSPro\AI_Assistant\test_data.py
"""
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    print("=" * 60)
    print("数据文件检查")
    print("=" * 60)

    # 项目根目录
    root_dir = Path(__file__).parent.parent
    print(f"项目根目录: {root_dir}")

    # 1. 检查原始数据
    print("\n1. 原始论文数据:")
    raw_dir = root_dir / "data" / "raw"
    if raw_dir.exists():
        csv_files = list(raw_dir.glob("*.csv"))
        print(f"   📁 目录: {raw_dir}")
        print(f"   📊 CSV文件数: {len(csv_files)}")

        for csv_file in csv_files:
            size = csv_file.stat().st_size / 1024  # KB
            df = pd.read_csv(csv_file)
            print(f"   📄 {csv_file.name}: {size:.1f}KB, {len(df)}篇论文")
    else:
        print(f"   ❌ 目录不存在: {raw_dir}")

    # 2. 检查处理后的数据
    print("\n2. 处理后的数据:")
    processed_dir = root_dir / "data" / "processed"
    if processed_dir.exists():
        npy_files = list(processed_dir.glob("*.npy"))
        print(f"   📁 目录: {processed_dir}")
        print(f"   📊 NPY文件数: {len(npy_files)}")

        for npy_file in npy_files:
            size = npy_file.stat().st_size / 1024  # KB
            try:
                data = np.load(npy_file)
                print(f"   📄 {npy_file.name}: {size:.1f}KB, 形状{data.shape}")
            except Exception as e:
                print(f"   ❌ {npy_file.name}: 读取失败 - {e}")
    else:
        print(f"   ❌ 目录不存在: {processed_dir}")

    # 3. 检查向量文件
    print("\n3. 向量文件验证:")
    embeddings_path = processed_dir / "embeddings.npy"
    if embeddings_path.exists():
        try:
            embeddings = np.load(embeddings_path)
            print(f"   ✅ embeddings.npy 存在")
            print(f"      形状: {embeddings.shape}")
            print(f"      论文数: {embeddings.shape[0]}")
            print(f"      向量维度: {embeddings.shape[1]}")
            print(f"      数据类型: {embeddings.dtype}")

            # 检查向量是否有效
            if np.all(embeddings == 0):
                print("   ⚠️  警告: 所有向量为零！")
            else:
                print(f"      向量范围: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        except Exception as e:
            print(f"   ❌ 读取失败: {e}")
    else:
        print("   ❌ embeddings.npy 不存在")


if __name__ == "__main__":
    main()