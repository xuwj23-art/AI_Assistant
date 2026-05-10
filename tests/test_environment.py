"""
测试1：检查环境和依赖
位置：D:\dataSciencePro\MscDSPro\AI_Assistant\test_environment.py
"""
import sys
import importlib


def check_package(package_name):
    """检查包是否安装"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name:20} 版本: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20} 未安装 - {e}")
        return False


def main():
    print("=" * 60)
    print("环境检查")
    print("=" * 60)

    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print()

    # 检查核心依赖
    packages = [
        "torch",
        "sentence_transformers",
        "transformers",
        "faiss",
        "numpy",
        "pandas",
        "sklearn",
        "fastapi",
        "uvicorn"
    ]

    results = []
    for pkg in packages:
        results.append(check_package(pkg))

    print("\n" + "=" * 60)
    print(f"总结: {sum(results)}/{len(results)} 个包安装成功")
    print("=" * 60)

    # 特别检查 PyTorch
    if check_package("torch"):
        import torch
        print(f"\nPyTorch CUDA可用: {torch.cuda.is_available()}")
        print(f"PyTorch 设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")


if __name__ == "__main__":
    main()