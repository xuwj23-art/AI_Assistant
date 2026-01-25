"""
可执行脚本: 从 arXiv 抓取论文并保存为 CSV

使用方式:
--------
cd src
python -m scripts.fetch_and_save_arxiv --query "large language models" --max-results 100

设计思路:
--------
这是一个"命令行工具",类似于 Linux 中的可执行程序。
它封装了 core/ 模块的功能,让用户无需写代码即可使用。

工程化要点:
----------
1. 使用 argparse 处理命令行参数 (而非硬编码)
2. 提供清晰的帮助信息 (--help)
3. 打印进度日志,方便用户了解执行状态
4. 异常处理,避免程序崩溃
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.arxiv.client import fetch_arxiv_papers
from core.arxiv.pipeline import papers_to_dataframe, save_dataframe_to_csv, clean_dataframe


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    类比:
    -----
    这类似于 C 语言中的 getopt(),用于解析 main(int argc, char *argv[])。

    Argparse 的优势:
    ---------------
    1. 自动生成 --help 帮助文档
    2. 自动做类型转换和验证
    3. 支持默认值、必填项、值范围检查等
    """
    parser = argparse.ArgumentParser(
        description="从 arXiv 抓取论文并保存为 CSV。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python -m scripts.fetch_and_save_arxiv --query "large language models" --max-results 100
  python -m scripts.fetch_and_save_arxiv --query "PEFT" --max-results 200 --output-dir data/raw
        """,
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="arXiv 检索关键词,例如 'large language models'",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=200,
        help="最多抓取多少篇论文 (默认: 200)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="CSV 文件保存目录 (默认: data/raw)",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="是否对数据进行清洗 (去重、过滤短摘要)",
    )

    return parser.parse_args()


def main() -> None:
    """
    主函数: 执行完整的抓取 → 转换 → 保存流程。
    """
    args = parse_args()

    query = args.query
    max_results = args.max_results
    output_dir = Path(args.output_dir)
    do_clean = args.clean

    print("=" * 60)
    print("arXiv 论文抓取工具")
    print("=" * 60)
    print(f"检索关键词: {query}")
    print(f"最大结果数: {max_results}")
    print(f"输出目录: {output_dir}")
    print(f"数据清洗: {'是' if do_clean else '否'}")
    print("=" * 60)

    # Step 1: 从 arXiv 抓取数据
    print("\n[1/4] 正在从 arXiv 抓取论文...")
    try:
        papers = fetch_arxiv_papers(query=query, max_results=max_results)
        print(f"✓ 成功抓取 {len(papers)} 篇论文")
    except Exception as e:
        print(f"✗ 抓取失败: {e}")
        return

    if len(papers) == 0:
        print("⚠ 未找到任何论文,请检查关键词是否正确。")
        return

    # Step 2: 转换为 DataFrame
    print("\n[2/4] 正在转换为 DataFrame...")
    df = papers_to_dataframe(papers)
    print(f"✓ DataFrame 形状: {df.shape} (行数, 列数)")

    # Step 3: 可选的数据清洗
    if do_clean:
        print("\n[3/4] 正在清洗数据...")
        original_len = len(df)
        df = clean_dataframe(df)
        removed = original_len - len(df)
        print(f"✓ 删除了 {removed} 条重复或异常数据,剩余 {len(df)} 条")
    else:
        print("\n[3/4] 跳过数据清洗 (使用 --clean 启用)")

    # Step 4: 保存到 CSV
    print("\n[4/4] 正在保存到 CSV...")
    # 将关键词中的空格替换为下划线,避免文件名问题
    safe_query = query.replace(" ", "_").replace("/", "_")
    output_path = output_dir / f"arxiv_{safe_query}.csv"

    try:
        save_dataframe_to_csv(df, output_path)
        print(f"✓ 已保存到: {output_path.resolve()}")
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        return

    # 显示简单统计
    print("\n" + "=" * 60)
    print("统计信息:")
    print("=" * 60)
    print(f"论文总数: {len(df)}")
    print(f"作者数量: {df['authors'].apply(len).sum()}")
    print(f"发布年份范围: {df['published'].min()} 至 {df['published'].max()}")
    print(f"平均摘要长度: {df['abstract'].str.len().mean():.0f} 字符")
    print("=" * 60)
    print("\n✓ 完成! 你现在可以:")
    print(f"  1. 在 Excel/VSCode 中打开 {output_path.name} 查看数据")
    print(f"  2. 在 Jupyter Notebook 中加载: pd.read_csv('{output_path}')")
    print(f"  3. 进入 Phase 2,对这些论文做 BERTopic 聚类\n")


if __name__ == "__main__":
    main()

