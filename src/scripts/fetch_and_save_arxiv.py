from __future__ import annotations
import argparse
from pathlib import Path
from core.arxiv.client import fetch_arxiv_papers
from core.config import RAW_DATA_DIR
from core.arxiv.pipeline import (
    papers_to_dataframe,
    save_dataframe_to_csv,
    clean_dataframe,
)

def parse_args() -> argparse.Namespace:
    """
    从用户命令行解析指令，返回Namespace打包对象
    返回的格式：
        Namespace(
        query='AI', 
        max-results=200,    # 默认值
        clean=False,        # 默认值
        output-dir='data/raw' # 默认值
        )  
    """
    parser = argparse.ArgumentParser(description="Arxiv抓取论文命令行脚本")

    parser.add_argument("--query", type=str,required=True,help="搜索关键词")
    parser.add_argument("--max-results", type=int,default=200, help="最多抓取多少篇")
    parser.add_argument("--clean",action="store_true", help="是否清洗数据（去重、过滤短摘要）")
    parser.add_argument("--output-dir",type=str,default="data/raw",help="输出目录（默认：data/raw）")

    # parser看到 --query，查蓝图(add_argument) -> 发现需要取下一个词 -> 拿到 "AI" -> 存入 query 字段
    return parser.parse_args()

def main() -> None:
    """
    主函数
    """

    # 解析参数
    args = parse_args()

    # 打印配置
    print("=" * 60)
    print("arXiv 论文抓取工具")
    print("=" * 60)
    print(f"关键字：{args.query}")
    print(f"最大数量：{args.max_results}")
    print(f"输出目录：{args.output_dir}")
    print(f"数据是否需要清洗: ({'是' if args.clean else '否'})")
    print("=" * 60)

    # 开始抓取
    print("\n正在从 arXiv 抓取论文...")
    papers = fetch_arxiv_papers(
        args.query,
        args.max_results
    )
    print(f"[success] 已经成功抓取{len(papers)}篇论文")

    # 转换为DF格式
    print("\n正在将原始文档转为dataframe格式...")
    df = papers_to_dataframe(papers)
    print(f"[success] 已经成功转换为DF格式:{df.shape}")

    # 数据清洗
    if args.clean:
        print("\n正在清洗数据...")
        print(f"清洗前共有 ({len(df)}) 行")
        df_clean = clean_dataframe(df)
        print(f"清洗后共有({len(df_clean)}) 行")
    else:
        df_clean = df
        print("\n 跳过数据清洗")


    # 使用配置文件
    if args.output_dir == "data/raw":
        # 使用默认配置
        output_dir = RAW_DATA_DIR
    else:
        # 使用用户指定的路径
        output_dir = Path(args.output_dir)
    
    #保存为csv
    print("\n正在保存为csv格式...")
    safe_name = args.query.replace(" ","_").replace("/","_")
    file_name = f"arxiv_{safe_name}.csv"
    file_path = Path(output_dir)/file_name
    save_dataframe_to_csv(df_clean,file_path)

    print(f"[success] 已经保存到：{file_path}")

if __name__ == "__main__":
    main()