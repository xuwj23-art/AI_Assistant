"""
arXiv 数据处理管道

负责将 ArxivPaper 列表转换为 pandas DataFrame,并提供数据保存/加载功能。

为什么需要这个模块?
- 分离关注点: client.py 负责获取数据, pipeline.py 负责数据转换和持久化
- 复用性: DataFrame 转换逻辑可以被多个脚本/Notebook 复用
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .models import ArxivPaper


def papers_to_dataframe(papers: List[ArxivPaper]) -> pd.DataFrame:
    """
    将 ArxivPaper 列表转换为 pandas DataFrame。

    为什么用 DataFrame?
    ------------------
    1. 统一的表格格式,便于查看和分析
    2. 丰富的数据操作接口 (筛选、排序、分组、统计等)
    3. 与 BERTopic、sklearn 等库无缝集成
    4. 支持多种格式导出 (CSV、Parquet、Excel 等)

    实现细节:
    ---------
    - 使用 model_dump(): Pydantic 2.x 的标准方法,将模型转为字典
    - from_records(): 高效地从字典列表构造 DataFrame
    """
    # 将每个 ArxivPaper 对象转换为字典
    records = [paper.model_dump() for paper in papers]

    # 从字典列表构造 DataFrame
    df = pd.DataFrame.from_records(records)

    return df


def save_dataframe_to_csv(df: pd.DataFrame, path: Path) -> None:
    """
    将 DataFrame 保存为 CSV 文件,确保上级目录存在。

    参数:
    -----
    df : pd.DataFrame
        要保存的数据
    path : Path
        输出文件路径 (推荐使用 pathlib.Path 而非字符串)

    注意事项:
    ---------
    - 使用 Path.parent.mkdir() 自动创建目录,避免 FileNotFoundError
    - index=False: 不保存 DataFrame 的行索引,让 CSV 更干净
    """
    # 确保父目录存在 (如 data/raw/)
    # parents=True: 递归创建多级目录
    # exist_ok=True: 如果目录已存在,不报错
    path.parent.mkdir(parents=True, exist_ok=True)

    # 保存为 CSV
    df.to_csv(path, index=False, encoding="utf-8")


def load_dataframe_from_csv(path: Path) -> pd.DataFrame:
    """
    从 CSV 文件加载 DataFrame。

    如果文件不存在,会抛出 FileNotFoundError。
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path, encoding="utf-8")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 DataFrame 进行基础清洗。

    清洗步骤:
    1. 去除重复论文 (根据 id 去重)
    2. 删除摘要过短的论文 (可能是数据错误)
    3. 重置索引

    为什么需要清洗?
    --------------
    - arXiv API 可能返回重复数据
    - 有些论文的 abstract 异常短或为空,会影响聚类质量
    """
    # 1. 去重 (保留第一次出现的)
    df = df.drop_duplicates(subset=["id"], keep="first")

    # 2. 过滤过短的摘要 (少于 50 个字符的可能是异常数据)
    df = df[df["abstract"].str.len() >= 50]

    # 3. 重置索引
    df = df.reset_index(drop=True)

    return df

