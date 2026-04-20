from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .model import ArxivPaper


def papers_to_dataframe(papers: List[ArxivPaper]) -> pd.DataFrame:
    """将 ArxivPaper 列表转换为 pandas DataFrame。

    参数:
        papers: ArxivPaper 对象列表

    返回:
        pandas DataFrame
    """
    records = [paper.model_dump() for paper in papers]
    df = pd.DataFrame.from_records(records)
    return df


def save_dataframe_to_csv(df: pd.DataFrame, path: Path) -> None:
    """将 DataFrame 保存为 CSV 文件，确保上级目录存在。

    参数:
        df: pandas DataFrame
        path: 保存路径（Path 对象）
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """清洗 DataFrame：去重、过滤、重置索引。

    参数:
        df: 原始 DataFrame

    返回:
        清洗后的 DataFrame
    """
    # 去重
    df = df.drop_duplicates(subset=["id"], keep="first")

    # 过滤短摘要
    df = df[df["abstract"].str.len() >= 50]

    # 重置索引
    df = df.reset_index(drop=True)

    return df