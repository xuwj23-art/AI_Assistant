"""
研究趋势分析模块

功能：
- 按年份统计各主题论文数量
- 找出近两年增长最快的主题
- 生成趋势数据供前端可视化
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np


def compute_topic_trends(
    papers: List[dict],
    topic_names: Optional[Dict[int, str]] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    按年份统计各主题的论文数量

    参数:
        papers: 论文列表，每篇论文为 dict，需包含 'published_date'（或 'published'）和 'topic_id' 字段
        topic_names: topic_id → 主题名称 的映射字典（可选）
        min_year: 起始年份（可选，默认自动检测）
        max_year: 结束年份（可选，默认当前年份）

    返回:
        DataFrame，行=年份，列=主题名称，值=论文数量
        示例:
                NLP    CV    RL
        2020     28    15    10
        2021     45    20    18
        2022     78    22    25
    """
    if not papers:
        return pd.DataFrame()

    records = []
    for paper in papers:
        # 兼容多种日期字段名
        date_val = paper.get("published_date") or paper.get("published")
        topic_id = paper.get("topic_id", -1)

        # 跳过噪声主题
        if topic_id == -1:
            continue

        # 解析年份
        year = _extract_year(date_val)
        if year is None:
            continue

        # 获取主题名称
        if topic_names and topic_id in topic_names:
            topic_label = topic_names[topic_id]
        else:
            topic_label = f"Topic {topic_id}"

        records.append({"year": year, "topic": topic_label})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # 确定年份范围
    actual_min = int(df["year"].min())
    actual_max = int(df["year"].max())
    if min_year is None:
        min_year = actual_min
    if max_year is None:
        max_year = max(actual_max, datetime.now().year)

    # 统计每年每主题的论文数
    trend_df = (
        df.groupby(["year", "topic"])
        .size()
        .reset_index(name="count")
        .pivot(index="year", columns="topic", values="count")
        .fillna(0)
        .astype(int)
    )

    # 补全缺失年份（填 0）
    all_years = list(range(min_year, max_year + 1))
    trend_df = trend_df.reindex(all_years, fill_value=0)
    trend_df.index.name = "year"

    return trend_df


def get_trending_topics(
    trend_df: pd.DataFrame,
    top_n: int = 5,
    recent_years: int = 2,
    baseline_years: int = 3,
) -> List[Tuple[str, float]]:
    """
    找出近期增长最快的主题

    算法:
        增长率 = (最近 recent_years 年平均论文数 + 1) / (前 baseline_years 年平均论文数 + 1)
        （+1 避免除以零）

    参数:
        trend_df: compute_topic_trends() 返回的 DataFrame
        top_n: 返回前 N 个主题
        recent_years: 近期年份窗口（默认 2 年）
        baseline_years: 基准年份窗口（默认 3 年）

    返回:
        [(主题名, 增长率), ...] 按增长率降序排列
    """
    if trend_df.empty:
        return []

    years = sorted(trend_df.index.tolist())
    if len(years) < 2:
        return []

    # 近期年份
    recent = years[-recent_years:]
    # 基准年份（近期之前的 baseline_years 年）
    baseline_end = years[-(recent_years + 1)] if len(years) > recent_years else years[0]
    baseline_start_idx = max(0, len(years) - recent_years - baseline_years)
    baseline = years[baseline_start_idx : len(years) - recent_years]

    growth_rates = {}
    for topic in trend_df.columns:
        recent_avg = trend_df.loc[trend_df.index.isin(recent), topic].mean()
        baseline_avg = (
            trend_df.loc[trend_df.index.isin(baseline), topic].mean()
            if baseline
            else 0.0
        )
        growth_rate = (recent_avg + 1) / (baseline_avg + 1)
        growth_rates[topic] = round(float(growth_rate), 3)

    # 按增长率降序排列
    sorted_topics = sorted(growth_rates.items(), key=lambda x: x[1], reverse=True)
    return sorted_topics[:top_n]


def trend_df_to_api_format(trend_df: pd.DataFrame) -> dict:
    """
    将趋势 DataFrame 转换为 API 返回格式

    返回格式:
    {
        "years": [2018, 2019, 2020, 2021, 2022, 2023],
        "topics": [
            {
                "name": "Transformer Architecture",
                "counts": [5, 12, 28, 45, 78, 120]
            },
            ...
        ]
    }
    """
    if trend_df.empty:
        return {"years": [], "topics": []}

    years = [int(y) for y in trend_df.index.tolist()]
    topics = []
    for col in trend_df.columns:
        topics.append({
            "name": str(col),
            "counts": [int(v) for v in trend_df[col].tolist()]
        })

    return {"years": years, "topics": topics}


def compute_trends_from_dataframe(
    df: pd.DataFrame,
    topic_id_col: str = "topic_id",
    topic_name_col: str = "topic_name",
    date_col: str = "published",
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    直接从 pandas DataFrame 计算趋势（适配 TopicService 的数据格式）

    参数:
        df: 包含论文数据的 DataFrame
        topic_id_col: 主题 ID 列名
        topic_name_col: 主题名称列名
        date_col: 日期列名

    返回:
        (trend_df, topic_names_dict)
    """
    if df.empty:
        return pd.DataFrame(), {}

    # 构建 topic_names 映射
    topic_names: Dict[int, str] = {}
    if topic_name_col in df.columns and topic_id_col in df.columns:
        for _, row in df[[topic_id_col, topic_name_col]].drop_duplicates().iterrows():
            tid = int(row[topic_id_col])
            tname = str(row[topic_name_col])
            if tid != -1:
                topic_names[tid] = tname

    # 转换为 papers 列表格式
    papers = []
    for _, row in df.iterrows():
        topic_id = row.get(topic_id_col, -1)
        if pd.isna(topic_id):
            topic_id = -1
        date_val = row.get(date_col)
        papers.append({
            "topic_id": int(topic_id),
            "published_date": date_val,
        })

    trend_df = compute_topic_trends(papers, topic_names=topic_names)
    return trend_df, topic_names


def _extract_year(date_val) -> Optional[int]:
    """
    从各种格式的日期值中提取年份

    支持:
        - datetime 对象
        - pandas Timestamp
        - 字符串 "2023-01-15", "2023-01-15T00:00:00", "2023"
        - 整数 2023
    """
    if date_val is None:
        return None

    # pandas NaT
    try:
        if pd.isna(date_val):
            return None
    except (TypeError, ValueError):
        pass

    # datetime / Timestamp
    if hasattr(date_val, "year"):
        return int(date_val.year)

    # 整数
    if isinstance(date_val, (int, float)):
        year = int(date_val)
        if 1900 <= year <= 2100:
            return year
        return None

    # 字符串
    if isinstance(date_val, str):
        date_str = date_val.strip()
        if not date_str:
            return None
        # 尝试解析前4位作为年份
        try:
            year = int(date_str[:4])
            if 1900 <= year <= 2100:
                return year
        except (ValueError, IndexError):
            pass

    return None
