"""
论文行 → PaperResponse 通用转换工具

抽取自 PaperService 与 TopicService 的重复实现，避免：
1. 维护成本：两份相同逻辑容易不同步。
2. 循环依赖：RAGService 之前为复用此逻辑而临时实例化 PaperService("")。
"""
from __future__ import annotations

import ast
from datetime import datetime
from typing import Any, List

import pandas as pd

from .models import PaperResponse


def _parse_list_field(value: Any) -> List[str]:
    """安全解析 CSV 中以字符串形式存储的列表字段（如 authors / categories）。"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return parsed
            return [str(parsed)]
        except (ValueError, SyntaxError):
            return [s]
    return [str(value)]


def _normalize_arxiv_id(id_val: Any) -> str:
    """从可能的 entry_id URL 中提取裸 arXiv ID（去版本号），其他情况原样返回。"""
    if id_val is None or (isinstance(id_val, float) and pd.isna(id_val)):
        return ""
    s = str(id_val)
    if "arxiv.org/abs/" in s:
        s = s.split("arxiv.org/abs/")[-1]
        # 去掉末尾的版本号（如 v1, v2）
        if "v" in s:
            s = s.split("v")[0]
    return s


def _resolve_pdf_url(row: pd.Series):
    """优先使用 pdf_url 列；为空时尝试根据 arXiv url 推导出 PDF 链接。"""
    pdf_url = row.get("pdf_url")
    if pdf_url is None or (isinstance(pdf_url, float) and pd.isna(pdf_url)):
        pdf_url = ""
    pdf_url = str(pdf_url).strip()

    if pdf_url:
        return pdf_url

    url = row.get("url", "")
    if url and "arxiv.org/abs/" in str(url):
        return str(url).replace("/abs/", "/pdf/") + ".pdf"
    return None


def row_to_paper(row: pd.Series) -> PaperResponse:
    """
    将 papers DataFrame 的一行转换为 PaperResponse。

    兼容 Stage 1 旧 schema 与 Stage 2 新 schema（17 列），
    并处理 NaN / 类型不一致 / 列缺失等边界。
    """
    authors = _parse_list_field(row.get("authors", "[]"))
    categories = _parse_list_field(row.get("categories", "[]"))

    id_val = _normalize_arxiv_id(row.get("id", ""))
    pdf_url = _resolve_pdf_url(row)

    title_val = row.get("title", "")
    abstract_val = row.get("abstract", "")
    if pd.isna(title_val):
        title_val = ""
    if pd.isna(abstract_val):
        abstract_val = ""
    # PaperResponse 要求 title/abstract min_length=1，对空值兜底，避免少量脏行让整个请求 500
    title_str = str(title_val).strip() or "(no title)"
    abstract_str = str(abstract_val).strip() or "(no abstract)"

    published_val = row.get("published")
    if pd.isna(published_val):
        published_val = datetime(2000, 1, 1)

    # pdf_url 必须是合法 URL 或 None；任何非法字符串（如 "NaT"、"nan"）都置 None
    if pdf_url is not None:
        pdf_str = str(pdf_url).strip()
        if (
            not pdf_str
            or pdf_str.lower() in {"nan", "nat", "none"}
            or not (pdf_str.startswith("http://") or pdf_str.startswith("https://"))
        ):
            pdf_url = None

    return PaperResponse(
        id=str(id_val),
        title=title_str,
        authors=authors,
        abstract=abstract_str,
        published=published_val,
        pdf_url=pdf_url,
        categories=categories,
    )
