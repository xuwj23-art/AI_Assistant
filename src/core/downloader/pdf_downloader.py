"""
PDF 下载器

提供两个能力:
1. 通过 paper_id（arxiv_id / openalex_id / 行 id）反查 PDF URL
2. 下载并缓存 PDF（默认存到 data/pdfs/），返回本地文件路径供 FastAPI FileResponse 使用

设计要点:
- 依赖现有 httpx (已在 requirements 中)，无新增依赖
- 缓存命中直接返回，避免重复下载
- 限制单文件大小（默认 50MB）防止恶意 URL 撑爆磁盘
"""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
from typing import Optional, Tuple
import httpx
import pandas as pd

from ..config import DATA_DIR
from ..auto_fetch import get_latest_data_paths
from ..api.paper_utils import _normalize_arxiv_id, _resolve_pdf_url


PDF_CACHE_DIR: Path = DATA_DIR / "pdfs"
PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class PDFDownloader:
    """PDF 下载与本地缓存"""

    def __init__(
        self,
        cache_dir: Path = PDF_CACHE_DIR,
        max_bytes: int = 50 * 1024 * 1024,
        timeout: int = 60,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.timeout = timeout

    def _cache_path_for(self, paper_id: str) -> Path:
        """根据 paper_id 生成稳定的缓存文件名"""
        safe = re.sub(r"[^\w\.-]", "_", paper_id)[:80]
        digest = hashlib.md5(paper_id.encode()).hexdigest()[:8]
        return self.cache_dir / f"{safe}_{digest}.pdf"

    def resolve_pdf_url(self, paper_id: str) -> Tuple[Optional[str], Optional[pd.Series]]:
        """
        通过 paper_id 反查 pdf_url 与 DataFrame 行

        匹配优先级:
            1. 完整 id 列精确匹配
            2. arxiv_id 列精确匹配（裸 ID）
            3. openalex_id 列精确匹配
            4. id 列模糊包含
        """
        paths = get_latest_data_paths()
        if not paths:
            return None, None

        csv_path = paths.get("topics_csv") or paths.get("csv")
        if not csv_path or not Path(csv_path).exists():
            return None, None

        df = pd.read_csv(csv_path)
        if df.empty:
            return None, None

        bare = _normalize_arxiv_id(paper_id)

        row: Optional[pd.Series] = None

        if 'id' in df.columns:
            mask = df['id'].astype(str) == paper_id
            if mask.any():
                row = df[mask].iloc[0]

        if row is None and 'arxiv_id' in df.columns and bare:
            mask = df['arxiv_id'].astype(str) == bare
            if mask.any():
                row = df[mask].iloc[0]

        if row is None and 'openalex_id' in df.columns:
            mask = df['openalex_id'].astype(str) == paper_id
            if mask.any():
                row = df[mask].iloc[0]

        if row is None and 'id' in df.columns and bare:
            mask = df['id'].astype(str).str.contains(bare, regex=False, na=False)
            if mask.any():
                row = df[mask].iloc[0]

        if row is None:
            return None, None

        pdf_url = _resolve_pdf_url(row)
        return pdf_url, row

    def download(self, paper_id: str) -> Tuple[Optional[Path], Optional[str]]:
        """
        下载 PDF 到本地缓存

        返回:
            (local_path, original_pdf_url)；任一步失败则 path 为 None
        """
        pdf_url, _ = self.resolve_pdf_url(paper_id)
        if not pdf_url:
            return None, None

        cache_file = self._cache_path_for(paper_id)
        if cache_file.exists() and cache_file.stat().st_size > 0:
            return cache_file, pdf_url

        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AIassistant/1.0)"},
            ) as client:
                with client.stream("GET", pdf_url) as resp:
                    resp.raise_for_status()
                    total = 0
                    with open(cache_file, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                            if not chunk:
                                continue
                            total += len(chunk)
                            if total > self.max_bytes:
                                f.close()
                                cache_file.unlink(missing_ok=True)
                                raise RuntimeError(
                                    f"PDF 超过限制 {self.max_bytes} 字节"
                                )
                            f.write(chunk)
            return cache_file, pdf_url
        except Exception as e:
            print(f"[PDFDownloader] 下载失败: {e}")
            cache_file.unlink(missing_ok=True)
            return None, pdf_url


_downloader: Optional[PDFDownloader] = None


def get_pdf_downloader() -> PDFDownloader:
    global _downloader
    if _downloader is None:
        _downloader = PDFDownloader()
    return _downloader
