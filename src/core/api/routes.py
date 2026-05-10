"""
API 路由

定义所有 API 端点
"""
from fastapi import APIRouter, HTTPException, Query, Path, Depends
from fastapi.responses import FileResponse, RedirectResponse
from typing import List
from .models import PaperResponse, PaperListResponse, SearchRequest, ErrorResponse
from .services import PaperService
from .dependencies import get_paper_service


# 创建路由器
router = APIRouter(
    prefix="/api",
    tags=["论文管理"]
)


@router.get(
    "/papers",
    response_model=PaperListResponse,
    summary="获取论文列表",
    description="获取所有论文，支持分页"
)
def get_papers(
    limit: int = Query(10, ge=1, le=100, description="返回数量"),
    service: PaperService = Depends(get_paper_service)
):
    """
    获取论文列表
    
    参数:
        - limit: 最多返回数量（1-100）
    
    返回:
        - 论文列表和总数
    """
    papers = service.get_all_papers(limit=limit)
    total = len(papers)
    
    return PaperListResponse(
        total=total,
        papers=papers
    )


@router.get(
    "/papers/search",
    response_model=PaperListResponse,
    summary="搜索论文",
    description="根据关键词搜索论文"
)
def search_papers(
    query: str = Query(..., min_length=1, description="搜索关键词"),
    max_results: int = Query(10, ge=1, le=100, description="最多返回数量"),
    fields: str = Query("title,abstract", description="搜索字段（逗号分隔）"),
    service: PaperService = Depends(get_paper_service)
):
    """
    搜索论文
    
    参数:
        - query: 搜索关键词
        - max_results: 最多返回数量
        - fields: 搜索字段（如 "title,abstract"）
    
    返回:
        - 匹配的论文列表
    """
    # 解析字段
    field_list = [f.strip() for f in fields.split(",")]
    
    # 搜索
    papers = service.search_papers(
        query=query,
        fields=field_list,
        max_results=max_results
    )
    
    return PaperListResponse(
        total=len(papers),
        papers=papers
    )


@router.get(
    "/papers/{paper_id}",
    response_model=PaperResponse,
    summary="获取单个论文",
    description="根据 ID 获取论文详情",
    responses={
        404: {"model": ErrorResponse, "description": "论文不存在"}
    }
)
def get_paper(
    paper_id: str = Path(..., description="论文 ID"),
    service: PaperService = Depends(get_paper_service)
):
    """
    获取单个论文
    
    参数:
        - paper_id: 论文的 arXiv ID
    
    返回:
        - 论文详细信息
    
    错误:
        - 404: 论文不存在
    """
    paper = service.get_paper_by_id(paper_id)
    
    if paper is None:
        raise HTTPException(
            status_code=404,
            detail=f"Paper not found: {paper_id}"
        )
    
    return paper


@router.get(
    "/papers/{paper_id}/pdf",
    summary="下载论文 PDF",
    description=(
        "下载指定论文的 PDF 文件。\n\n"
        "- `mode=download`（默认）：服务器端代理下载并缓存到本地，再以附件形式返回\n"
        "- `mode=redirect`：直接 302 跳转到原始 PDF URL（适合不想代理的场景）\n"
    ),
    responses={
        404: {"model": ErrorResponse, "description": "论文或 PDF 链接不存在"},
        502: {"model": ErrorResponse, "description": "下载源失败"},
    },
)
def download_paper_pdf(
    paper_id: str = Path(..., description="论文 ID（arxiv_id / openalex_id / 完整 id）"),
    mode: str = Query("download", regex="^(download|redirect)$"),
):
    """
    下载论文 PDF。
    """
    from ..downloader import get_pdf_downloader
    downloader = get_pdf_downloader()

    pdf_url, _row = downloader.resolve_pdf_url(paper_id)
    if not pdf_url:
        raise HTTPException(
            status_code=404,
            detail=f"未找到论文 {paper_id} 或其 PDF 链接"
        )

    if mode == "redirect":
        return RedirectResponse(url=pdf_url, status_code=302)

    local_path, _ = downloader.download(paper_id)
    if local_path is None or not local_path.exists():
        raise HTTPException(
            status_code=502,
            detail=f"PDF 下载失败（URL: {pdf_url}）"
        )

    safe_name = (paper_id.replace('/', '_')[:60] or "paper") + ".pdf"
    return FileResponse(
        path=str(local_path),
        media_type="application/pdf",
        filename=safe_name,
    )


@router.get(
    "/stats",
    summary="获取统计信息",
    description="获取论文库的统计数据"
)
def get_stats(
    service: PaperService = Depends(get_paper_service)
):
    """
    获取统计信息
    
    返回:
        - 总论文数
        - 日期范围
        - 其他统计数据
    """
    return service.get_stats()