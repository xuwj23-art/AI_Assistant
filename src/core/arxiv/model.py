from pydantic import BaseModel, HttpUrl
from datetime import datetime
from typing import List,Optional


# class Author(BaseModel):
#     """ 定义作者的类，有可能是机构 """
#     name: str
#     affiliation: Optional[str] = None


class ArxivPaper(BaseModel):
    """ 定义从Arxiv抓取的论文格式 """
    id: str
    title: str
    abstract: str
    authors: List[str]
    published: datetime
    updated: datetime
    categories: List[str]
    url: HttpUrl
    pdf_url: Optional[HttpUrl] = None
    comment: Optional[str] = None
    
