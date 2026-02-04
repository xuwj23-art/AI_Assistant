from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    """
    根路径，返回欢迎信息
    """
    return {"message":"Hello, FastAPI!"}

@app.get("/hello/{name}")
def say_hello(name: str):
    """
    向指定的人打招呼
    参数:
        name:要打招呼的人的名字
    """
    return {"message":f"Hello, {name}!"}

@app.get("/add/{a}/{b}")
def add_numbers(a: int,b: int):
    return {"result": a+b}

@app.get("/search")
def search_papers(
    query: str,
    max_results: int = 10
):
    """
    搜索论文
    参数:
        query: 搜索关键词
        max_results: 最多返回数量(默认10)
    """

    return {
        "query":query,
        "max_results":max_results,
        "papers": []
    }

from pydantic import BaseModel
from typing import List

class PaperRequest(BaseModel):
    """论文搜索请求"""
    query: str
    max_results: int = 10
    fields: List[str] = ["title", "authors"]

class PaperResponse(BaseModel):
    """论文信息"""
    id: str
    title: str
    authors: List[str]

@app.post("/papers/search",
response_model=List[PaperResponse])
def search_papers_V2(request: PaperRequest):
    """
    搜索论文(v2，使用模型)
    参数:
        request:搜索请求
    """

    return [
        PaperResponse(
            id="2301.00001",
            title="Example Paper",
            authors=["Alice", "Bob"]
        )
    ]