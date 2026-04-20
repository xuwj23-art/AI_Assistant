"""
FastAPI 应用入口
启动 API 服务
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 导入路由
from core.api.routes import router as paper_router
from core.api.topic_routes import router as topic_router
from core.api.rag_routes import router as rag_router

# 创建应用
app = FastAPI(
    title="论文助手API",
    description="""
    功能
    
    - 论文查询和管理
    - 主题发现和分析
    - 智能搜索
    - 统计分析
    - RAG智能问答 - 基于论文的对话系统
    
    使用方法
    
    1. 查看 /docs 获取完整文档
    2. 使用 /api/papers 获取论文列表
    3. 使用 /api/papers/search 搜索论文
    4. 使用 /api/topics 查看主题分类
    5. 使用 /api/chat 进行智能问答
    """,
    version="1.0.0"
)

# 配置 CORS（允许跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(paper_router)
app.include_router(topic_router)
app.include_router(rag_router)


# 根路径
@app.get("/", tags=["系统"])
def read_root():
    return {
        "name": "论文助手 API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "papers": "/api/papers",
            "search": "/api/papers/search",
            "stats": "/api/stats",
            "topics": "/api/topics",
            "chat": "/api/chat"
        }
    }


# 健康检查
@app.get("/health", tags=["系统"])
def health_check():
    return {"status": "healthy"}


# 调试端点
@app.get("/debug/routes", tags=["系统"])
def debug_routes():
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "methods": list(route.methods) if hasattr(route, 'methods') else []
        })
    return {"total_routes": len(routes), "routes": routes}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )