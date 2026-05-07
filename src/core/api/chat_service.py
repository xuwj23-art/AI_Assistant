"""
对话服务
"""
from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path

from ..nlp.rag import RAGService
from ..config import PROCESSED_DATA_DIR, RAW_DATA_DIR


class ChatService:
    """对话服务"""

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def init_session(self, session_id: Optional[str] = None, topic_id: Optional[int] = None) -> Dict[str, Any]:
        """初始化会话"""
        session_id = self.rag_service.create_session(session_id)

        self.sessions[session_id] = {
            "session_id": session_id,
            "topic_id": topic_id,
            "created_at": pd.Timestamp.now().isoformat(),
            "message_count": 0
        }

        return {
            "session_id": session_id,
            "topic_id": topic_id,
            "topic_name": f"Topic_{topic_id}" if topic_id else None,
            "message": "对话已初始化，您可以开始提问了"
        }

    def send_message(self, session_id: str, message: str, top_k: int = 5) -> Dict[str, Any]:
        """发送消息"""
        # 添加到历史
        self.rag_service.add_to_history(session_id, "user", message)

        # 搜索相关论文
        similar_papers = self.rag_service.search_similar_papers(query=message, top_k=top_k)
        papers = [p for p, _ in similar_papers]
        scores = [s for _, s in similar_papers]

        # 生成回答
        result = self.rag_service.generate_answer_with_summary(
            query=message,
            context_papers=papers,
            include_summary=True
        )

        # 添加相关性分数
        for i, source in enumerate(result["sources"]):
            if i < len(scores):
                source["relevance"] = scores[i]

        # 添加到历史
        self.rag_service.add_to_history(session_id, "assistant", result["answer"])

        # 更新统计
        if session_id in self.sessions:
            self.sessions[session_id]["message_count"] += 1

        return result

    def get_history(self, session_id: str) -> Dict[str, Any]:
        """获取历史"""
        return {
            "session_id": session_id,
            "history": self.rag_service.get_history(session_id)
        }

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        return self.sessions.get(session_id)


# 全局实例
_chat_service: Optional[ChatService] = None


def reset_chat_service():
    """重置对话服务（数据更新后调用，下次请求时重新初始化）"""
    global _chat_service
    _chat_service = None
    print("[ChatService] 已重置，下次请求时将重新加载数据")


def get_chat_service() -> ChatService:
    """
    获取对话服务实例（延迟初始化）

    如果没有数据，返回 HTTP 503 提示用户先搜索。
    数据由 /api/search 端点触发抓取后自动可用。
    """
    global _chat_service

    if _chat_service is None:
        # 使用 auto_fetch 查找最新数据
        from ..auto_fetch import get_latest_data_paths
        paths = get_latest_data_paths()

        if paths is None:
            raise FileNotFoundError(
                "尚无论文数据。请先在搜索框中输入关键词进行搜索。"
            )

        csv_path = paths["csv"]
        if not csv_path.exists():
            raise FileNotFoundError(
                "尚无论文数据。请先在搜索框中输入关键词进行搜索。"
            )

        print(f"使用论文文件: {csv_path}")

        # 查找向量文件
        embeddings_path = paths.get("embeddings")
        if embeddings_path and not embeddings_path.exists():
            embeddings_path = None

        if embeddings_path is None:
            # 回退查找标准路径
            std_emb = PROCESSED_DATA_DIR / "embeddings.npy"
            if std_emb.exists():
                embeddings_path = std_emb
            else:
                print("警告：未找到向量文件，将在初始化时生成")

        # 加载数据
        print(f"加载论文数据: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"已加载 {len(df)} 篇论文")

        # 创建RAG服务
        print("初始化RAG服务...")
        rag_service = RAGService(
            papers_df=df,
            embeddings_path=embeddings_path,
            enable_summary=True,
            summary_model="light"
        )

        _chat_service = ChatService(rag_service)
        print("[OK] 对话服务初始化成功！")

    return _chat_service