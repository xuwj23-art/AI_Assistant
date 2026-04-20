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


def get_chat_service() -> ChatService:
    """获取对话服务实例"""
    global _chat_service

    if _chat_service is None:
        # 直接使用 arxiv_Transformer.csv
        csv_path = RAW_DATA_DIR / "arxiv_Transformer.csv"

        if not csv_path.exists():
            # 如果找不到，尝试查找其他CSV文件
            csv_files = list(RAW_DATA_DIR.glob("arxiv_*.csv"))
            if not csv_files:
                raise FileNotFoundError(
                    f"没有找到论文数据，请先运行 fetch_and_save_arxiv\n"
                    f"期待的文件: {csv_path}\n"
                    f"或者运行: python -m scripts.fetch_and_save_arxiv --query transformer --max-results 50"
                )
            csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)

        print(f"使用论文文件: {csv_path}")

        # 查找向量文件
        embeddings_path = PROCESSED_DATA_DIR / "embeddings.npy"
        if not embeddings_path.exists():
            print("=" * 50)
            print("警告：未找到向量文件")
            print(f"期待路径: {embeddings_path}")
            print("将在初始化时生成向量（可能需要几分钟）")
            print("建议先运行: python -m scripts.generate_embeddings")
            print("=" * 50)
            embeddings_path = None

        # 加载数据
        print(f"加载论文数据: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"已加载 {len(df)} 篇论文")

        # 创建RAG服务
        print("初始化RAG服务...")
        rag_service = RAGService(
            papers_df=df,
            embeddings_path=embeddings_path,
            enable_summary = True,
            summary_model = "light"
        )

        _chat_service = ChatService(rag_service)
        print("✅ 对话服务初始化成功！")

    return _chat_service