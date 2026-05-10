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
            "message": "Session initialised. You can start asking questions."
        }

    def send_message(self, session_id: str, message: str, top_k: int = 5) -> Dict[str, Any]:
        """
        发送消息（多轮上下文感知 + 元问题快路径）

        - 元问题（询问对话本身）→ 跳过 FAISS 检索，直接用历史让 LLM 答
        - 普通问题 → 检索论文 → 把历史+检索内容传给 LLM
        """
        # 自愈：如果 session 不存在（uvicorn 热重载、服务重置等），用传入 id 重新建立
        # 这样前端不必处理 404 重试，体验更流畅
        if session_id not in self.sessions:
            print(f"[ChatService] session {session_id!r} 不存在，自动重建")
            self.init_session(session_id=session_id)

        # 取出当前历史（不含本轮 message），用于 LLM 上下文
        history_for_llm = self.rag_service.get_history_for_llm(session_id)

        # 写入当前 user 消息
        self.rag_service.add_to_history(session_id, "user", message)

        # ===== 元问题快路径：不做检索，节省 ~30s =====
        # history_for_llm 是 OpenAI 格式 [{role, content}, ...]，每轮 = user+assistant 2 条
        n_turns = len(history_for_llm) // 2
        if self.rag_service.is_meta_question(message, n_history_turns=n_turns):
            print(f"[ChatService] 检测到元问题（已有 {n_turns} 轮历史），跳过 RAG 检索: {message!r}")
            result = self.rag_service.generate_meta_answer(
                query=message,
                chat_history=history_for_llm,
            )
            self.rag_service.add_to_history(session_id, "assistant", result["answer"])
            if session_id in self.sessions:
                self.sessions[session_id]["message_count"] += 1
            return result

        # ===== 普通问题：完整 RAG 流程 =====
        # 路径 0：query 引用具体 topic_id → 走 topic-scoped 检索
        topic_ref = self.rag_service.detect_topic_reference(message)
        # 若消息未显式引用 topic，但 session 初始化时绑定了 topic_id → 作为隐式上下文
        session_topic_id: Optional[int] = (
            self.sessions.get(session_id, {}).get("topic_id")
        )
        if topic_ref is None and session_topic_id is not None:
            topic_ref = session_topic_id
        topic_scope_log: str = ""
        topic_context: Optional[Dict[str, Any]] = None
        if topic_ref is not None:
            # 拉取 topic 元信息（label / keywords / paper_count）→ 注入 LLM
            # 这样 LLM 不会把 "topic N" 当成未知词来回答
            try:
                from .topic_routes import get_topic_service
                topic_svc = get_topic_service()
                tinfo = topic_svc.get_topic_by_id(topic_ref)
                if tinfo is not None:
                    topic_context = {
                        "topic_id": tinfo.topic_id,
                        "topic_name": tinfo.topic_name,
                        "topic_label": tinfo.topic_label,
                        "keywords": tinfo.keywords,
                        "paper_count": tinfo.paper_count,
                    }
                    kw_preview = [kw.word if hasattr(kw, "word") else str(kw) for kw in tinfo.keywords[:5]]
                    print(
                        f"[ChatService] 已注入 topic_context: id={tinfo.topic_id} "
                        f"label={tinfo.topic_label!r} kw={kw_preview}"
                    )
            except Exception as exc:
                print(f"[ChatService] 拉取 topic 元信息失败（忽略）: {exc}")

            # 检查 "代表性 / 是什么 / 介绍 / 包含哪些" 等定义型问句 → 走主题质心
            import re
            wants_representative = bool(
                re.search(
                    r"代表|最相关|最重要|核心论文|top\s*\d+|representative|most\s+relevant"
                    r"|是什么|什么意思|介绍|讲讲|讲一下|说一下|包含|涉及|哪些方向|哪些领域|哪些内容"
                    r"|what\s+is|tell\s+me\s+about|cover|include",
                    message,
                    re.IGNORECASE,
                )
            )
            if wants_representative:
                similar_papers = self.rag_service.get_topic_representative_papers(
                    topic_ref, top_k=top_k
                )
                topic_scope_log = f"topic_id={topic_ref} 代表性"
                print(f"[ChatService] 检测到 'topic {topic_ref} 定义/代表性问句' → 走主题质心排序")
            else:
                # 普通"topic N 中关于 X" → FAISS 但限定 topic
                similar_papers = self.rag_service.search_similar_papers(
                    query=message, top_k=top_k, restrict_topic_id=topic_ref
                )
                topic_scope_log = f"topic_id={topic_ref} 限定检索"
                print(f"[ChatService] 检测到 topic {topic_ref} 引用 → 限定 FAISS 检索")

            # 把 topic_scope 信息写到 result，方便前端展示
            search_query = message
            rewritten_for_log: str = ""
        else:
            # 路径 1：追问改写（无 topic 引用时）
            search_query = message
            rewritten_for_log = ""
            if self.rag_service.needs_query_rewrite(message, has_history=bool(history_for_llm)):
                rewritten = self.rag_service.rewrite_query_for_search(
                    query=message,
                    chat_history=history_for_llm,
                )
                if rewritten:
                    rewritten_for_log = rewritten
                    search_query = rewritten
                    print(f"[ChatService] 追问检测: {message!r} → 改写为 {rewritten!r}")

            similar_papers = self.rag_service.search_similar_papers(
                query=search_query, top_k=top_k
            )

            # 改写后仍找不到，再用原句兜底（关键词可能恰好命中）
            if not similar_papers and search_query != message:
                print(f"[ChatService] 改写后无命中，用原句再试: {message!r}")
                similar_papers = self.rag_service.search_similar_papers(
                    query=message, top_k=top_k
                )

        papers = [p for p, _ in similar_papers]
        scores = [s for _, s in similar_papers]

        # 生成回答（携带历史 → DeepSeek 实现真正多轮）
        # 注意：送给 LLM 的 query 仍是用户原句（保留语气与意图）
        result = self.rag_service.generate_answer_with_summary(
            query=message,
            context_papers=papers,
            include_summary=True,
            chat_history=history_for_llm,
            topic_context=topic_context,
        )
        if rewritten_for_log:
            result["rewritten_query"] = rewritten_for_log
        if topic_scope_log:
            result["topic_scope"] = topic_scope_log

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

    def clear_history(self, session_id: str) -> bool:
        """清空指定会话的对话历史"""
        ok = self.rag_service.clear_history(session_id)
        if ok and session_id in self.sessions:
            self.sessions[session_id]["message_count"] = 0
        return ok

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
                "No paper data found. Please search for a keyword first."
            )

        csv_path = paths["csv"]
        if not csv_path.exists():
            raise FileNotFoundError(
                "No paper data found. Please search for a keyword first."
            )

        print(f"Using paper file: {csv_path}")

        embeddings_path = paths.get("embeddings")
        if embeddings_path and not embeddings_path.exists():
            embeddings_path = None

        if embeddings_path is None:
            std_emb = PROCESSED_DATA_DIR / "embeddings.npy"
            if std_emb.exists():
                embeddings_path = std_emb
            else:
                print("Warning: no embeddings file found; will generate on init")

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