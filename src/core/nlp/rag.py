# src/core/nlp/rag.py
from __future__ import annotations
from .summarizer import create_summarizer, PaperSummarizer
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import pickle
from datetime import datetime
import hashlib

import faiss
from sklearn.preprocessing import normalize

# 复用你的EmbeddingGenerator
from .embeddings import EmbeddingGenerator
from ..api.models import PaperResponse
from ..config import MODELS_DIR, PROCESSED_DATA_DIR


class RAGService:
    """
    RAG服务：基于向量检索的论文问答系统
    支持AI摘要总结
    """
    
    def __init__(
        self,
        papers_df: pd.DataFrame,
        embeddings_path: Optional[Path] = None,
        model_name: str = "all-mpnet-base-v2",
        index_path: Optional[Path] = None,
        enable_summary: bool = True,
        summary_model: str = "light"
    ):
        """
        初始化RAG服务
        
        参数:
            enable_summary: 是否启用AI摘要
            summary_model: 摘要模型类型 ("light", "balanced", "full")
            papers_df: 论文DataFrame
            embeddings_path: 预计算向量路径（.npy文件）
            model_name: Sentence Transformer模型名称
            index_path: FAISS索引路径
        """
        self.papers_df = papers_df
        self.model_name = model_name
        self.enable_summary = enable_summary
        
        # 复用你的EmbeddingGenerator
        print(f"初始化EmbeddingGenerator，模型: {model_name}")
        self.embedding_generator = EmbeddingGenerator(
            model_name=model_name,
            batch_size=64
        )
        
        # 加载或创建向量
        if embeddings_path and embeddings_path.exists():
            print(f"加载预计算向量: {embeddings_path}")
            self.embeddings = EmbeddingGenerator.load_embeddings(embeddings_path)
        else:
            print("生成论文向量...")
            self.embeddings = self.embedding_generator.encode_papers(papers_df)
        
        # 加载或创建FAISS索引
        if index_path and index_path.exists():
            print(f"加载FAISS索引: {index_path}")
            self.index = faiss.read_index(str(index_path))
        else:
            print("创建FAISS索引...")
            self.index = self._create_faiss_index()

        # 初始化摘要器
        if enable_summary:
            print(f"初始化AI摘要器，模型类型: {summary_model}")
            self.summarizer = create_summarizer(model_type=summary_model)
        else:
            self.summarizer = None

        # 对话历史缓存
        self.session_history: Dict[str, List[Dict[str, Any]]] = {}

        print(f"RAG服务初始化完成，共 {len(papers_df)} 篇论文，向量维度: {self.embeddings.shape[1]}")
        if enable_summary:
            print("[OK] AI摘要功能已启用")

    def summarize_paper(self, paper: PaperResponse) -> str:
        """
        生成论文摘要的AI总结

        参数:
            paper: 论文对象

        返回:
            AI生成的摘要总结
        """
        if not self.enable_summary or not self.summarizer:
            # 如果未启用摘要，返回原始摘要的前200字
            return paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract

        try:
            # 使用AI生成摘要
            summary = self.summarizer.summarize(paper.abstract)
            return summary
        except Exception as e:
            print(f"摘要生成失败: {e}")
            # 失败时返回截断的原始摘要
            return paper.abstract[:200] + "..."

    def generate_answer_with_summary(
            self,
            query: str,
            context_papers: List[PaperResponse],
            include_summary: bool = True,
            top_k_for_summary: int = 3,
            chat_history: Optional[List[Dict[str, str]]] = None,
            topic_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        生成答案，包含AI摘要

        参数:
            query: 用户问题
            context_papers: 相关论文列表（按相关性降序）
            include_summary: 是否包含AI摘要
            top_k_for_summary: 取前 K 篇做摘要 / 来源展示
            chat_history: 多轮对话历史（OpenAI 格式 [{role, content}, ...]，
                          不含当前 query）；用于 DeepSeek 多轮上下文

        返回:
            {answer, sources, question, summary_enabled, model_used}
        """
        sources: List[Dict[str, Any]] = []
        summaries_for_synthesis: List[str] = []

        for paper in context_papers[:top_k_for_summary]:
            # 单次生成 AI 摘要并复用，避免重复调用模型（性能 ↑ 50%）
            if include_summary and self.enable_summary:
                paper_summary = self.summarize_paper(paper)
            else:
                paper_summary = (
                    paper.abstract[:200] + "..."
                    if len(paper.abstract) > 200
                    else paper.abstract
                )

            sources.append({
                "id": paper.id,
                "title": paper.title,
                "authors": paper.authors[:2],
                "published": paper.published.isoformat() if paper.published else None,
                "pdf_url": str(paper.pdf_url) if paper.pdf_url else None,
                "abstract": paper.abstract,
                "ai_summary": paper_summary if include_summary else None,
                "relevance": getattr(paper, '_relevance', None),
            })
            summaries_for_synthesis.append(paper_summary)

        if not context_papers:
            # 检索为空但有对话历史 → 让 LLM 基于历史给"承上"型回答，避免硬冷的失败模板
            if chat_history:
                history_only_answer = self._answer_from_history_only(query, chat_history)
                if history_only_answer:
                    return {
                        "answer": history_only_answer,
                        "sources": [],
                        "question": query,
                        "summary_enabled": self.enable_summary,
                        "model_used": "deepseek_history",
                    }
            return {
                "answer": (
                    f"Sorry, no papers directly related to \"{query}\" were found. "
                    "Try more specific keywords or different search terms."
                ),
                "sources": [],
                "question": query,
                "summary_enabled": self.enable_summary,
                "model_used": "none",
            }

        # 优先级 1：DeepSeek（多轮、上下文感知）
        deepseek_answer = self._generate_with_deepseek(
            query=query,
            papers=context_papers[:top_k_for_summary],
            paper_summaries=summaries_for_synthesis,
            chat_history=chat_history or [],
            topic_context=topic_context,
        )
        if deepseek_answer:
            return {
                "answer": deepseek_answer,
                "sources": sources,
                "question": query,
                "summary_enabled": self.enable_summary,
                "model_used": "deepseek",
            }

        # 优先级 2：本地综合（无 DeepSeek key 或 API 失败时）
        answer = self._synthesize_answer(
            query,
            context_papers[:top_k_for_summary],
            summaries_for_synthesis,
        )

        return {
            "answer": answer,
            "sources": sources,
            "question": query,
            "summary_enabled": self.enable_summary,
            "model_used": "local",
        }

    # ---- 追问检测（需要把当前 query 与历史合并改写后再检索）----
    # 关键词必须是"完整短语 / 词组"，避免单字误触发（如 "再" 会匹配 "再次/再者"）
    _FOLLOWUP_KEYWORDS_ZH = (
        "再讲", "再说说", "再细", "细讲", "展开讲", "展开说",
        "详细说", "详细讲", "进一步", "深入讲",
        "继续讲", "接着讲", "接着说",
        "上面那", "上面说", "上述", "前面那", "前面提",
        "上一个", "上一条", "上一篇",
        "刚才说", "刚刚说", "刚提到",
        "第一个方向", "第二个方向", "第三个方向", "第四个方向", "第五个方向",
        "第1个", "第2个", "第3个", "第4个", "第5个",
        "这个方向", "这个方法", "这个论文", "这个观点",
        "该方向", "该方法", "该论文",
        "它的优点", "它的缺点", "它们的", "它们之间",
    )
    _FOLLOWUP_KEYWORDS_EN = (
        "tell me more", "elaborate", "expand on", "more about",
        "more detail", "go deeper", "what about",
        "the first one", "the second one", "the third one",
        "the first direction", "the second direction", "the third direction",
        "this approach", "this method", "this paper",
        "you mentioned", "as you said",
    )

    @classmethod
    def needs_query_rewrite(cls, query: str, has_history: bool) -> bool:
        """
        判断当前 query 是否依赖上下文（需要改写后再检索）。

        触发条件（任一）：
        - 含明显的追问 / 指代关键词（多字短语，避免单字误触发）
        - query 极短（< 8 字符 / 2 个英文 token）且有对话历史
        """
        if not has_history or not query:
            return False
        q = query.strip()
        # 关键词检测（已是 ≥2 字短语，足够精确）
        for kw in cls._FOLLOWUP_KEYWORDS_ZH:
            if kw in q:
                return True
        ql = q.lower()
        for kw in cls._FOLLOWUP_KEYWORDS_EN:
            if kw in ql:
                return True
        # 极短问题大概率是追问（旧阈值 12/4 太宽，调到 8/2）
        if len(q) < 8 and len(ql.split()) < 2:
            return True
        return False

    def rewrite_query_for_search(
            self,
            query: str,
            chat_history: List[Dict[str, str]],
    ) -> Optional[str]:
        """
        基于对话历史把追问改写成独立完整的检索 query。

        例：
            历史包含 "LLM-as-a-Judge 对齐评估" 主题
            原句:    "再细讲讲第 3 个方向的代表性方法"
            改写:    "LLM-as-a-Judge alignment evaluation representative methods"

        失败 / 无 LLM 时返回 None，调用方自行决定使用原句。
        """
        if not chat_history:
            return None
        try:
            from ..llm.provider import get_llm_provider
            provider = get_llm_provider()
            if not provider.deepseek_available:
                return None
        except Exception:
            return None

        # 取最近 4 轮历史（够用且省 token）
        recent = []
        for h in chat_history[-8:]:
            role = h.get("role")
            content = h.get("content", "")
            if role in ("user", "assistant") and content:
                # 助手回答可能很长，截短
                if role == "assistant" and len(content) > 600:
                    content = content[:600] + "..."
                recent.append(f"[{role}] {content}")
        history_block = "\n\n".join(recent)

        rewrite_prompt = (
            "Below is a multi-turn research conversation and the user's latest follow-up question. "
            "Rewrite the [Latest Follow-up] into a standalone, complete English search query suitable "
            "for a vector search engine, replacing pronouns (e.g. 'the third direction', 'it') with "
            "the explicit research topic terms from the conversation.\n\n"
            f"Conversation history:\n{history_block}\n\n"
            f"Latest follow-up: {query}\n\n"
            "Requirements:\n"
            "- Output only one English search phrase (5-15 words), no explanation\n"
            "- Use core research topic words + modifiers, e.g. 'LLM-as-a-Judge alignment evaluation representative methods'\n"
            "- No quotes, no prefix, no trailing punctuation\n\n"
            "Rewritten search query:"
        )

        try:
            # max_tokens=60：一个 5-15 词的英文短语足够，省 quota
            answer = provider.chat_completion(
                [{"role": "user", "content": rewrite_prompt}],
                max_tokens=60,
                temperature=0.2,
            )
        except Exception as exc:
            print(f"[RAG] query 改写失败: {exc}")
            return None

        if not answer:
            return None
        rewritten = answer.strip().strip('"\u201C\u201D\u2018\u2019').strip()
        # 单行 + 长度合理
        if "\n" in rewritten:
            rewritten = rewritten.split("\n", 1)[0].strip()
        if 3 < len(rewritten) < 250:
            return rewritten
        return None

    # ---- 元问题检测（不需要 RAG 检索的对话管理类问题）----
    # 关键词必须明确指向"对话本身"，避免误把概念问题当元问题
    # 例如旧版 "刚刚" 会匹配 "我刚刚遇到的术语" → 误判
    _META_KEYWORDS_ZH = (
        "我刚刚问", "我刚才问", "我刚刚的问题", "我刚才的问题",
        "我之前问", "我上一个问题", "我上次问",
        "你刚刚说", "你刚才说", "你之前说",
        "总结一下我们", "总结一下之前", "总结一下对话", "总结对话",
        "我们聊了什么", "我们聊到", "我们之前聊",
        "对话历史", "重复一下你的回答", "再说一遍你刚才",
    )
    _META_KEYWORDS_EN = (
        "what did i ask", "what was my question", "what was my last question",
        "what did you say earlier", "what did you say before",
        "summarize our conversation", "summarize our chat",
        "summarize what we discussed", "what have we discussed",
        "repeat your last answer", "say that again",
        "our conversation so far", "our chat so far",
    )

    @classmethod
    def is_meta_question(cls, query: str, n_history_turns: int = 0) -> bool:
        """
        判断是否为对话管理类问题（不需要检索论文）。

        修复点：要求**同时满足两条**才判定为元问题：
            1. 含明确的元对话关键词（指向"对话本身"，不是泛泛的"刚才"）
            2. 至少有 2 轮历史可被引用

        这样可以避免把"我刚遇到的术语是什么意思"误判为元问题。
        """
        if not query:
            return False
        # 没有足够历史 → 不可能是有效的元问题（即使含关键词也直接走检索更稳）
        if n_history_turns < 2:
            return False

        q_lower = query.strip().lower()
        for kw in cls._META_KEYWORDS_ZH:
            if kw in query:
                return True
        for kw in cls._META_KEYWORDS_EN:
            if kw in q_lower:
                return True
        return False

    def generate_meta_answer(
            self,
            query: str,
            chat_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        处理元问题：不做检索，直接让 LLM 基于对话历史作答。
        无 LLM 时回退一段简短模板。
        """
        # 优先 DeepSeek
        try:
            from ..llm.provider import get_llm_provider
            provider = get_llm_provider()
            if provider.deepseek_available and chat_history:
                messages: List[Dict[str, str]] = [
                    {
                        "role": "system",
                        "content": (
                            "You are a research literature assistant. The user is asking a meta-question "
                            "about this conversation itself (e.g. asking what was discussed, requesting a "
                            "summary of the dialogue, etc.).\n"
                            "Based on the conversation history provided below, respond concisely in English "
                            "(about 100-200 words). No need to cite papers."
                        ),
                    }
                ]
                # 历史最多取 8 轮
                for h in chat_history[-16:]:
                    if h.get("role") in ("user", "assistant") and h.get("content"):
                        messages.append({"role": h["role"], "content": h["content"]})
                messages.append({"role": "user", "content": query})

                ans = provider.chat_completion(
                    messages, max_tokens=600, temperature=0.5
                )
                if ans and ans.strip():
                    return {
                        "answer": ans.strip(),
                        "sources": [],
                        "question": query,
                        "summary_enabled": self.enable_summary,
                        "model_used": "deepseek_meta",
                    }
        except Exception as exc:
            print(f"[RAG] meta 回答 DeepSeek 失败: {exc}")

        # 本地兜底：列出最近问过的问题
        last_user_qs = [
            h["content"] for h in chat_history[-10:]
            if h.get("role") == "user"
        ]
        if last_user_qs:
            recent = "\n".join(f"- {q}" for q in last_user_qs[-3:])
            text = f"Recent questions in our conversation:\n{recent}"
        else:
            text = "No conversation history available yet."
        return {
            "answer": text,
            "sources": [],
            "question": query,
            "summary_enabled": self.enable_summary,
            "model_used": "local_meta",
        }

    def _answer_from_history_only(
            self,
            query: str,
            chat_history: List[Dict[str, str]],
    ) -> Optional[str]:
        """
        当检索没找到论文但有对话历史时，让 LLM 基于已有上下文给出"承上"回答。
        典型场景：用户问"再细讲讲第 3 个方向"，但改写后的 query 仍未命中相关论文。
        """
        try:
            from ..llm.provider import get_llm_provider
            provider = get_llm_provider()
            if not provider.deepseek_available:
                return None
        except Exception:
            return None

        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a research literature assistant. The current turn retrieved no new papers "
                    "directly relevant to the new question. Please give an honest, helpful answer based "
                    "solely on the existing conversation history:\n"
                    "1. If the follow-up refers to a specific sub-direction or paper mentioned earlier, "
                    "   expand on it (keep [N] citation numbers) and honestly note that "
                    "   'no new papers were retrieved for this turn'.\n"
                    "2. If you cannot build on the history at all, briefly explain and suggest different "
                    "   keywords or an additional search.\n"
                    "3. Always respond in English.\n"
                    "4. Do not fabricate papers or methods not present in the conversation history."
                ),
            }
        ]
        for h in chat_history[-12:]:
            role = h.get("role")
            content = h.get("content", "")
            if role in ("user", "assistant") and content:
                if role == "assistant" and len(content) > 1500:
                    content = content[:1500] + "..."
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        try:
            return provider.chat_completion(
                messages, max_tokens=2000, temperature=0.5
            )
        except Exception as exc:
            print(f"[RAG] history-only 回答失败: {exc}")
            return None

    def _generate_with_deepseek(
            self,
            query: str,
            papers: List[PaperResponse],
            paper_summaries: List[str],
            chat_history: List[Dict[str, str]],
            history_turns: int = 6,
            max_abstract_chars: int = 800,
            topic_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        用 DeepSeek 生成多轮上下文感知的回答。无 key/失败 → 返回 None。

        参数:
            history_turns: 最多取最近 N 轮对话（user+assistant 交替）
            max_abstract_chars: 单篇论文摘要送入 LLM 的最大字符数
        """
        try:
            from ..llm.provider import get_llm_provider
            provider = get_llm_provider()
        except Exception as exc:
            print(f"[RAG] LLMProvider 加载失败: {exc}")
            return None

        if not provider.deepseek_available:
            return None

        # ---- 构造检索 context（每篇论文：标题 + 作者 + 摘要片段）----
        ctx_lines = []
        for i, (paper, summary) in enumerate(zip(papers, paper_summaries), 1):
            authors = ", ".join(paper.authors[:3]) if paper.authors else "Unknown"
            year = (
                paper.published.year if paper.published else "?"
            )
            abstract = paper.abstract or ""
            if len(abstract) > max_abstract_chars:
                abstract = abstract[:max_abstract_chars].rstrip() + "..."
            ctx_lines.append(
                f"[{i}] {paper.title} ({authors}, {year})\n"
                f"    Abstract: {abstract}"
            )
        retrieved_block = "\n\n".join(ctx_lines)

        # ---- system prompt ----
        system_prompt = (
            "You are a senior research literature assistant skilled at providing in-depth, comprehensive "
            "survey-style answers on specific research directions.\n"
            "Requirements:\n"
            "1. **Factual accuracy**: Answer only from the retrieved content; if the content cannot support "
            "   a claim, state this clearly — never fabricate information.\n"
            "2. **Cite papers**: Use [1] [2] etc. corresponding to the retrieved snippets below. Only include "
            "   a citation number if you actually use that paper in your answer.\n"
            "3. **Clear, well-developed structure**:\n"
            "   - Begin with 1-2 sentences of overview\n"
            "   - Expand each direction/point with: core idea + specific methods/conclusions from papers + citation\n"
            "   - End with a synthesis or comparison where appropriate\n"
            "4. **Always respond in English** regardless of the language of the user's question.\n"
            "5. **Multi-turn awareness**: The user may follow up on previous answers (e.g. 'tell me more about "
            "   point 3') — maintain coherent context.\n"
            "6. **Adaptive length**: broad/comparative questions 600-1000 words; focused follow-ups 200-400 words; "
            "   do not pad unnecessarily."
        )

        # ---- topic meta-info block (explains what "Topic N" is) ----
        topic_block = ""
        if topic_context:
            tid = topic_context.get("topic_id")
            tlabel = topic_context.get("topic_label") or topic_context.get("topic_name") or ""
            tkeywords = topic_context.get("keywords") or []
            tcount = topic_context.get("paper_count")
            # tkeywords can be List[str] or List[TopicKeyword] (with .word), normalise to str
            kw_strs = [kw.word if hasattr(kw, "word") else str(kw) for kw in tkeywords[:10]]
            kw_str = ", ".join(kw_strs) if kw_strs else "(none)"
            topic_block = (
                f"[Topic metadata] The \"Topic {tid}\" referenced in this conversation is a research "
                f"sub-cluster automatically identified by BERTopic from the current search results "
                f"(it is NOT a concept from any single paper):\n"
                f"  - Topic label (LLM-named): {tlabel}\n"
                f"  - Top TF-IDF keywords: {kw_str}\n"
                f"  - Papers in this topic: {tcount}\n"
                f"The retrieved snippets below are representative papers from this topic. Use them to "
                f"explain what Topic {tid} is and what research directions it covers — do not deny its existence.\n\n"
            )

        # ---- user message: query + topic meta + retrieved context ----
        user_with_ctx = (
            f"Question: {query}\n\n"
            f"{topic_block}"
            f"Retrieved papers (ordered by relevance):\n\n{retrieved_block}\n\n"
            "Please answer the question based on the papers above, using [N] citation format."
        )

        # ---- messages：system + 历史最近 N 轮 + 当前 user ----
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        # 截取最近 N 轮（每轮含 user+assistant 两条），保留最近的 2*N 条
        if chat_history:
            recent = chat_history[-history_turns * 2:]
            for h in recent:
                role = h.get("role")
                content = h.get("content", "")
                if role in ("user", "assistant") and content:
                    # 历史 assistant 内容可能很长，做温和截断
                    if role == "assistant" and len(content) > 1000:
                        content = content[:1000] + "..."
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_with_ctx})

        # deepseek-v4-pro 是推理模型，token 预算需同时覆盖 reasoning_content + content。
        # 实测 reasoning 占 300-600 tokens；为了让 content 能放下 600-1000 字综述式回答，
        # 整体 budget 调到 2800（推理留 ~600 + 答案 ~2200）。
        try:
            answer = provider.chat_completion(
                messages,
                max_tokens=2800,
                temperature=0.5,
            )
        except Exception as exc:
            print(f"[RAG] DeepSeek 调用异常: {exc}")
            return None

        if answer and answer.strip():
            return answer.strip()
        return None

    def _synthesize_answer(
            self,
            query: str,
            papers: List[PaperResponse],
            paper_summaries: List[str],
    ) -> str:
        """
        把 top-k 篇论文的标题 + AI 摘要拼成一段综述式回答。

        优先策略:
            如果 summarizer 可用，把多篇 AI 摘要再过一次摘要器，输出更紧凑的综合答案；
        回退策略:
            分点列出 + 每篇贴一句话总结。
        """
        # 综述风格：先来一段总结，再分点列出来源
        intro = ""
        if self.enable_summary and self.summarizer:
            try:
                joined = " ".join(
                    f"Paper {i+1}: {p.title}. {s}"
                    for i, (p, s) in enumerate(zip(papers, paper_summaries))
                )
                # 截断防止超 token 上限
                joined = joined[:2000]
                joined_for_model = (
                    f"Question: {query}. Synthesize an integrated summary "
                    f"answering this question based on these papers: {joined}"
                )
                intro_text = self.summarizer.summarize(joined_for_model)
                if intro_text and len(intro_text) > 20:
                    intro = f"💡 {intro_text}\n\n"
            except Exception as e:
                print(f"[RAG] 综述生成失败，使用模板回退: {e}")

        body_lines = [f"Found {len(papers)} highly relevant papers for \"{query}\":\n"]
        for i, (paper, summary) in enumerate(zip(papers, paper_summaries), 1):
            line = f"**{i}. {paper.title}**\n   📝 {summary}\n"
            body_lines.append(line)

        return intro + "\n".join(body_lines)
    
    def _create_faiss_index(self) -> faiss.Index:
        """
        创建 FAISS 余弦相似度索引

        注意:
            FAISS 的 IndexFlatIP 计算内积；要等价于余弦相似度，
            索引向量与查询向量都必须 L2 归一化。
            之前版本只在查询时归一化，导致检索分数偏差。
        """
        emb = self.embeddings.astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb_norm = emb / norms
        # 同步把归一化结果写回，让命中分数与"真实余弦"一致
        self.embeddings = emb_norm

        dimension = emb_norm.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(emb_norm)
        return index
    
    def search_similar_papers(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.15,
        restrict_topic_id: Optional[int] = None,
    ) -> List[Tuple[PaperResponse, float]]:
        """
        搜索与 query 余弦相似度最高的论文

        参数:
            query: 用户提问 / 关键词
            top_k: 返回论文数量
            threshold: 余弦相似度阈值，低于此值的命中会被过滤
            restrict_topic_id: 若不为 None，仅返回该主题下的论文（实现 topic-scoped RAG）
        """
        query_embedding = self.embedding_generator.encode_texts(
            [query], show_progress=False
        )
        query_embedding = normalize(query_embedding, norm='l2').astype(np.float32)

        # 若限定主题，扩大检索池（topic 内可能只有几篇，直接 top_k * 2 不够）
        search_k = top_k * 8 if restrict_topic_id is not None else top_k * 2
        scores, indices = self.index.search(query_embedding, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < threshold or idx >= len(self.papers_df):
                continue

            row = self.papers_df.iloc[idx]

            # 主题过滤
            if restrict_topic_id is not None:
                row_tid = row.get("topic_id")
                try:
                    if int(row_tid) != int(restrict_topic_id):
                        continue
                except (TypeError, ValueError):
                    continue

            # 转换为 PaperResponse：直接调用公共工具 row_to_paper，
            # 避免临时实例化 PaperService 引发的循环依赖与"空路径"警告
            from ..api.paper_utils import row_to_paper
            paper = row_to_paper(row)

            results.append((paper, float(score)))

            if len(results) >= top_k:
                break

        return results

    # ---- topic 引用检测（实现 topic-scoped RAG）----
    @staticmethod
    def detect_topic_reference(query: str) -> Optional[int]:
        """
        从 query 中提取明确指向的 topic_id。

        匹配模式（任意一种）：
            "topic 5"  / "topic5" / "Topic 0"
            "主题 3"   / "主题3"
            "第 2 个主题" / "第2个主题"
        匹配失败返回 None。
        """
        if not query:
            return None
        import re as _re
        patterns = [
            r"topic\s*(\d{1,2})",
            r"主题\s*(\d{1,2})",
            r"第\s*(\d{1,2})\s*个?\s*主题",
        ]
        for pat in patterns:
            m = _re.search(pat, query, _re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    pass
        return None

    def get_topic_representative_papers(
            self,
            topic_id: int,
            top_k: int = 5,
    ) -> List[Tuple[PaperResponse, float]]:
        """
        返回某 topic 内"最具代表性"的 top_k 论文（按到主题质心的余弦相似度降序）。

        与 search_similar_papers 不同：
        - 不依赖 query embedding，直接用 topic 自身的质心
        - 用于"Topic N 的代表论文 / 最相关论文"这类问题
        """
        if topic_id is None:
            return []
        try:
            topic_df = self.papers_df[self.papers_df["topic_id"] == topic_id]
        except Exception:
            return []
        if topic_df.empty or topic_id == -1:
            return []

        topic_indices = topic_df.index.to_numpy()
        topic_vectors = self.embeddings[topic_indices].astype(np.float32)

        centroid = topic_vectors.mean(axis=0)
        cnorm = np.linalg.norm(centroid) + 1e-12
        centroid_unit = centroid / cnorm

        vnorms = np.linalg.norm(topic_vectors, axis=1, keepdims=True) + 1e-12
        vec_unit = topic_vectors / vnorms
        scores = (vec_unit @ centroid_unit).tolist()

        # 排序后取前 top_k
        from ..api.paper_utils import row_to_paper
        idx_score = sorted(
            zip(topic_indices.tolist(), scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        results = []
        for idx, sc in idx_score:
            paper = row_to_paper(self.papers_df.iloc[idx])
            results.append((paper, float(sc)))
        return results

    def generate_answer(
            self,
            query: str,
            context_papers: List[PaperResponse]
    ) -> Dict[str, Any]:
        """
        生成答案
        """
        sources = []

        # 即使没有论文，也返回一些信息
        if not context_papers:
            # 尝试降低阈值再搜索一次
            results = self.search_similar_papers(query, top_k=3, threshold=0.1)
            if results:
                context_papers = [p for p, _ in results]
            else:
                return {
                    "answer": f"Sorry, no papers found for \"{query}\". Please try different keywords.",
                    "sources": [],
                    "question": query
                }

        for i, paper in enumerate(context_papers[:3]):
            abstract = paper.abstract
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."

            sources.append({
                "id": paper.id,
                "title": paper.title,
                "authors": paper.authors[:2],
                "published": paper.published.isoformat() if paper.published else None,
                "pdf_url": str(paper.pdf_url) if paper.pdf_url else None,
                "relevance": getattr(paper, '_relevance', None)  # 添加相关性分数
            })

        # 生成答案
        answer = self._simple_answer(query, context_papers)

        return {
            "answer": answer,
            "sources": sources,
            "question": query
        }

    def _simple_answer(self, query: str, papers: List[PaperResponse]) -> str:
        answer = f"Here are the most relevant papers found for \"{query}\":\n\n"

        for i, paper in enumerate(papers[:3], 1):
            answer += f"{i}. {paper.title}\n"
            answer += f"   Abstract: {paper.abstract[:200]}...\n\n"

        answer += "These papers may be helpful for your query."
        return answer
    
    def _simulate_answer(self, query: str, papers: List[PaperResponse]) -> str:
        """模拟生成答案"""
        keywords = query.lower().split()
        
        relevant_sentences = []
        for paper in papers[:3]:
            abstract = paper.abstract.lower()
            sentences = abstract.split('. ')
            for sentence in sentences:
                if any(keyword in sentence for keyword in keywords):
                    relevant_sentences.append(sentence)
        
        if relevant_sentences:
            relevant_sentences = list(set(relevant_sentences))[:3]
            answer = "Based on the related papers, here is what I found:\n\n"
            for i, sent in enumerate(relevant_sentences, 1):
                answer += f"{i}. {sent.capitalize()}.\n\n"
        else:
            answer = "Related papers include:\n"
            for paper in papers[:3]:
                answer += f"- {paper.title}\n"
            answer += "\nPlease check the full abstracts for more details."
        
        return answer
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """创建新的对话会话"""
        if session_id is None:
            timestamp = datetime.now().isoformat()
            session_id = hashlib.md5(timestamp.encode()).hexdigest()[:12]
        
        self.session_history[session_id] = []
        return session_id
    
    def add_to_history(self, session_id: str, role: str, content: str):
        """添加对话历史"""
        if session_id not in self.session_history:
            self.session_history[session_id] = []
        
        self.session_history[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """获取对话历史（含 timestamp 等元数据）"""
        return self.session_history.get(session_id, [])

    def get_history_for_llm(self, session_id: str) -> List[Dict[str, str]]:
        """获取仅含 role/content 的历史（OpenAI 格式，供 LLM 多轮用）"""
        history = self.session_history.get(session_id, [])
        return [
            {"role": h["role"], "content": h["content"]}
            for h in history
            if h.get("role") in ("user", "assistant") and h.get("content")
        ]

    def clear_history(self, session_id: str) -> bool:
        """清空指定会话的对话历史，返回是否成功"""
        if session_id in self.session_history:
            self.session_history[session_id] = []
            return True
        return False
    
    def save_index(self, path: Path):
        """保存FAISS索引"""
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        print(f"索引已保存: {path}")