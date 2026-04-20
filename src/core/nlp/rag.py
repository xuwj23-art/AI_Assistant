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
            print("✅ AI摘要功能已启用")

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
            include_summary: bool = True
    ) -> Dict[str, Any]:
        """
        生成答案，包含AI摘要

        参数:
            query: 用户问题
            context_papers: 相关论文列表
            include_summary: 是否包含AI摘要

        返回:
            包含答案和来源的字典
        """
        sources = []

        for i, paper in enumerate(context_papers[:3]):
            # 获取AI摘要或截断的原始摘要
            if include_summary and self.enable_summary:
                paper_summary = self.summarize_paper(paper)
            else:
                paper_summary = paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract

            sources.append({
                "id": paper.id,
                "title": paper.title,
                "authors": paper.authors[:2],
                "published": paper.published.isoformat() if paper.published else None,
                "pdf_url": str(paper.pdf_url) if paper.pdf_url else None,
                "abstract": paper.abstract,  # 原始摘要
                "ai_summary": paper_summary if include_summary else None,  # AI总结
                "relevance": getattr(paper, '_relevance', None)
            })

        # 生成答案
        if context_papers:
            answer = self._enhanced_answer(query, context_papers, sources)
        else:
            answer = "没有找到相关的论文信息。"

        return {
            "answer": answer,
            "sources": sources,
            "question": query,
            "summary_enabled": self.enable_summary
        }

    def _enhanced_answer(
            self,
            query: str,
            papers: List[PaperResponse],
            sources: List[Dict]
    ) -> str:
        """
        增强版答案生成，包含AI摘要
        """
        answer = f"关于「{query}」，我找到以下相关论文：\n\n"

        for i, paper in enumerate(papers[:3], 1):
            answer += f"{i}. {paper.title}\n"

            # 如果有AI摘要，使用它
            if self.enable_summary and self.summarizer:
                summary = self.summarize_paper(paper)
                answer += f"   📝 AI总结：{summary}\n"
            else:
                # 否则使用原始摘要
                abstract = paper.abstract[:150] + "..." if len(paper.abstract) > 150 else paper.abstract
                answer += f"   摘要：{abstract}\n"

            answer += "\n"

        answer += "以上论文可能对您有帮助。"
        return answer
    
    def _create_faiss_index(self) -> faiss.Index:
        """创建FAISS索引"""
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(self.embeddings.astype(np.float32))
        return index
    
    def search_similar_papers(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.2
    ) -> List[Tuple[PaperResponse, float]]:
        """搜索相似论文"""
        # 使用embedding_generator生成查询向量
        query_embedding = self.embedding_generator.encode_texts([query])
        # 归一化（因为FAISS使用内积）
        query_embedding = normalize(query_embedding, norm='l2')
        
        # 搜索
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k * 2)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < threshold or idx >= len(self.papers_df):
                continue
            
            # 获取论文
            row = self.papers_df.iloc[idx]
            
            # 转换为PaperResponse
            from ..api.services import PaperService
            temp_service = PaperService("")
            paper = temp_service._row_to_paper(row)
            
            results.append((paper, float(score)))
            
            if len(results) >= top_k:
                break
        
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
                    "answer": f"抱歉，没有找到关于「{query}」的相关论文。您可以尝试其他关键词。",
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
        """
        简单答案生成方法
        """
        answer = f"关于「{query}」，我找到以下相关论文：\n\n"

        for i, paper in enumerate(papers[:3], 1):
            answer += f"{i}. {paper.title}\n"
            answer += f"   摘要：{paper.abstract[:200]}...\n\n"

        answer += "以上论文可能对您有帮助。"
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
            answer = "根据相关论文，我找到以下信息：\n\n"
            for i, sent in enumerate(relevant_sentences, 1):
                answer += f"{i}. {sent.capitalize()}.\n\n"
        else:
            answer = "相关论文包括：\n"
            for paper in papers[:3]:
                answer += f"- {paper.title}\n"
            answer += "\n建议查看完整摘要获取更多信息。"
        
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
        """获取对话历史"""
        return self.session_history.get(session_id, [])
    
    def save_index(self, path: Path):
        """保存FAISS索引"""
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        print(f"索引已保存: {path}")