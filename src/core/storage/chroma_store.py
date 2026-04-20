"""
ChromaDB 向量持久化存储

用于缓存已抓取论文的向量，避免重复向量化
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
from ..sources.base import Paper


class ChromaStore:
    """
    ChromaDB 向量存储
    
    功能:
    - 持久化论文向量到本地数据库
    - 增量添加新论文（自动去重）
    - 语义检索（基于向量相似度）
    - 缓存命中检查（避免重复向量化）
    """
    
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "papers",
        embedding_function: Optional[Any] = None
    ):
        """
        初始化 ChromaDB 存储
        
        参数:
            persist_directory: 持久化目录路径
            collection_name: 集合名称（默认 "papers"）
            embedding_function: 向量化函数（可选，用于查询时向量化）
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Scientific papers vector store"}
        )
        
        self.embedding_function = embedding_function
        
        print(f"[ChromaStore] 初始化完成")
        print(f"  持久化目录: {self.persist_directory}")
        print(f"  集合名称: {collection_name}")
        print(f"  已存储论文数: {self.collection.count()}")
    
    def upsert_papers(
        self,
        papers: List[Paper],
        embeddings: np.ndarray
    ) -> int:
        """
        增量写入论文（自动去重）
        
        参数:
            papers: 论文列表
            embeddings: 对应的向量数组 (shape: [n_papers, embedding_dim])
        
        返回:
            实际写入的论文数量
        """
        if len(papers) != len(embeddings):
            raise ValueError(f"论文数量 ({len(papers)}) 与向量数量 ({len(embeddings)}) 不匹配")
        
        # 准备数据
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []
        
        for paper, embedding in zip(papers, embeddings):
            # 使用 Paper 的唯一 ID
            paper_id = paper.get_unique_id()
            
            # 检查是否已存在（避免重复）
            if self.is_cached(paper_id):
                continue
            
            ids.append(paper_id)
            documents.append(paper.abstract)  # 存储摘要文本
            
            # 存储元数据
            metadata = {
                "title": paper.title,
                "authors": ",".join(paper.authors[:5]),  # 最多5个作者
                "source": paper.source,
                "published": paper.published.isoformat() if paper.published else None,
                "doi": paper.doi or "",
                "arxiv_id": paper.arxiv_id or "",
                "url": str(paper.url) if paper.url else "",
                "citations_count": paper.citations_count or 0
            }
            metadatas.append(metadata)
            embeddings_list.append(embedding.tolist())
        
        # 批量写入
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings_list
            )
            print(f"[ChromaStore] 新增 {len(ids)} 篇论文到缓存")
        else:
            print(f"[ChromaStore] 所有论文已存在，跳过写入")
        
        return len(ids)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        语义检索
        
        参数:
            query_embedding: 查询向量 (shape: [embedding_dim])
            top_k: 返回前 k 个结果
            filter_dict: 过滤条件（如 {"source": "arxiv"}）
        
        返回:
            检索结果列表，每项包含 {id, document, metadata, distance}
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_dict
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def is_cached(self, paper_id: str) -> bool:
        """
        检查论文是否已缓存
        
        参数:
            paper_id: 论文唯一 ID（通过 Paper.get_unique_id() 获取）
        
        返回:
            True 如果已缓存，False 否则
        """
        try:
            result = self.collection.get(ids=[paper_id])
            return len(result['ids']) > 0
        except:
            return False
    
    def get_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取论文
        
        参数:
            paper_id: 论文唯一 ID
        
        返回:
            论文数据字典，不存在返回 None
        """
        try:
            result = self.collection.get(ids=[paper_id], include=["documents", "metadatas", "embeddings"])
            if result['ids']:
                return {
                    "id": result['ids'][0],
                    "document": result['documents'][0],
                    "metadata": result['metadatas'][0],
                    "embedding": result['embeddings'][0] if result['embeddings'] else None
                }
        except:
            pass
        return None
    
    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        获取所有论文的向量
        
        返回:
            (paper_ids, embeddings) 元组
            - paper_ids: 论文 ID 列表
            - embeddings: 向量数组 (shape: [n_papers, embedding_dim])
        """
        result = self.collection.get(include=["embeddings"])
        
        paper_ids = result['ids']
        embeddings = np.array(result['embeddings'])
        
        return paper_ids, embeddings
    
    def count(self) -> int:
        """返回已存储的论文数量"""
        return self.collection.count()
    
    def reset(self):
        """清空集合（谨慎使用）"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"description": "Scientific papers vector store"}
        )
        print(f"[ChromaStore] 集合已重置")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        返回:
            统计字典
        """
        total_count = self.count()
        
        # 统计各数据源的论文数
        source_stats = {}
        if total_count > 0:
            all_data = self.collection.get(include=["metadatas"])
            for metadata in all_data['metadatas']:
                source = metadata.get('source', 'unknown')
                source_stats[source] = source_stats.get(source, 0) + 1
        
        return {
            "total_papers": total_count,
            "source_distribution": source_stats,
            "persist_directory": str(self.persist_directory)
        }
