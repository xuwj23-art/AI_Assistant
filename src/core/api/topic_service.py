"""
主题服务类

负责主题数据的加载和处理
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from .models import TopicResponse, TopicKeyword, PaperResponse
from .paper_utils import row_to_paper
import ast


class TopicService:
    """
    主题服务类
    
    负责加载和管理主题相关数据
    """
    
    def __init__(self, data_path: str, model_path: Optional[str] = None):
        """
        初始化主题服务
        
        参数:
            data_path: 带主题标签的论文CSV文件路径
            model_path: BERTopic模型路径（可选）
        """
        self.data_path = Path(data_path)
        self.model_path = Path(model_path) if model_path else None
        self._df: Optional[pd.DataFrame] = None
        self._topic_model = None
        
    def _load_data(self) -> pd.DataFrame:
        """
        加载论文数据（带主题标签）
        """
        if self._df is None:
            if not self.data_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
            
            self._df = pd.read_csv(
                self.data_path,
                parse_dates=['published', 'updated']
            )
            print(f"[TopicService] 加载了 {len(self._df)} 篇论文数据")
            
            # 检查是否有主题列
            if 'topic_id' not in self._df.columns:
                print("[TopicService] 警告: 数据中没有 topic_id 列")
                self._df['topic_id'] = -1
            if 'topic_name' not in self._df.columns:
                print("[TopicService] 警告: 数据中没有 topic_name 列")
                self._df['topic_name'] = 'Unknown'
        
        return self._df
    
    def _load_model(self):
        """
        加载 BERTopic 模型（可选）
        """
        if self._topic_model is None and self.model_path and self.model_path.exists():
            from core.nlp.topic_modeling import TopicModeler
            self._topic_model = TopicModeler.load_model(self.model_path, verbose=False)
            print(f"[TopicService] 已加载 BERTopic 模型: {self.model_path}")
        return self._topic_model
    
    def get_all_topics(self) -> List[TopicResponse]:
        """
        获取所有主题列表

        返回:
            主题列表
        """
        df = self._load_data()

        agg_spec = {'id': 'count', 'topic_name': 'first'}
        if 'topic_label' in df.columns:
            agg_spec['topic_label'] = 'first'

        topic_stats = df.groupby('topic_id').agg(agg_spec).reset_index()

        # 重命名列保持顺序
        cols = ['topic_id', 'paper_count', 'topic_name']
        if 'topic_label' in df.columns:
            cols.append('topic_label')
        topic_stats.columns = cols

        # 过滤掉噪声主题（topic_id == -1）
        topic_stats = topic_stats[topic_stats['topic_id'] != -1]

        topics = []
        for _, row in topic_stats.iterrows():
            topic_id = int(row['topic_id'])
            keywords = self._get_topic_keywords(topic_id)
            topic_label = None
            if 'topic_label' in row and pd.notna(row.get('topic_label')):
                tl = str(row['topic_label']).strip()
                if tl and tl.lower() != 'nan':
                    topic_label = tl

            topics.append(TopicResponse(
                topic_id=topic_id,
                topic_name=str(row['topic_name']),
                topic_label=topic_label,
                keywords=keywords,
                paper_count=int(row['paper_count'])
            ))

        topics.sort(key=lambda x: x.paper_count, reverse=True)
        return topics
    
    def get_topic_by_id(self, topic_id: int) -> Optional[TopicResponse]:
        """
        根据ID获取主题详情

        参数:
            topic_id: 主题ID

        返回:
            主题详情，如果不存在返回None
        """
        df = self._load_data()
        topic_papers = df[df['topic_id'] == topic_id]
        if topic_papers.empty:
            return None

        topic_name = topic_papers['topic_name'].iloc[0]
        topic_label = None
        if 'topic_label' in topic_papers.columns:
            tl_val = topic_papers['topic_label'].iloc[0]
            if pd.notna(tl_val):
                tl = str(tl_val).strip()
                if tl and tl.lower() != 'nan':
                    topic_label = tl

        paper_count = len(topic_papers)
        keywords = self._get_topic_keywords(topic_id)

        return TopicResponse(
            topic_id=topic_id,
            topic_name=str(topic_name),
            topic_label=topic_label,
            keywords=keywords,
            paper_count=paper_count
        )
    
    def get_papers_by_topic(self, topic_id: int, limit: int = 50) -> List[PaperResponse]:
        """
        获取某个主题下的论文，按"主题代表性"降序排序。

        排序逻辑：
        - 计算 topic_id 下所有论文向量的均值作为主题质心
        - 每篇论文与质心的余弦相似度 = 该论文的"主题代表性分数"
        - 分数越高，越靠前（越能代表这个主题）
        - 向量文件不可用时回退到 CSV 原顺序

        每篇 PaperResponse 会附带 `_relevance` 浮点字段（0~1），
        前端可用来显示星级或排序位置。
        """
        df = self._load_data()

        # 筛选该主题的论文（保留全部，再用相关性排）
        topic_df = df[df['topic_id'] == topic_id].copy()
        if topic_df.empty:
            return []

        # 尝试加载向量计算代表性分数
        relevance_scores = self._compute_topic_relevance(topic_id, topic_df)

        if relevance_scores is not None:
            topic_df = topic_df.assign(_relevance=relevance_scores)
            topic_df = topic_df.sort_values("_relevance", ascending=False)
        topic_df = topic_df.head(limit)

        papers = []
        for _, row in topic_df.iterrows():
            paper = self._row_to_paper(row)
            if "_relevance" in row.index and pd.notna(row.get("_relevance")):
                paper.relevance = float(row["_relevance"])
            papers.append(paper)
        return papers

    def _compute_topic_relevance(self, topic_id: int, topic_df: pd.DataFrame):
        """
        计算 topic_df 中每篇论文与该主题质心的余弦相似度。

        - 论文向量从 data/processed/embeddings.npy（与 CSV 行对齐）加载
        - 质心 = 该 topic_id 下所有向量的均值（再 L2 归一化）
        - 噪声主题 -1 直接返回 None（不参与代表性排序）
        - 失败 / 文件缺失 → 返回 None，调用方按原顺序处理
        """
        if topic_id == -1:
            return None
        try:
            import numpy as np
            from ..config import PROCESSED_DATA_DIR

            emb_path = PROCESSED_DATA_DIR / "embeddings.npy"
            if not emb_path.exists():
                return None

            embeddings = np.load(str(emb_path))
            full_df = self._load_data()
            if len(embeddings) != len(full_df):
                # 向量文件与 CSV 行数不一致，无法对齐
                return None

            # 取该主题对应的向量索引
            topic_indices = topic_df.index.to_numpy()
            topic_vectors = embeddings[topic_indices].astype(np.float32)

            # 质心 + L2 归一化
            centroid = topic_vectors.mean(axis=0)
            cnorm = np.linalg.norm(centroid) + 1e-12
            centroid_unit = centroid / cnorm

            vnorms = np.linalg.norm(topic_vectors, axis=1, keepdims=True) + 1e-12
            vec_unit = topic_vectors / vnorms

            # 余弦相似度（0-1 区间，归一化后 = 点积）
            scores = vec_unit @ centroid_unit
            # 把 [-1,1] 映射到 [0,1] 让前端展示更直观
            return ((scores + 1.0) / 2.0).tolist()
        except Exception as e:
            print(f"[TopicService] 计算主题代表性失败（{e}），回退原顺序")
            return None
    
    def _get_topic_keywords(self, topic_id: int, top_n: int = 10) -> List[TopicKeyword]:
        """
        获取主题的关键词
        
        参数:
            topic_id: 主题ID
            top_n: 返回前N个关键词
        
        返回:
            关键词列表
        """
        # 尝试从模型中获取
        model = self._load_model()
        if model:
            try:
                topic_words = model.get_topic(topic_id, top_n=top_n)
                if topic_words:
                    return [
                        TopicKeyword(word=word, score=float(score))
                        for word, score in topic_words
                    ]
            except Exception as e:
                print(f"[TopicService] 从模型获取关键词失败: {e}")
        
        # 如果模型不可用，从主题名称中提取
        df = self._load_data()
        topic_papers = df[df['topic_id'] == topic_id]
        
        if topic_papers.empty:
            return []
        
        topic_name = topic_papers['topic_name'].iloc[0]
        
        # 解析主题名称（格式如 "transformer_attention_model"）
        words = str(topic_name).split('_')
        
        # 生成简单的关键词（分数递减）
        keywords = []
        for i, word in enumerate(words[:top_n]):
            score = 1.0 - (i * 0.1)  # 第一个词1.0，第二个0.9，依次递减
            score = max(score, 0.3)  # 最低0.3
            keywords.append(TopicKeyword(word=word, score=score))
        
        return keywords
    
    def _row_to_paper(self, row: pd.Series) -> PaperResponse:
        """将 DataFrame 行转换为 PaperResponse（委托给公共工具）。"""
        return row_to_paper(row)
    
    def get_stats(self) -> Dict:
        """
        获取主题统计信息
        
        返回:
            统计信息字典
        """
        df = self._load_data()
        
        # 统计每个主题的论文数
        topic_counts = df[df['topic_id'] != -1].groupby('topic_id').size().to_dict()
        
        return {
            "total_topics": len(topic_counts),
            "total_papers": len(df),
            "papers_with_topic": len(df[df['topic_id'] != -1]),
            "outliers": len(df[df['topic_id'] == -1]),
            "topic_distribution": topic_counts
        }
