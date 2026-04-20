"""
主题服务类

负责主题数据的加载和处理
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from .models import TopicResponse, TopicKeyword, PaperResponse
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
        
        # 按主题分组统计
        topic_stats = df.groupby('topic_id').agg({
            'id': 'count',
            'topic_name': 'first'
        }).reset_index()
        
        topic_stats.columns = ['topic_id', 'paper_count', 'topic_name']
        
        # 过滤掉噪声主题（topic_id == -1）
        topic_stats = topic_stats[topic_stats['topic_id'] != -1]
        
        # 构建响应列表
        topics = []
        for _, row in topic_stats.iterrows():
            topic_id = int(row['topic_id'])
            keywords = self._get_topic_keywords(topic_id)
            
            topics.append(TopicResponse(
                topic_id=topic_id,
                topic_name=str(row['topic_name']),
                keywords=keywords,
                paper_count=int(row['paper_count'])
            ))
        
        # 按论文数量降序排列
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
        
        # 查找该主题的论文
        topic_papers = df[df['topic_id'] == topic_id]
        
        if topic_papers.empty:
            return None
        
        # 获取主题名称和论文数量
        topic_name = topic_papers['topic_name'].iloc[0]
        paper_count = len(topic_papers)
        
        # 获取关键词
        keywords = self._get_topic_keywords(topic_id)
        
        return TopicResponse(
            topic_id=topic_id,
            topic_name=str(topic_name),
            keywords=keywords,
            paper_count=paper_count
        )
    
    def get_papers_by_topic(self, topic_id: int, limit: int = 50) -> List[PaperResponse]:
        """
        获取某个主题下的所有论文
        
        参数:
            topic_id: 主题ID
            limit: 最多返回数量
        
        返回:
            论文列表
        """
        df = self._load_data()
        
        # 筛选该主题的论文
        topic_papers = df[df['topic_id'] == topic_id].head(limit)
        
        # 转换为 PaperResponse 列表
        papers = []
        for _, row in topic_papers.iterrows():
            papers.append(self._row_to_paper(row))
        
        return papers
    
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
        """
        将DataFrame行转换为PaperResponse
        
        参数:
            row: DataFrame的一行
        
        返回:
            PaperResponse对象
        """
        # 处理作者列表
        authors = row.get('authors', '[]')
        if isinstance(authors, str):
            try:
                authors = ast.literal_eval(authors)
            except:
                authors = [authors]
        
        # 处理分类
        categories = row.get('categories', '[]')
        if isinstance(categories, str):
            try:
                categories = ast.literal_eval(categories)
            except:
                categories = [categories]
        
        # 处理ID
        id_val = row.get('id', '')
        if pd.isna(id_val):
            id_val = ''
        else:
            id_val = str(id_val)
            if 'arxiv.org/abs/' in id_val:
                id_val = id_val.split('arxiv.org/abs/')[-1]
                if 'v' in id_val:
                    id_val = id_val.split('v')[0]
        
        # 处理PDF URL
        pdf_url = row.get('pdf_url')
        if pd.isna(pdf_url) or pdf_url == '' or str(pdf_url).strip() == '':
            url = row.get('url', '')
            if url and 'arxiv.org/abs/' in str(url):
                pdf_url = str(url).replace('/abs/', '/pdf/') + '.pdf'
            else:
                pdf_url = None
        else:
            pdf_url = str(pdf_url).strip()
        
        # 处理必填字段
        title_val = row.get('title', '')
        abstract_val = row.get('abstract', '')
        if pd.isna(title_val):
            title_val = ''
        if pd.isna(abstract_val):
            abstract_val = ''
        
        # 处理日期
        published_val = row.get('published')
        if pd.isna(published_val):
            from datetime import datetime
            published_val = datetime(2000, 1, 1)
        
        return PaperResponse(
            id=str(id_val),
            title=str(title_val),
            authors=authors if isinstance(authors, list) else [],
            abstract=str(abstract_val),
            published=published_val,
            pdf_url=pdf_url,
            categories=categories if isinstance(categories, list) else []
        )
    
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
