from __future__ import annotations
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from typing import List,Optional,Tuple
import pandas as pd
import numpy as np


class TopicModeler:
    """ 论文主题建模器 """
    # # 384维 → 5维
    # from umap import UMAP

    # reducer = UMAP(
    #     n_components=5,      # 降到 5 维
    #     n_neighbors=15,      # 考虑 15 个邻居
    #     min_dist=0.0,        # 最小距离
    #     metric='cosine'      # 使用余弦距离
    # )

    # low_dim_embeddings = reducer.fit_transform(embeddings)

    # from hdbscan import HDBSCAN

    # clusterer = HDBSCAN(
    #     min_cluster_size=10,     # 最小簇大小
    #     min_samples=5,           # 最小样本数
    #     metric='euclidean',      # 距离度量
    #     cluster_selection_method='eom'  # 簇选择方法
    # )

    # clusters = clusterer.fit_predict(low_dim_embeddings)
    def __init__(
        self,
        n_components: int = 5,
        n_neighbors: int = 15,
        min_topic_size: int = 10,
        language: str = "english",
        embedding_model: str = "all-mpnet-base-V2",
        verbose: bool = True
    ):
        """
        初始化主题建模器

        参数：
            n_components: 降维维度
            n_neighbors: 邻居数
            min_topic_size: 最小主题大小
            language: 语言
            embedding_model: Sentence Transformer 模型名称
            verbose: 是否显示信息
        """
        self.verbose = verbose

        if self.verbose:
            print(f"初始化主题建模器...")
            print(f" 嵌入模型:{embedding_model} ")
            print(f" 最小主题大小{min_topic_size}")
        
        # 嵌入模型：
        self.embedding_model = SentenceTransformer(embedding_model)

        # UMAP降维:
        self.umap_model = UMAP(
            n_components=5,      # 降到 5 维
            n_neighbors=15,      # 考虑 15 个邻居
            min_dist=0.0,        # 最小距离
            metric='cosine'      # 使用余弦距离
        )

        # HDBSCAN 聚类
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=10,     # 最小簇大小
            min_samples=5,           # 最小样本数
            metric='euclidean',      # 距离度量
            cluster_selection_method='eom'  # 簇选择方法
        )

        # 词频统计/向量化器
        self.vectorizer_model = CountVectorizer(
            stop_words="english" if language == "english" else None,
            ngram_range=(1,2),# 1-2个词的短语,例如"Machine Learning"
            min_df = 2 # 至少出现两次
        )

        # BERTopic 模型
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            language=language,
            calculate_probabilities=False,
            verbose=verbose
        )

        if self.verbose:
            print("主题建模器初始化完成！")
    
    def fit(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        训练主题模型

        参数:
            documents: 文档列表
            embeddings: 预计算的向量(可选)
        
        返回：
            (主题标签列表，主题概率矩阵)
        
        使用说明：
        
        """
        if self.verbose:
            print("开始训练主题模型...")
            print(f"文档数量：{len(documents)}")
        
        # 训练
        topics,probs = self.topic_model.fit_transform(
            documents,
            embeddings=embeddings
        )
        """
        参数：
            documents: 用于拟合的文档列表
            嵌入向量：预训练的文档嵌入向量。这些可以用于替代 sentence-transformer 模型
            图像：要拟合的图像路径列表或图像本身
            y：用于（半）监督建模的目标类别。如果没有类别，请使用-1指定特定实例。
        返回：
            predictions: 每个文档的主题预测
            probabilities: 每个文档分配主题的概率。
        如果 BERTopic 中的`calculate_probabilities`设置为 True，那么
        它会计算所有文档中所有主题的概率而不仅仅是分配的主题。但是，这会减慢计算速度并可能增加内存使用。
        """
        if self.verbose:
            n_topics = len(set(topics)) - (1 if -1 in topics else 0)
            n_noise = list(topics).count(-1)

            print("训练完成!")
            print(f"主题数: {n_topics}")
            print(f"噪声文档数: {n_noise}")

            return topics,probs
            
    def get_topic_info(self) -> pd.DataFrame:
        """
        获取主题信息
        
        返回:
            包含主题信息的 DataFrame
        """
        return self.topic_model.get_topic_info()

    def get_topic(
        self,
        topic_id: int,
        top_n: int = 10
    ) -> List[Tuple(str,float)]:
        """
        获取主题的关键词
        fit()函数运行结束后,BERTopic 已经在内部把几千篇论文分好了类，
        并且给每篇论文贴上了一个数字标签(如 0, 1, 2...)

        参数：
            topic_id: 主题ID
            top_n: 返回前N个关键词
        返回:
            [(词,分数),...]列表
        举例:
            输入: topic_id=5, top_n=3
            输出: [('llm', 0.85), ('generative', 0.72), ('transformer', 0.65)]
        """
        # get_topic()查看某个特定主题（Topic ID）到底是由哪些词构成的
        topic_words = self.topic_model.get_topic(topic_id)
        if not topic_words:
            return []
        return topic_words[:top_n]

    def get_topic_name(
        self,
        topic_id: int,
        max_words: int = 3
    ) -> str:
        """
        生成主题名称
        get_topic() 拿到关键词后,把topic_id对应的词的前几个词拼接起来,
        自动生成一个标签(-1 代表噪声)

        参数:
            topic_id: 主题ID
            max_words: 最多使用几个关键词
        返回:
            主题名称字符串
        举例:
            输入: topic_id=5
            内部逻辑: 调用 get_topic(5) 拿到 ['llm', 'generative', ...]
            输出: "llm_generative_transformer"
        """
        if topic_id == -1:
            return "Outliers"
        topic_words = self.get_topic(topic_id,top_n=max_words)
        if not topic_words:
            return f"Topic {topic_id}"

        words = [word for word,score in topic_words]
        return "_".join(words)

    def reduce_topics(
        self,
        documents: List[str],
        n_topics: int
    ) -> None:
        """
        减少主题数量(合并相似主题)
        函数会计算主题之间的相似度，把最相似的主题两两合并，直到剩下 `n_topics` 个为止
        
        参数:
            documents: 原始文档
            n_topics: 目标主题数
        说明:
            一旦合并了主题(比如 Topic 5 和 Topic 8 合并成新 Topic 5),
            模型需要重新计算新主题的关键词(c-TF-IDF),
            这需要再次读取原始文本,所以需要传入documents。
        """
        if self.verbose:
            print(f"减少主题数量到 {n_topics}...")
        
        self.topic_model.reduce_topics(documents,nr_topics=n_topics)

        if self.verbose:
            print(f"主题数量已经减少到 {n_topics}")

    def save_model(self,save_path: Path) -> None:
        """
        保存模型
        参数:
            save_path:保存路径
        """
        save_path.parent.mkdir(parents=True,exist_ok=True)
        self.topic_model.save(str(save_path))

        if self.verbose:
            print(f"模型已保存到: {save_path}")

    @staticmethod
    def load_model(load_path: Path,verbose: bool = True)->TopicModeler:
        """
        加载模型
        参数:
            load_path: 模型路径
            verbose: 是否显示信息
        返回:
            TopicModeler模型
        """
        if not load_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {load_path}")
        
        if verbose:
            print(f"加载模型: {load_path}")
        
        modeler = TopicModeler(verbose=verbose)

        modeler.topic_model = BERTopic.load(str(load_path))

        if verbose:
            print("模型加载完成")

        return modeler

def train_topic_model_for_papers(
    csv_path: Path,
    embeddings_path: Path,
    output_path: Path,
    text_column: str = "abstract",
    min_topic_size: int = 10,
    n_topics: Optional[int] = None
) -> Tuple[TopicModeler,pd.DataFrame]:
    """
    便捷函数：为论文训练主题模型
    
    参数:
        csv_path: 论文 CSV 路径
        embeddings_path: 向量文件路径
        output_path: 模型保存路径
        text_column: 文本列名
        min_topic_size: 最小主题大小
        n_topics: 目标主题数（可选，用于减少主题）
    
    返回:
        (TopicModeler 实例, 带主题标签的 DataFrame)
    """
    topic_name_map = {}
    # 读取csv文档：Documents
    df = pd.read_csv(csv_path)
    documents = df[text_column].fillna("").tolist()

    # 实例化模型，并传入参数
    embeddings = np.load(embeddings_path)
    topic_model = TopicModeler(min_topic_size=min_topic_size)

    # 开始训练
    topic_list,array = topic_model.fit(documents,embeddings)

    # 最多n_topics个主题
    if n_topics is not None:
        print(f"[INFO] 正在将主题合并至 {n_topics} 个...")
        topic_model.reduce_topics(documents, n_topics=n_topics)
        # 合并后 ID 变了，必须更新 topic_list，否则后续映射全是错的
        topic_list = topic_model.topic_model.topics_
    
    # 将topic_list 插入DF表格中
    df['topic_id'] = topic_list

    # 处理topic，自动标签化
    unique_topics = set(topic_list) 
    for topic in unique_topics:
        topic_name = topic_model.get_topic_name(topic)
        topic_name_map[topic] = topic_name

    # 将该字典其映射df表格
    df['topic_name'] = df['topic_id'].map(topic_name_map)

    # 保存模型
    topic_model.save_model(output_path)
    
    # 10. 保存带结果的 CSV
    output_csv_path = output_path.parent / f"{output_path.stem}_papers.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"结果已保存至: {output_csv_path}")

    return topic_model, df