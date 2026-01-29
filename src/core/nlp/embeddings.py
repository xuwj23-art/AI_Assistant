from __future__ import annotations
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List,Optional
from core.config import RAW_DATA_DIR
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingGenerator:
    """ 论文摘要向量生成器 """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: str = "cpu",
        batch_size: int = 64
    ):
        """
        初始化论文摘要向量生成器
        Args:
            model_name: 模型名称
            device: 设备(cpu, cuda(如果有GPU))
            batch_size: 批量大小
        """
        print(f"正在加载模型： {model_name}")
        self.model = SentenceTransformer(model_name,device=device)
        self.batch_size = batch_size
        self.model_name = model_name
        print(f"模型加载完成！向量维度：{self.model.get_sentence_embedding_dimension()}")


    def encode_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        将文本转换为向量
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
        """
        print(f"正在向量化 [{len(texts)}] 个文本...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,#注意encode中的官方参数名是show_progress_bar
            convert_to_numpy=True
        )

        print(f"已完成向量化! shape : {embeddings.shape}")

        return embeddings

    def encode_papers(
        self,
        df: pd.DataFrame,
        text_column: str = "abstract",
        show_progress: bool = True
    ) -> np.ndarray:
        """
        为论文DataFrame生成向量
        Args:
            df: 包含论文数据的 DataFrame
            text_cloumn:需要向量化的列名，默认是"abstract"
            show_progress:是否显示进度条
        """
        if text_column not in df.columns:
            raise ValueError(f"列 '{text_column}' 不存在于 DataFrame 中")
        
        abstract_texts = df[text_column].tolist()

        return self.encode_texts(
            abstract_texts,
            show_progress=show_progress
        )

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        save_path: Path
    ) -> None:
        """
        将向量保存到指定位置
        Args:
            embeddings: 需要保存的向量数组
            save_path: 保存路径(.npy格式)
        """
        save_path.parent.mkdir(parents=True,exist_ok=True)
        np.save(save_path,embeddings)
        print(f"向量已经保存到 : {save_path}")

    @staticmethod
    def load_embeddings(load_path:Path) -> np.ndarray:
        """
        从文件加载向量
        Args:
            load_path: 向量读取路径
        return:
            ny数组
        """
        if not load_path.exists():
            raise FileNotFoundError(f"向量文件不存在: {load_path}")
        
        embeddings = np.load(load_path)
        print(f"向量已加载:{load_path}, shape : {embeddings.shape}")

        return embeddings

    
def create_embeddings_for_papers(
    csv_path: Path,
    output_path: Path,
    model_name: str = "all-mpnet-base-v2",
    text_column: str = "abstract",
    batch_size: int = 64
)-> np.ndarray:
    """
    便捷函数：为论文 CSV 生成并保存向量
    
    参数:
        csv_path: 论文 CSV 文件路径
        output_path: 向量输出路径（.npy）
        model_name: 模型名称
        text_column: 要向量化的列
        batch_size: 批处理大小
    
    返回:
        生成的向量数组
    """

    # 读取数据
    print(f"读取数据 : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"共有 {len(df)} 篇论文")

    # 创建生成器
    generator = EmbeddingGenerator(
        model_name=model_name,
        batch_size=batch_size
    )

    # 生成向量
    embeddings = generator.encode_papers(df,text_column=text_column)

    # 保存向量
    generator.save_embeddings(embeddings,output_path)

    return embeddings