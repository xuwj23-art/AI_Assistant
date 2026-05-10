"""
基于向量的主题相似度计算

- 加载 papers DataFrame + 预计算 embeddings
- 按 topic_id 聚合得到"主题质心向量"（mean pooling，L2 归一化）
- 余弦相似度 = 向量点积
- 结果会被缓存（依据数据 mtime + 主题 id 集合）

相比原 Jaccard 关键词重叠：
  ✅ 语义相似（"transformer" 与 "attention mechanism" 算同义）
  ✅ 不依赖关键词数量分布
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def compute_topic_centroids(
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    计算每个主题的质心向量（L2 归一化）

    参数:
        df: 必须含 'topic_id' 列；行序与 embeddings 第一维对齐
        embeddings: shape (N, D)

    返回:
        {topic_id: centroid_vec_norm}
    """
    if 'topic_id' not in df.columns:
        return {}

    if len(df) != embeddings.shape[0]:
        # 不对齐时只用前 min(len) 行
        n = min(len(df), embeddings.shape[0])
        df = df.iloc[:n]
        embeddings = embeddings[:n]

    centroids: Dict[int, np.ndarray] = {}
    for tid, idx in df.groupby('topic_id').indices.items():
        if int(tid) == -1:
            continue
        vecs = embeddings[idx]
        if vecs.size == 0:
            continue
        mean_vec = vecs.mean(axis=0)
        centroids[int(tid)] = _l2_normalize(mean_vec)
    return centroids


def cosine_similar_topics(
    target_id: int,
    centroids: Dict[int, np.ndarray],
    top_n: int = 5,
) -> List[Tuple[int, float]]:
    """
    返回与 target_id 余弦相似度最高的 top_n 个主题

    返回:
        [(topic_id, cosine_score), ...]   按分数降序
    """
    if target_id not in centroids:
        return []
    target_vec = centroids[target_id]
    sims: List[Tuple[int, float]] = []
    for tid, vec in centroids.items():
        if tid == target_id:
            continue
        score = float(np.dot(target_vec, vec))
        sims.append((tid, score))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]


class TopicSimilarityCache:
    """
    主题质心缓存（按 (csv_path, embeddings_path) 的 mtime 失效）
    """

    def __init__(self):
        self._signature: Optional[tuple] = None
        self._centroids: Dict[int, np.ndarray] = {}

    def _signature_for(
        self,
        csv_path: Path,
        emb_path: Path,
    ) -> tuple:
        return (
            str(csv_path),
            csv_path.stat().st_mtime if csv_path.exists() else 0,
            str(emb_path),
            emb_path.stat().st_mtime if emb_path.exists() else 0,
        )

    def get(
        self,
        csv_path: Path,
        emb_path: Path,
    ) -> Dict[int, np.ndarray]:
        """获取主题质心字典；不可用时返回空 dict"""
        if not (csv_path and csv_path.exists() and emb_path and emb_path.exists()):
            return {}

        sig = self._signature_for(csv_path, emb_path)
        if sig == self._signature and self._centroids:
            return self._centroids

        try:
            df = pd.read_csv(csv_path)
            embeddings = np.load(str(emb_path))
            self._centroids = compute_topic_centroids(df, embeddings)
            self._signature = sig
            return self._centroids
        except Exception as e:
            print(f"[TopicSimilarityCache] 计算失败: {e}")
            return {}

    def reset(self):
        self._signature = None
        self._centroids = {}


# 模块级单例
_cache = TopicSimilarityCache()


def get_topic_centroids(
    csv_path: Path,
    emb_path: Path,
) -> Dict[int, np.ndarray]:
    """获取（或重算）主题质心字典"""
    return _cache.get(csv_path, emb_path)


def reset_centroid_cache():
    """搜索流水线产出新数据后调用"""
    _cache.reset()
