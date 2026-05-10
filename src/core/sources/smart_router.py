"""
智能来源路由 (Smart Source Routing)

根据查询关键词自动判断研究领域，动态调整各数据源的抓取权重：
  - arXiv 擅长：CS / AI / ML / Math / Stat / Physics
  - OpenAlex 擅长：Bio / Med / Chem / Soc / Econ 等全学科

实现思路（无新依赖）:
  复用项目已加载的 SentenceTransformer (`all-mpnet-base-v2`)
  对查询和"学科原型句"计算余弦相似度，再聚合到来源权重。
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from ..nlp.embeddings import EmbeddingGenerator


# ===== 学科原型库（每个原型用一句简短描述，向量化后做语义匹配）=====
# 每条目: (学科标签, 偏向数据源, 描述句)
_DOMAIN_PROTOTYPES: List[Tuple[str, str, str]] = [
    # ----- arXiv 强项 -----
    ("computer_science", "arxiv", "computer science software algorithms data structures"),
    ("machine_learning", "arxiv", "machine learning deep learning neural networks training"),
    ("nlp", "arxiv", "natural language processing transformers language models"),
    ("computer_vision", "arxiv", "computer vision image recognition object detection"),
    ("artificial_intelligence", "arxiv", "artificial intelligence reasoning agents"),
    ("mathematics", "arxiv", "mathematics theorem proof topology algebra geometry"),
    ("statistics", "arxiv", "statistics probability bayesian inference hypothesis testing"),
    ("physics", "arxiv", "physics quantum particle relativity cosmology"),
    # ----- OpenAlex 强项（覆盖更广的传统学科）-----
    ("biology", "openalex", "biology genetics genomics cells molecular evolution"),
    ("medicine", "openalex", "medicine clinical disease patient diagnosis treatment hospital"),
    ("chemistry", "openalex", "chemistry molecules reactions compounds synthesis"),
    ("neuroscience", "openalex", "neuroscience brain cognition neurons psychology"),
    ("economics", "openalex", "economics finance markets monetary policy trade"),
    ("social_science", "openalex", "social science sociology political education policy"),
    ("environmental", "openalex", "environment climate ecology pollution sustainability"),
    ("materials", "openalex", "materials science nanotechnology polymers metals"),
]


class SmartSourceRouter:
    """
    智能来源路由器：把"查询语义"映射到"来源权重"。

    使用方法:
        router = SmartSourceRouter()
        weights = router.route("CRISPR gene editing therapy")
        # -> {"arxiv": 0.25, "openalex": 0.75}
    """

    # 原型向量缓存（模块级，避免重复编码）
    _proto_cache: Dict[str, np.ndarray] = {}

    def __init__(
        self,
        sources: List[str] = None,
        embedding_generator: EmbeddingGenerator = None,
        floor: float = 0.15,
    ):
        """
        参数:
            sources: 可用数据源列表，如 ["arxiv", "openalex"]
            embedding_generator: 可选，外部已经加载好的 EmbeddingGenerator
                （避免重复加载 mpnet 模型，~420MB）
            floor: 权重最低值（保证每个源至少分到 floor 比例，避免某源被完全忽略）
        """
        self.sources = sources or ["arxiv", "openalex"]
        self.floor = max(0.0, min(0.5, floor))
        self._gen = embedding_generator
        self._proto_vecs: np.ndarray | None = None
        self._proto_meta: List[Tuple[str, str]] = []

    def _ensure_generator(self):
        """惰性加载 EmbeddingGenerator（首次路由时才触发）"""
        if self._gen is None:
            self._gen = EmbeddingGenerator(
                model_name="all-mpnet-base-v2", batch_size=32
            )

    def _load_prototypes(self):
        """惰性向量化学科原型并缓存（整个进程只算一次）"""
        if self._proto_vecs is not None:
            return

        cache_key = "default_v1"
        if cache_key in SmartSourceRouter._proto_cache:
            self._proto_vecs = SmartSourceRouter._proto_cache[cache_key]
            self._proto_meta = [(d, s) for d, s, _ in _DOMAIN_PROTOTYPES]
            return

        self._ensure_generator()
        sentences = [s for _, _, s in _DOMAIN_PROTOTYPES]
        vecs = self._gen.encode_texts(sentences, show_progress=False)
        # L2 归一化以便用点积代替余弦
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms

        self._proto_vecs = vecs
        self._proto_meta = [(d, s) for d, s, _ in _DOMAIN_PROTOTYPES]
        SmartSourceRouter._proto_cache[cache_key] = vecs

    def route(self, query: str) -> Dict[str, float]:
        """
        根据查询计算各数据源的抓取权重。

        参数:
            query: 用户搜索关键词

        返回:
            {"arxiv": 0.6, "openalex": 0.4}  权重和≈1
        """
        if not query or not query.strip():
            return self._uniform_weights()

        if len(self.sources) <= 1:
            return {self.sources[0]: 1.0} if self.sources else {}

        self._load_prototypes()
        self._ensure_generator()

        q_vec = self._gen.encode_texts([query], show_progress=False)[0]
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)

        sims = self._proto_vecs @ q_vec  # (N,)
        # softmax 收紧分布，让最相似的原型主导
        sims = sims / max(sims.std(), 1e-3)
        exp = np.exp(sims - sims.max())
        weights_per_proto = exp / exp.sum()

        # 聚合到来源
        source_scores: Dict[str, float] = {s: 0.0 for s in self.sources}
        for w, (_, src) in zip(weights_per_proto, self._proto_meta):
            if src in source_scores:
                source_scores[src] += float(w)

        # 应用 floor：每个源至少分到 floor 比例
        return self._apply_floor(source_scores)

    def _apply_floor(self, scores: Dict[str, float]) -> Dict[str, float]:
        """对低于 floor 的源补到 floor，再重新归一化"""
        if not scores:
            return scores
        adjusted = {k: max(v, self.floor) for k, v in scores.items()}
        total = sum(adjusted.values())
        if total <= 0:
            return self._uniform_weights()
        return {k: round(v / total, 4) for k, v in adjusted.items()}

    def _uniform_weights(self) -> Dict[str, float]:
        n = len(self.sources)
        if n == 0:
            return {}
        w = round(1.0 / n, 4)
        return {s: w for s in self.sources}

    def explain(self, query: str, top_k: int = 3) -> Dict:
        """
        返回路由决策详情（调试 / UI 展示用）

        返回示例:
        {
            "weights": {"arxiv": 0.7, "openalex": 0.3},
            "top_domains": [("machine_learning", "arxiv", 0.32), ...]
        }
        """
        if not query.strip() or len(self.sources) <= 1:
            return {"weights": self._uniform_weights(), "top_domains": []}

        self._load_prototypes()
        self._ensure_generator()
        q_vec = self._gen.encode_texts([query], show_progress=False)[0]
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
        raw_sims = self._proto_vecs @ q_vec  # 余弦相似度

        # 取 top_k 最相关学科
        idx_sorted = np.argsort(-raw_sims)[:top_k]
        top_domains = [
            (self._proto_meta[i][0], self._proto_meta[i][1], float(raw_sims[i]))
            for i in idx_sorted
        ]

        return {
            "weights": self.route(query),
            "top_domains": top_domains,
        }
