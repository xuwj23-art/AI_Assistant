"""
LLM 主题命名（Topic Naming）

将 BERTopic 关键词列表转换为可读短标题。
优先级：Gemini API → 规则格式化器（本地，离线可用）

不再使用 t5-small（生成质量不佳），由 LLMProvider 统一管理。
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional


class LLMTopicNamer:
    """
    主题命名门面（Facade），保持向后兼容的单例接口。
    实际调用委托给 src.core.llm.provider.LLMProvider。

    示例:
        namer = LLMTopicNamer.get_singleton()
        namer.name_topic(["bert", "multilingual", "cross_lingual"])
        # Gemini 可用  → "Multilingual BERT for Cross-Lingual Transfer"
        # Gemini 不可用 → "BERT · Multilingual · Cross Lingual"
    """

    _instance: Optional["LLMTopicNamer"] = None
    _lock = threading.Lock()

    @classmethod
    def get_singleton(cls) -> "LLMTopicNamer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}
        self._provider = None  # 惰性加载，避免启动时网络调用

    def _get_provider(self):
        if self._provider is None:
            from ..llm.provider import get_llm_provider
            self._provider = get_llm_provider()
        return self._provider

    @staticmethod
    def _normalize_keywords(keywords: List[str], top_k: int = 8) -> List[str]:
        seen, out = set(), []
        for kw in keywords:
            if not kw:
                continue
            k = str(kw).strip().lower()
            if k and k not in seen and len(k) > 1:
                seen.add(k)
                out.append(k)
            if len(out) >= top_k:
                break
        return out

    def name_topic(
        self,
        keywords: List[str],
        max_keywords: int = 8,
        **_kwargs,
    ) -> str:
        """
        生成单个主题的可读标题。

        参数:
            keywords: 关键词列表（top-N，分数高的优先）
            max_keywords: 传入 LLM 的最多关键词数

        返回:
            短标题字符串。
        """
        kws = self._normalize_keywords(keywords, top_k=max_keywords)
        if not kws:
            return "Topic"

        cache_key = "|".join(kws)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            provider = self._get_provider()
            label = provider.name_topic(kws)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("主题命名失败，回退规则化: %s", e)
            from ..llm.provider import rule_based_name
            label = rule_based_name(kws)

        self._cache[cache_key] = label
        return label

    def name_topics_batch(
        self,
        topic_keywords_map: Dict[int, List[str]],
    ) -> Dict[int, str]:
        """批量命名多个主题"""
        return {tid: self.name_topic(kws) for tid, kws in topic_keywords_map.items()}
