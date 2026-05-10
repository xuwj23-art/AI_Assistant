"""统一 LLM 接口模块（Gemini / DeepSeek / 规则回退）"""
from .provider import LLMProvider, get_llm_provider

__all__ = ["LLMProvider", "get_llm_provider"]
