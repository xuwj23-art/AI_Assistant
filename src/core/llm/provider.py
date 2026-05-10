"""
统一 LLM 提供者
- GeminiProvider  : REST API via httpx，用于主题命名（免费层）
- DeepSeekProvider: OpenAI 兼容接口，用于 RAG 智能问答
- 规则回退        : 无 API key 时本地格式化关键词
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 规则格式化器（离线，无需模型）
# ---------------------------------------------------------------------------

# 需要全大写的已知缩写/模型名
_ACRONYMS: set[str] = {
    "bert", "gpt", "t5", "roberta", "xlnet", "albert", "distilbert", "electra",
    "llama", "mistral", "falcon", "gemini", "claude", "chatgpt",
    "llm", "llms", "vlm", "vlms", "mllm", "mllms", "slm",
    "nlp", "nlu", "nmt", "qa",
    "cv", "ml", "ai", "agi", "rag", "cnn", "rnn", "lstm", "gru", "gnn", "vae", "gan",
    "clip", "vit", "swin", "moe", "rl", "rlhf", "lora", "peft", "sft", "dpo",
    "tts", "asr", "ocr", "kg", "ner", "pos", "mt", "kgc",
    "api", "url", "json", "html", "xml", "sql", "iot", "gpu", "cpu", "tpu",
    "uav", "ad", "av",
}

_STOPWORDS: set[str] = {
    "the", "a", "an", "of", "in", "and", "or", "for", "with", "on", "at",
    "to", "by", "from", "using", "based", "via", "its", "this", "that",
}


def _format_keyword(kw: str) -> str:
    """将单个关键词格式化为展示形式（下划线/空格→分词，识别专有名词）"""
    # 同时按下划线、连字符、空格切分，覆盖 BERTopic 输出的多词关键词（如 "multilingual bert"）
    parts = re.split(r"[_\-\s]+", kw.strip())
    result = []
    for p in parts:
        if not p:
            continue
        if p.lower() in _ACRONYMS:
            result.append(p.upper())
        else:
            result.append(p.capitalize())
    return " ".join(result)


def rule_based_name(keywords: list[str], max_kws: int = 5) -> str:
    """
    规则格式化：过滤停用词 → 去重 → 识别专有名词大写 → 拼接
    """
    seen: set[str] = set()
    filtered: list[str] = []
    for kw in keywords:
        norm = kw.lower().replace("_", " ")
        if norm not in _STOPWORDS and norm not in seen:
            seen.add(norm)
            filtered.append(kw)
        if len(filtered) >= max_kws:
            break
    return " · ".join(_format_keyword(k) for k in filtered)


# ---------------------------------------------------------------------------
# Gemini REST 调用（主题命名）
# ---------------------------------------------------------------------------

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _call_gemini(model: str, api_key: str, prompt: str, max_tokens: int = 60) -> str:
    """调用 Gemini generateContent REST API，返回生成文本"""
    url = f"{_GEMINI_BASE}/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.25,
            "topP": 0.9,
        },
    }
    # 30s 超时（免费层的首次冷启动可能较慢）；连接 5s
    timeout = httpx.Timeout(30.0, connect=5.0)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, params={"key": api_key}, json=payload)
        resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"Gemini returned no candidates: {data}")
    return candidates[0]["content"]["parts"][0]["text"].strip()


def _gemini_topic_prompt(keywords: list[str]) -> str:
    kw_str = ", ".join(keywords[:10])
    return (
        "You are an expert ML researcher labelling a topic cluster.\n"
        f"Cluster keywords (sorted by importance): {kw_str}\n\n"
        "Task: write a SHORT, SPECIFIC research topic title in English.\n"
        "Requirements:\n"
        "- 3 to 7 words, in title case\n"
        "- Capitalize known model/method acronyms exactly: BERT, GPT, T5, LLM, LLMs, RAG, CNN, RNN, GNN, VAE, GAN, CLIP, ViT, RLHF\n"
        "- Use proper academic phrasing (e.g. 'Multilingual BERT for Cross-Lingual Transfer')\n"
        "- Do NOT start with 'Research', 'Topic', 'Study of', 'A ', 'An ', or 'The '\n"
        "- Do NOT include quotes, punctuation at end, or explanations\n\n"
        "Output ONLY the title on one line."
    )


# ---------------------------------------------------------------------------
# DeepSeek / OpenAI-compatible 调用（RAG 问答）
# ---------------------------------------------------------------------------


def _call_openai_compat(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 800,
    temperature: float = 0.4,
    request_timeout: float = 60.0,
) -> str:
    """
    通过 httpx 直接调用 OpenAI-兼容 Chat Completions REST API。

    避免使用 openai SDK，因为 openai 1.12 与 httpx 0.28+ 存在 'proxies'
    参数兼容性问题；直接走 REST 协议更稳定，且零额外依赖。
    """
    base = base_url.rstrip("/")
    # DeepSeek 接受 .../v1 或裸域名两种 base_url，统一拼到 /v1/chat/completions
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    timeout = httpx.Timeout(request_timeout, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"OpenAI-compat 返回 choices 为空: {data}")
    msg = choices[0].get("message") or {}
    content = msg.get("content", "")
    return content.strip()


# ---------------------------------------------------------------------------
# 统一 LLMProvider
# ---------------------------------------------------------------------------


class LLMProvider:
    """
    统一 LLM 接口。
    - name_topic(keywords) : 用 Gemini 命名主题（失败 → 规则回退）
    - chat_completion(msgs): 用 DeepSeek 生成对话回答（失败 → None）
    """

    def __init__(
        self,
        gemini_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        deepseek_key: Optional[str] = None,
        deepseek_model: Optional[str] = None,
        deepseek_base_url: Optional[str] = None,
    ):
        self._gemini_key = gemini_key
        self._gemini_model = gemini_model or "gemini-2.0-flash-lite"
        self._deepseek_key = deepseek_key
        self._deepseek_model = deepseek_model or "deepseek-chat"
        self._deepseek_base_url = deepseek_base_url or "https://api.deepseek.com"

    # ---- 能力探测 ----

    @property
    def gemini_available(self) -> bool:
        return bool(self._gemini_key)

    @property
    def deepseek_available(self) -> bool:
        return bool(self._deepseek_key)

    # ---- 主题命名 ----

    def name_topic(self, keywords: list[str]) -> str:
        """
        生成主题标题。
        优先 Gemini API；失败/无 key 则用规则格式化器。
        """
        if self.gemini_available:
            prompt = _gemini_topic_prompt(keywords)
            # 最多尝试两次（首次冷启动可能慢）
            for attempt in range(2):
                try:
                    result = _call_gemini(self._gemini_model, self._gemini_key, prompt)
                    cleaned = self._clean_title(result)
                    if cleaned and 3 <= len(cleaned) <= 120:
                        return cleaned
                    logger.warning("Gemini 返回内容异常: %r", result)
                    break
                except Exception as exc:
                    logger.warning(
                        "Gemini 主题命名失败 (attempt %d): %s",
                        attempt + 1,
                        exc,
                    )
        return rule_based_name(keywords)

    @staticmethod
    def _clean_title(text: str) -> str:
        """去除 Gemini 可能输出的前缀/标点/引号"""
        if not text:
            return ""
        t = text.strip().strip('"\u201C\u201D\u2018\u2019').rstrip(".,;:!?\u3002").strip()
        # 去掉常见冗余前缀
        for prefix in ("Title:", "Topic:", "title:", "topic:"):
            if t.startswith(prefix):
                t = t[len(prefix):].strip()
        # 单行约束
        if "\n" in t:
            t = t.split("\n", 1)[0].strip()
        return t

    # ---- RAG 对话 ----

    def chat_completion(
        self,
        messages: list[dict],
        max_tokens: int = 1500,
        temperature: float = 0.4,
    ) -> Optional[str]:
        """
        生成对话回答（DeepSeek）。
        失败或无 key 时返回 None（调用方决定回退逻辑）。

        重试策略（控制 quota）：
        - 仅对 *瞬时错误*（timeout / 5xx / 429）重试一次，间隔 1.5s
        - 其它错误（4xx 鉴权、payload 不合法等）立即放弃，不浪费 quota

        注意：deepseek-v4-pro 等推理模型会用 max_tokens 同时容纳
        reasoning_content + content，建议 ≥ 1500 才能稳定输出完整答案。
        """
        if not self.deepseek_available:
            return None

        import time

        max_attempts = 2  # 最多 1 次重试
        last_exc: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                return _call_openai_compat(
                    api_key=self._deepseek_key,
                    base_url=self._deepseek_base_url,
                    model=self._deepseek_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code if exc.response is not None else 0
                last_exc = exc
                # 仅对 429 / 5xx 重试；4xx 鉴权或 payload 错误直接放弃
                if code == 429 or 500 <= code < 600:
                    logger.warning(
                        "DeepSeek HTTP %s (attempt %d/%d): %s",
                        code, attempt + 1, max_attempts, exc,
                    )
                    if attempt + 1 < max_attempts:
                        time.sleep(1.5 * (attempt + 1))  # 1.5s, 3.0s
                        continue
                logger.error("DeepSeek HTTP %s 不可重试，放弃: %s", code, exc)
                return None
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                logger.warning(
                    "DeepSeek 网络异常 (attempt %d/%d): %s",
                    attempt + 1, max_attempts, exc,
                )
                if attempt + 1 < max_attempts:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return None
            except Exception as exc:
                logger.error("DeepSeek 调用失败（不重试）: %s", exc)
                return None

        if last_exc is not None:
            logger.error("DeepSeek 多次重试仍失败: %s", last_exc)
        return None

    def describe(self) -> dict:
        return {
            "gemini": {
                "available": self.gemini_available,
                "model": self._gemini_model if self.gemini_available else "N/A",
            },
            "deepseek": {
                "available": self.deepseek_available,
                "model": self._deepseek_model if self.deepseek_available else "N/A",
                "base_url": self._deepseek_base_url if self.deepseek_available else "N/A",
            },
        }


# ---------------------------------------------------------------------------
# 单例工厂（从环境变量读取配置）
# ---------------------------------------------------------------------------

_provider: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """返回全局单例 LLMProvider（从 .env 读取配置）"""
    global _provider
    if _provider is not None:
        return _provider

    # 尝试加载 .env
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(override=False)
    except ImportError:
        pass

    _provider = LLMProvider(
        gemini_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_TOPIC_MODEL", "gemini-2.0-flash-lite"),
        deepseek_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    info = _provider.describe()
    logger.info(
        "LLMProvider 初始化完成 | Gemini=%s(%s) | DeepSeek=%s(%s)",
        info["gemini"]["available"],
        info["gemini"]["model"],
        info["deepseek"]["available"],
        info["deepseek"]["model"],
    )
    return _provider


def reset_provider() -> None:
    """重置单例（测试用）"""
    global _provider
    _provider = None
