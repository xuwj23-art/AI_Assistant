"""
AI论文摘要总结模块
使用本地模型生成论文摘要
"""
from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


class PaperSummarizer:
    """
    论文摘要总结器
    使用BART或T5等模型生成简洁摘要
    """

    def __init__(
            self,
            model_name: str = "facebook/bart-large-cnn",
            device: str = "cpu",
            max_length: int = 150,
            min_length: int = 50
    ):
        """
        初始化摘要生成器

        参数:
            model_name: 模型名称
                - "facebook/bart-large-cnn" (约1.6GB) - 效果好，但较大
                - "t5-small" (约250MB) - 较小，速度快
                - "t5-base" (约900MB) - 平衡版
            device: 设备 (cpu/cuda)
            max_length: 最大摘要长度
            min_length: 最小摘要长度
        """
        print(f"正在加载摘要模型: {model_name}")

        # 设置环境变量避免警告
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # 创建pipeline
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" and torch.cuda.is_available() else -1
        )

        self.max_length = max_length
        self.min_length = min_length

        print(f"✅ 摘要模型加载完成！")

    def summarize(
            self,
            text: str,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None
    ) -> str:
        """
        生成摘要

        参数:
            text: 要总结的文本
            max_length: 最大摘要长度
            min_length: 最小摘要长度

        返回:
            生成的摘要
        """
        if not text or len(text) < 100:
            return text

        # 使用实例默认值或传入值
        max_len = max_length or self.max_length
        min_len = min_length or self.min_length

        try:
            # 生成摘要
            summary = self.summarizer(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False  # 使用贪婪搜索，保证确定性
            )

            return summary[0]['summary_text']

        except Exception as e:
            print(f"摘要生成失败: {e}")
            # 失败时返回截断的原文
            return text[:max_len * 3] + "..."

    def summarize_batch(
            self,
            texts: List[str],
            max_length: Optional[int] = None,
            min_length: Optional[int] = None
    ) -> List[str]:
        """
        批量生成摘要

        参数:
            texts: 文本列表
            max_length: 最大摘要长度
            min_length: 最小摘要长度

        返回:
            摘要列表
        """
        results = []
        for text in texts:
            results.append(self.summarize(text, max_length, min_length))
        return results


class LightweightSummarizer:
    """
    轻量级摘要生成器
    使用更小的模型，适合资源受限环境
    """

    def __init__(self, model_name: str = "t5-small"):
        """
        初始化轻量级摘要器
        """
        print(f"正在加载轻量级摘要模型: {model_name}")

        from transformers import T5Tokenizer, T5ForConditionalGeneration

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        print(f"✅ 轻量级摘要模型加载完成！")

    def summarize(self, text: str, max_length: int = 150) -> str:
        """
        生成摘要（T5模型需要添加"summarize: "前缀）
        """
        if not text or len(text) < 100:
            return text

        # T5模型需要特定前缀
        input_text = "summarize: " + text

        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


# 工厂函数
def create_summarizer(
        model_type: str = "light",
        device: str = "cpu"
) -> Any:
    """
    创建摘要生成器实例

    参数:
        model_type: 模型类型
            - "light": 轻量级 (t5-small, ~250MB)
            - "balanced": 平衡版 (t5-base, ~900MB)
            - "full": 完整版 (bart-large-cnn, ~1.6GB)
        device: 设备

    返回:
        摘要生成器实例
    """
    if model_type == "light":
        return LightweightSummarizer("t5-small")
    elif model_type == "balanced":
        return LightweightSummarizer("t5-base")
    elif model_type == "full":
        return PaperSummarizer("facebook/bart-large-cnn", device=device)
    else:
        return LightweightSummarizer("t5-small")