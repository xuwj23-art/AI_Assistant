"""
重新生成已有 CSV 中的 topic_label 列（使用新的 LLMTopicNamer + Gemini）

用途：避免重新抓取论文。直接把现有数据里 BERTopic 抽出的关键词
（topic_name 形如 'bert_multilingual_cross'）喂给新的命名器，
然后覆盖 topic_label 列并写回 CSV。

运行：
    cd E:\AIassistant_v2
    python scripts/regenerate_topic_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from core.nlp.topic_namer import LLMTopicNamer  # noqa: E402
from core.llm.provider import get_llm_provider  # noqa: E402


def keywords_from_topic_name(topic_name: str) -> list[str]:
    """BERTopic 的 topic_name 形如 'bert_multilingual_cross_lingual'，下划线分隔关键词"""
    if not isinstance(topic_name, str):
        return []
    return [w for w in topic_name.split("_") if w]


def regenerate(csv_paths: list[Path]) -> None:
    namer = LLMTopicNamer.get_singleton()
    info = get_llm_provider().describe()
    print(f"[Provider] Gemini available: {info['gemini']['available']} (model={info['gemini']['model']})")
    print(f"[Provider] DeepSeek available: {info['deepseek']['available']} (model={info['deepseek']['model']})")
    print()

    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"[skip] {csv_path} 不存在")
            continue

        print(f"[load] {csv_path.name}")
        df = pd.read_csv(csv_path)
        if "topic_id" not in df.columns or "topic_name" not in df.columns:
            print(f"[skip] {csv_path.name} 缺少 topic_id / topic_name 列")
            continue

        # 每个 topic_id 取一份关键词
        topic_kw_map: dict = {}
        for tid, row in df.groupby("topic_id").first().iterrows():
            topic_kw_map[tid] = keywords_from_topic_name(row.get("topic_name", ""))

        # 调用命名器（噪声主题 -1 直接命名为 Outliers）
        labels: dict = {}
        for tid, kws in topic_kw_map.items():
            if tid == -1:
                labels[tid] = "Outliers"
                continue
            label = namer.name_topic(kws)
            labels[tid] = label
            print(f"  topic {tid:>2}: {kws[:4]} -> {label!r}")

        df["topic_label"] = df["topic_id"].map(labels)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[save] {csv_path.name} 已写回 ({len(labels)} 个主题)")
        print()


def main() -> None:
    processed = ROOT / "data" / "processed"
    targets = sorted(processed.glob("*_with_topics.csv"))
    if not targets:
        print("未找到任何 *_with_topics.csv")
        return
    print(f"发现 {len(targets)} 个 CSV: {[p.name for p in targets]}\n")
    regenerate(targets)


if __name__ == "__main__":
    main()
