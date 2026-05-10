"""
按需在线抓取流水线

用户在前端输入关键词后，触发完整的数据处理流程：
  1. 在线抓取论文（arXiv + OpenAlex）
  2. 保存 CSV
  3. 生成向量（embeddings）
  4. 训练主题模型（BERTopic）
  5. 保存带主题标签的 CSV

所有步骤通过回调函数报告进度，前端可实时展示。
"""
from __future__ import annotations
import time
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any

import pandas as pd
import numpy as np

from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from .sources.base import Paper
from .sources.arxiv_adapter import ArxivSource
from .sources.aggregator import MultiSourceAggregator


# 进度回调类型: (step, total_steps, message)
ProgressCallback = Callable[[int, int, str], None]


def _default_progress(step: int, total: int, msg: str):
    print(f"[{step}/{total}] {msg}")


def papers_to_dataframe(papers: List[Paper]) -> pd.DataFrame:
    """将 Paper 对象列表转换为 DataFrame"""
    data = []
    for paper in papers:
        data.append({
            'id': paper.get_unique_id(),
            'doi': paper.doi or '',
            'arxiv_id': paper.arxiv_id or '',
            'openalex_id': paper.openalex_id or '',
            'title': paper.title,
            'abstract': paper.abstract,
            'authors': str(paper.authors),
            'published': paper.published.isoformat() if paper.published else None,
            'updated': paper.updated.isoformat() if paper.updated else None,
            'source': paper.source,
            'url': str(paper.url) if paper.url else '',
            'pdf_url': str(paper.pdf_url) if paper.pdf_url else '',
            'categories': str(paper.categories),
            'citations_count': paper.citations_count or 0,
            'venue': paper.venue or ''
        })
    return pd.DataFrame(data)


def run_pipeline(
    query: str,
    max_results: int = 50,
    sources: Optional[List[str]] = None,
    progress: ProgressCallback = _default_progress,
) -> Dict[str, Any]:
    """
    执行完整的按需抓取流水线

    参数:
        query: 用户输入的搜索关键词
        max_results: 抓取论文数量（默认 50，平衡速度和质量）
        sources: 数据源列表，默认 ["arxiv"]
        progress: 进度回调函数

    返回:
        {
            "query": str,
            "csv_path": str,           # 原始 CSV 路径
            "topics_csv_path": str,     # 带主题标签的 CSV 路径
            "embeddings_path": str,     # 向量文件路径
            "model_path": str,          # BERTopic 模型路径
            "paper_count": int,         # 论文数量
            "topic_count": int,         # 主题数量
        }
    """
    if sources is None:
        sources = ["arxiv"]

    total_steps = 5
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query.replace(' ', '_')[:30]

    # ========== 步骤 1: 在线抓取论文 ==========
    progress(1, total_steps, f"正在从 {', '.join(sources)} 在线抓取论文...")

    source_objects = []
    if "arxiv" in sources:
        source_objects.append(ArxivSource())
    try:
        from .openalex.client import OpenAlexSource
        if "openalex" in sources:
            source_objects.append(OpenAlexSource())
    except ImportError:
        pass

    if not source_objects:
        source_objects.append(ArxivSource())

    # 多源场景才计算智能路由权重；单源直接全量
    smart_weights = None
    routing_explain = None
    if len(source_objects) >= 2:
        try:
            from .sources.smart_router import SmartSourceRouter
            router = SmartSourceRouter(
                sources=[s.get_source_name() for s in source_objects]
            )
            routing_explain = router.explain(query, top_k=3)
            smart_weights = routing_explain["weights"]
            print(f"[智能路由] 查询 '{query}' 推断领域: "
                  + ", ".join(f"{d}({s},{c:.2f})" for d, s, c in routing_explain["top_domains"]))
            print(f"[智能路由] 来源权重: {smart_weights}")
        except Exception as e:
            print(f"[智能路由] 计算失败，退回均分: {e}")
            smart_weights = None

    aggregator = MultiSourceAggregator(sources=source_objects, max_workers=len(source_objects))
    papers = aggregator.search_all(
        query=query,
        max_results=max_results,
        oversample_ratio=1.3,
        weights=smart_weights,
    )

    if not papers:
        raise RuntimeError(f"未能抓取到任何论文，请检查网络连接或更换关键词: {query}")

    # 统计各来源论文数量
    source_stats = aggregator.get_source_stats(papers)
    print(f"[来源统计] {source_stats}")

    # ========== 步骤 2: 保存 CSV ==========
    progress(2, total_steps, f"已抓取 {len(papers)} 篇论文，正在保存...")

    df = papers_to_dataframe(papers)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_filename = f"papers_{safe_query}_{timestamp}.csv"
    csv_path = RAW_DATA_DIR / csv_filename
    df.to_csv(csv_path, index=False, encoding='utf-8')

    # 同时保存一份标准名称的 CSV（供其他服务使用）
    standard_csv = RAW_DATA_DIR / f"arxiv_{safe_query}.csv"
    df.to_csv(standard_csv, index=False, encoding='utf-8')

    # ========== 步骤 3: 生成向量 ==========
    progress(3, total_steps, "正在生成论文向量（embedding）...")

    from .nlp.embeddings import EmbeddingGenerator

    generator = EmbeddingGenerator(model_name="all-mpnet-base-v2", batch_size=64)
    abstracts = df['abstract'].fillna("").tolist()
    embeddings = generator.encode_texts(abstracts, show_progress=True)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = PROCESSED_DATA_DIR / f"embeddings_{safe_query}.npy"
    np.save(str(embeddings_path), embeddings)

    # 同时保存标准名称
    standard_emb = PROCESSED_DATA_DIR / "embeddings.npy"
    np.save(str(standard_emb), embeddings)

    # ========== 步骤 4: 训练主题模型 ==========
    progress(4, total_steps, "正在训练主题模型（BERTopic）...")

    from .nlp.topic_modeling import TopicModeler

    # 根据论文数量动态调整 min_topic_size
    # 旧：n_papers // 10（200 篇 → 10）对同质化语料（如全是 LLM 论文）太严格，
    #     HDBSCAN 找不到 ≥10 篇的子簇，结果只剩 1-2 个主题
    # 新：n_papers // 25 + 上限 8，让 200 篇能切出 5-8 个主题
    n_papers = len(df)
    min_topic_size = max(3, min(8, max(4, n_papers // 25)))
    print(f"[auto_fetch] n_papers={n_papers}, min_topic_size={min_topic_size}")

    modeler = TopicModeler(min_topic_size=min_topic_size, verbose=True)
    topics, probs = modeler.fit(abstracts, embeddings)

    # ===== 主题数量自适应控制 =====
    n_topics_now = len(set(topics)) - (1 if -1 in topics else 0)
    target_min = 4 if n_papers >= 80 else 2
    max_topics = min(12, max(5, n_papers // 25))  # 200 篇 → 上限 8

    # 路径 A：太少 → 用 KMeans 强制分簇（数据极度同质场景）
    if n_topics_now < target_min and n_papers >= 50:
        print(
            f"[auto_fetch] HDBSCAN 仅得 {n_topics_now} 个主题（<{target_min}），"
            f"数据可能过于同质，改用 KMeans 强制分簇..."
        )
        try:
            from sklearn.cluster import KMeans
            from bertopic import BERTopic

            target_k = min(8, max(target_min + 1, n_papers // 30))
            kmeans = KMeans(n_clusters=target_k, random_state=42, n_init=10)
            kmeans_modeler = BERTopic(
                embedding_model=modeler.embedding_model,
                umap_model=modeler.umap_model,
                hdbscan_model=kmeans,  # KMeans 替代 HDBSCAN
                vectorizer_model=modeler.vectorizer_model,
                language="english",
                calculate_probabilities=False,
                verbose=True,
            )
            topics, probs = kmeans_modeler.fit_transform(abstracts, embeddings=embeddings)
            modeler.topic_model = kmeans_modeler
            n_topics_now = len(set(topics)) - (1 if -1 in topics else 0)
            print(f"[auto_fetch] KMeans 分簇完成，最终 {n_topics_now} 个主题（K={target_k}）")
        except Exception as e:
            print(f"[auto_fetch] KMeans 兜底失败（保留 HDBSCAN 结果）: {e}")

    # 路径 B：太多 → 用 BERTopic.reduce_topics 归并
    elif n_topics_now > max_topics:
        print(f"[auto_fetch] 当前 {n_topics_now} 个主题超过上限 {max_topics}，归并中...")
        try:
            modeler.reduce_topics(abstracts, n_topics=max_topics)
            topics = modeler.topic_model.topics_
            n_topics_now = len(set(topics)) - (1 if -1 in topics else 0)
            print(f"[auto_fetch] 归并完成，最终 {n_topics_now} 个主题")
        except Exception as e:
            print(f"[auto_fetch] 主题归并失败（保留原结果）: {e}")

    # 添加主题标签到 DataFrame
    df['topic_id'] = topics
    topic_name_map = {}
    topic_keywords_map: Dict[int, List[str]] = {}
    for topic_id in set(topics):
        topic_name_map[topic_id] = modeler.get_topic_name(topic_id)
        # 收集 top-8 关键词用于 LLM 命名
        try:
            top_kw = modeler.get_topic(topic_id, top_n=8) or []
            topic_keywords_map[topic_id] = [w for w, _ in top_kw]
        except Exception:
            topic_keywords_map[topic_id] = topic_name_map[topic_id].split('_')
    df['topic_name'] = df['topic_id'].map(topic_name_map)

    # === LLM 主题命名（生成可读标题，写入 topic_label 列） ===
    try:
        from .nlp.topic_namer import LLMTopicNamer
        namer = LLMTopicNamer.get_singleton()
        # 噪声主题 -1 不命名
        labels_map = {
            tid: ("Outliers" if tid == -1 else namer.name_topic(kws))
            for tid, kws in topic_keywords_map.items()
        }
        df['topic_label'] = df['topic_id'].map(labels_map)
        print(f"[LLM 命名] 已生成 {len(labels_map)} 个主题标题")
    except Exception as e:
        print(f"[LLM 命名] 跳过（{e}）；保留默认 topic_name")
        df['topic_label'] = df['topic_name']

    # 保存带主题标签的 CSV
    topics_csv_path = PROCESSED_DATA_DIR / f"papers_{safe_query}_with_topics.csv"
    df.to_csv(topics_csv_path, index=False, encoding='utf-8')

    # 同时保存标准名称（供 topic_service 使用）
    standard_topics_csv = PROCESSED_DATA_DIR / f"arxiv_{safe_query}_with_topics.csv"
    df.to_csv(standard_topics_csv, index=False, encoding='utf-8')

    # 保存模型
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "bertopic_model"
    modeler.save_model(model_path)

    # ========== 层级主题（用于多级 Sunburst）==========
    # 在主题数 ≥ 3 时计算 Ward 凝聚层级结构，保存为 Plotly 可直接使用的平铺 JSON
    hier_path: Optional[Path] = None
    try:
        import json as _json
        hier_data = _compute_hierarchy_sunburst(modeler, df, labels_map)
        hier_path = PROCESSED_DATA_DIR / f"arxiv_{safe_query}_hierarchy.json"
        with open(hier_path, "w", encoding="utf-8") as _f:
            _json.dump(hier_data, _f, ensure_ascii=False)
        # 同时写标准副本（方便服务重启后加载）
        standard_hier = PROCESSED_DATA_DIR / "hierarchy.json"
        with open(standard_hier, "w", encoding="utf-8") as _f:
            _json.dump(hier_data, _f, ensure_ascii=False)
        print(f"[层级主题] 已保存，共 {len(hier_data['labels'])} 个节点")
    except Exception as _e:
        print(f"[层级主题] 计算失败（{_e}），跳过；Sunburst 将回退到 2 层结构")

    # ========== 步骤 5: 完成 ==========
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    progress(5, total_steps, f"完成！共 {len(papers)} 篇论文，{n_topics} 个主题")

    result = {
        "query": query,
        "csv_path": str(standard_csv),
        "topics_csv_path": str(standard_topics_csv),
        "embeddings_path": str(standard_emb),
        "model_path": str(model_path),
        "hierarchy_path": str(hier_path) if hier_path else None,
        "paper_count": len(papers),
        "topic_count": n_topics,
        "source_stats": source_stats,
        "routing": routing_explain,
    }

    return result


def _compute_hierarchy_sunburst(
    modeler: Any,
    df: pd.DataFrame,
    labels_map: Dict[int, str],
) -> Dict[str, Any]:
    """
    调用 BERTopic hierarchical_topics()，将凝聚层级树转为 Plotly Sunburst 平铺格式。

    返回结构（与 /api/topics/sunburst 同格式）:
        {
            "labels":    [...],   # 节点显示名
            "parents":   [...],   # 父节点 id（根节点为 ""）
            "values":    [...],   # 论文数
            "ids":       [...],   # 节点唯一 id
            "topic_ids": [...],   # 原始 topic_id（-1=根, -2=内部节点, ≥0=叶子主题）
        }

    层级说明（以 6 个主题为例）:
        All Topics (root)
        ├─ Cluster A  (Ward 第1层合并)
        │   ├─ Topic 0: BERT fine-tuning
        │   └─ Topic 1: BERT variants
        └─ Cluster B
            ├─ Topic 2: Instruction Tuning
            └─ Cluster C
                ├─ Topic 3: RLHF
                └─ Topic 4: Alignment
    """
    # ── 统计每个有效主题的论文数 ─────────────────────────────
    topic_counts: Dict[int, int] = {
        int(k): int(v)
        for k, v in df[df["topic_id"] != -1]["topic_id"].value_counts().items()
    }

    # ── 叶子节点显示名："Topic N: LLM命名" ────────────────────
    leaf_display: Dict[int, str] = {
        tid: f"Topic {tid}: {labels_map.get(tid, f'Topic {tid}')}"
        for tid in topic_counts
    }

    # ── 主题数 < 3 时层级意义不大，直接返回 2 层平铺 ──────────
    # 注意：parents 必须引用 ids（不是 labels），根节点 id="root"
    if len(topic_counts) < 3:
        s_tids = sorted(topic_counts)
        return {
            "labels":    ["All Topics"] + [leaf_display[t] for t in s_tids],
            "parents":   [""] + ["root"] * len(s_tids),
            "values":    [sum(topic_counts.values())] + [topic_counts[t] for t in s_tids],
            "ids":       ["root"] + [f"topic_{t}" for t in s_tids],
            "topic_ids": [-1] + s_tids,
        }

    # ── 调用 BERTopic 凝聚层级聚类 ────────────────────────────
    abstracts = df["abstract"].fillna("").tolist()
    hier_df = modeler.get_hierarchical_topics(abstracts)

    # ── 构建父节点映射与内部节点名称 ──────────────────────────
    child_to_parent: Dict[int, int] = {}
    internal_names: Dict[int, str] = {}

    for _, row in hier_df.iterrows():
        pid = int(row["Parent_ID"])
        lid = int(row["Child_Left_ID"])
        rid = int(row["Child_Right_ID"])
        child_to_parent[lid] = pid
        child_to_parent[rid] = pid

        # 清理 BERTopic 自动生成的内部节点名（下划线分词，去掉多余后缀）
        pname = str(row.get("Parent_Name", "")).strip()
        if not pname or pname.lower() == "nan":
            pname = f"Cluster {pid}"
        else:
            pname = pname.replace("_", " ").replace(" and ", " · ")
            words = pname.split()
            pname = " ".join(words[:4])
            if len(words) > 4:
                pname += "…"
        internal_names[pid] = pname

    # ── 根节点：是 Parent_ID 但从不作为 Child，取最大值兜底 ────
    all_parent_ids = set(hier_df["Parent_ID"].astype(int))
    all_child_ids = (
        set(hier_df["Child_Left_ID"].astype(int))
        | set(hier_df["Child_Right_ID"].astype(int))
    )
    pure_roots = all_parent_ids - all_child_ids
    root_id = max(pure_roots) if pure_roots else int(hier_df["Parent_ID"].max())

    # ── 收集所有节点 ──────────────────────────────────────────
    all_nodes: set = set(topic_counts.keys())
    for _, row in hier_df.iterrows():
        all_nodes.update([
            int(row["Parent_ID"]),
            int(row["Child_Left_ID"]),
            int(row["Child_Right_ID"]),
        ])

    # ── 自底向上递归计算各节点论文数 ──────────────────────────
    # 不依赖 hier_df 行序（Ward 合并顺序不保证"子 ID < 父 ID"），
    # 改用 parent_to_children 反向映射 + 记忆化递归，保证正确性。
    _p2c: Dict[int, List[int]] = {}
    for child, parent in child_to_parent.items():
        _p2c.setdefault(parent, []).append(child)

    node_values: Dict[int, int] = dict(topic_counts)

    def _fill_value(nid: int) -> int:
        if nid in node_values:
            return node_values[nid]
        total = sum(_fill_value(c) for c in _p2c.get(nid, []))
        node_values[nid] = total
        return total

    for nid in all_nodes:
        _fill_value(nid)

    def _id(nid: int) -> str:
        return f"topic_{nid}" if nid in topic_counts else f"node_{nid}"

    # ── 组装 Plotly 平铺列表 ──────────────────────────────────
    labels:   List[str] = ["All Topics"]
    parents:  List[str] = [""]
    values:   List[int] = [node_values.get(root_id, sum(topic_counts.values()))]
    ids:      List[str] = ["root"]
    tids:     List[int] = [-1]

    for nid in sorted(all_nodes):
        parent_nid = child_to_parent.get(nid)
        if parent_nid is None:
            # 这是 root_id 本身：直接以 "All Topics" 代替，跳过此节点
            continue

        # 父节点 id 字符串
        pid_str = "root" if parent_nid == root_id else _id(parent_nid)

        if nid in topic_counts:
            node_label = leaf_display.get(nid, f"Topic {nid}")
            tid = nid
        else:
            node_label = internal_names.get(nid, f"Cluster {nid}")
            tid = -2  # 内部聚类节点

        labels.append(node_label)
        parents.append(pid_str)
        values.append(node_values.get(nid, 0))
        ids.append(_id(nid))
        tids.append(tid)

    return {"labels": labels, "parents": parents, "values": values, "ids": ids, "topic_ids": tids}


# ---- 活跃数据集覆盖 ----
# search_routes 在新搜索或历史切换后调用 set_active_paths() 写入此变量。
# get_latest_data_paths(query=None) 会优先返回这里的值，
# 确保 TopicService / ChatService 重新初始化时加载正确的数据集，
# 而不是按文件修改时间自动选最新的那个。
_active_paths_override: Optional[Dict[str, Any]] = None


def set_active_paths(paths: Optional[Dict[str, Any]]) -> None:
    """设置当前活跃数据集路径（由 search_routes 在搜索/切换后调用）"""
    global _active_paths_override
    _active_paths_override = paths


def get_latest_data_paths(query: Optional[str] = None) -> Optional[Dict[str, Path]]:
    """
    查找最新的数据文件路径

    参数:
        query: 可选的查询关键词，用于精确匹配

    返回:
        {"csv": Path, "topics_csv": Path, "embeddings": Path, "model": Path}
        如果没有找到数据返回 None
    """
    # 优先：若已通过搜索/历史切换明确指定了活跃数据集，直接返回
    if _active_paths_override and not query:
        # 把字符串路径统一转为 Path 对象，保持返回类型一致
        result: Dict[str, Any] = {}
        for k, v in _active_paths_override.items():
            result[k] = Path(v) if isinstance(v, str) else v
        return result

    # 查找带主题标签的 CSV
    if query:
        safe_query = query.replace(' ', '_')[:30]
        topics_csv = PROCESSED_DATA_DIR / f"arxiv_{safe_query}_with_topics.csv"
        raw_csv = RAW_DATA_DIR / f"arxiv_{safe_query}.csv"
        if topics_csv.exists():
            hier = PROCESSED_DATA_DIR / f"arxiv_{safe_query}_hierarchy.json"
            return {
                "csv": raw_csv if raw_csv.exists() else topics_csv,
                "topics_csv": topics_csv,
                "embeddings": PROCESSED_DATA_DIR / f"embeddings_{safe_query}.npy",
                "model": MODELS_DIR / "bertopic_model",
                "hierarchy": hier if hier.exists() else None,
            }

    # 回退：查找任意已有的数据
    topic_csvs = list(PROCESSED_DATA_DIR.glob("*_with_topics.csv"))
    if topic_csvs:
        latest = max(topic_csvs, key=lambda p: p.stat().st_mtime)
        # 根据 topics_csv 文件名推断对应的层级 JSON
        safe_q = latest.stem.removeprefix("arxiv_").removesuffix("_with_topics")
        hier = PROCESSED_DATA_DIR / f"arxiv_{safe_q}_hierarchy.json"
        if not hier.exists():
            hier = PROCESSED_DATA_DIR / "hierarchy.json"
        return {
            "csv": latest,
            "topics_csv": latest,
            "embeddings": PROCESSED_DATA_DIR / "embeddings.npy",
            "model": MODELS_DIR / "bertopic_model",
            "hierarchy": hier if hier.exists() else None,
        }

    # 再回退：查找原始 CSV
    raw_csvs = list(RAW_DATA_DIR.glob("*.csv"))
    if raw_csvs:
        latest = max(raw_csvs, key=lambda p: p.stat().st_mtime)
        return {
            "csv": latest,
            "topics_csv": None,
            "embeddings": PROCESSED_DATA_DIR / "embeddings.npy",
            "model": MODELS_DIR / "bertopic_model",
            "hierarchy": None,
        }

    return None


def list_all_datasets() -> List[Dict[str, Any]]:
    """
    扫描 data/processed/ 下所有历史数据集，返回摘要列表。

    只需读取文件名和行数，不加载向量，速度极快（毫秒级）。

    返回格式（按修改时间倒序）:
        [
            {
                "query":        "BERT",            # 原始搜索关键词
                "safe_query":   "BERT",            # 文件名中使用的版本
                "paper_count":  200,               # 论文数
                "topic_count":  6,                 # 主题数（不含噪声 -1）
                "modified":     1234567890.0,      # 文件修改时间戳
                "modified_str": "2026-05-10 17:16",
                "topics_csv":   "/abs/path/...",   # topics CSV 路径
                "embeddings":   "/abs/path/...",   # 对应向量路径（可能不存在）
                "has_embeddings": True,
            },
            ...
        ]
    """
    datasets = []

    # 只扫描带主题标签的 CSV（它们是完整数据集的代表）
    for csv_path in sorted(
        PROCESSED_DATA_DIR.glob("arxiv_*_with_topics.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        # 从文件名还原 safe_query：arxiv_<safe_query>_with_topics.csv
        stem = csv_path.stem  # e.g. "arxiv_BERT_with_topics"
        safe_query = stem.removeprefix("arxiv_").removesuffix("_with_topics")
        # 还原为可读关键词（下划线 → 空格）
        query_display = safe_query.replace("_", " ")

        # 快速统计（只读两列，不加载向量）
        try:
            df_mini = pd.read_csv(csv_path, usecols=["topic_id"])
            paper_count = len(df_mini)
            valid_topics = df_mini["topic_id"][df_mini["topic_id"] != -1]
            topic_count = valid_topics.nunique()
        except Exception:
            paper_count = 0
            topic_count = 0

        # 匹配对应的向量文件
        emb_path = PROCESSED_DATA_DIR / f"embeddings_{safe_query}.npy"
        has_embeddings = emb_path.exists()

        # 匹配 hierarchy JSON（多层 Sunburst 所需）
        hier_path = PROCESSED_DATA_DIR / f"arxiv_{safe_query}_hierarchy.json"

        mtime = csv_path.stat().st_mtime
        from datetime import datetime as _dt
        modified_str = _dt.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        datasets.append({
            "query": query_display,
            "safe_query": safe_query,
            "paper_count": paper_count,
            "topic_count": topic_count,
            "modified": mtime,
            "modified_str": modified_str,
            "topics_csv": str(csv_path),
            "embeddings": str(emb_path) if has_embeddings else None,
            "has_embeddings": has_embeddings,
            "hierarchy": str(hier_path) if hier_path.exists() else None,
        })

    return datasets
