# Stage 6 — PDF 下载 与 智能来源路由 设计文档

> **版本**: v1.0 | **编写日期**: 2026-04-29
> **状态**: 设计确认阶段，待实施
> **前置依赖**: Stage 2（多源抓取 + Paper 统一模型）已完成

---

## 目录

1. [功能一：论文 PDF 三层降级下载](#1-功能一论文-pdf-三层降级下载)
2. [功能二：零样本分类智能来源路由](#2-功能二零样本分类智能来源路由)
3. [架构集成方案](#3-架构集成方案)
4. [新增文件清单](#4-新增文件清单)
5. [依赖变更](#5-依赖变更)
6. [实施步骤清单](#6-实施步骤清单)

---

## 1. 功能一：论文 PDF 三层降级下载

### 1.1 背景与现有基础

`Paper` 模型（`src/core/sources/base.py`）已有 `pdf_url` 字段，各数据源覆盖情况：

| 数据源 | pdf_url 覆盖率 | 说明 |
|--------|--------------|------|
| arXiv | ~100% | 所有论文均有公开 PDF，URL 格式固定 |
| OpenAlex | ~30-40% | 仅开放获取论文有 `open_access.oa_url` |

### 1.2 三层降级策略（最终方案）

```
Layer 1: 直接下载（优先，零成本）
  条件: paper.pdf_url 不为空，或 paper.arxiv_id 不为空
  操作:
    - 若有 pdf_url  → 直接 requests.get(pdf_url) 下载
    - 若有 arxiv_id → 构造 https://arxiv.org/pdf/{arxiv_id}.pdf 下载
  预期覆盖: arXiv 论文 100%

Layer 2a: Unpaywall API（Layer 1 失败时，优先尝试）
  条件: paper.doi 不为空
  请求: GET https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}
  解析: response["best_oa_location"]["url_for_pdf"]
  限制: 需配置邮箱（免费），无严格速率限制（合理使用）
  预期覆盖: 开放获取论文约 50%

Layer 2b: Semantic Scholar API（Layer 2a 失败时）
  条件: paper.doi 或 paper.arxiv_id 不为空
  请求: GET https://api.semanticscholar.org/graph/v1/paper/{id}?fields=openAccessPdf
  解析: response["openAccessPdf"]["url"]
  限制: 100 请求/5分钟（免费），需加请求间隔 ~3s
  预期覆盖: 约 30%

Layer 3: 失败处理
  返回: {"status": "unavailable", "page_url": paper.url}
  提示: 用户手动访问论文页面下载
```

**整体预期覆盖率**：
- arXiv 来源论文：~100%
- OpenAlex 开放获取论文：~60-70%（Unpaywall + S2 联合）
- OpenAlex 付费期刊：~0%（无合法免费渠道）
- **综合覆盖率估算：60-75%**

### 1.3 下载结果数据结构

```python
@dataclass
class DownloadResult:
    paper_id: str       # paper.get_unique_id()
    status: str         # "success" | "unavailable" | "error"
    pdf_url: str        # 实际使用的 PDF URL（成功时）
    local_path: str     # 本地保存路径（成功时）
    source_used: str    # "direct" | "unpaywall" | "semantic_scholar" | "none"
    file_size_kb: int   # 文件大小（KB，成功时）
    error_msg: str      # 失败原因（status != success 时）
```

### 1.4 本地存储规则

```
data/pdfs/
├── arxiv_1706.03762.pdf           # arXiv 论文
├── doi_10.1038_s41586-021.pdf     # DOI 论文（特殊字符替换为下划线）
└── title_a3f2b1c9d8e7.pdf         # 无 DOI/arXiv 时用标题哈希
```

文件命名规则：`{paper.get_unique_id().replace(":", "_").replace("/", "_")}.pdf`

### 1.5 新增 API 端点

```
POST /api/papers/{paper_id}/download
  描述: 触发下载，返回下载结果
  响应: DownloadResult JSON

GET  /api/papers/{paper_id}/pdf
  描述: 直接流式返回 PDF 文件（StreamingResponse）
  响应: application/pdf 文件流，或 404

GET  /api/papers/{paper_id}/pdf-status
  描述: 查询本地是否已有缓存 PDF
  响应: {"cached": true/false, "local_path": "...", "file_size_kb": 123}
```

### 1.6 核心类设计

```python
# src/core/downloader/pdf_downloader.py

class PaperDownloader:
    def __init__(self, pdf_dir: Path, unpaywall_email: str = ""):
        self.pdf_dir = pdf_dir
        self.unpaywall_email = unpaywall_email
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "AIAssistant/1.0 (research tool)"

    def download(self, paper: Paper) -> DownloadResult:
        # 检查本地缓存
        cached = self._check_cache(paper)
        if cached:
            return cached

        # Layer 1: 直接下载
        result = self._try_direct(paper)
        if result.status == "success":
            return result

        # Layer 2a: Unpaywall
        if paper.doi and self.unpaywall_email:
            result = self._try_unpaywall(paper)
            if result.status == "success":
                return result

        # Layer 2b: Semantic Scholar
        result = self._try_semantic_scholar(paper)
        if result.status == "success":
            return result

        # Layer 3: 失败
        return DownloadResult(
            paper_id=paper.get_unique_id(),
            status="unavailable",
            pdf_url="",
            local_path="",
            source_used="none",
            file_size_kb=0,
            error_msg="No open-access PDF found via any channel"
        )
```

---

## 2. 功能二：零样本分类智能来源路由

### 2.1 背景与目标

**问题**：不同学科领域在各数据源的论文覆盖率差异显著：
- arXiv 擅长：物理、计算机科学、数学、统计
- OpenAlex 擅长：医学、生物、社会科学、经济学、工程

**目标**：用户输入主题后，系统自动判断该主题属于哪个学科，并按最优比例分配各数据源的抓取数量。

### 2.2 最终方案：零样本文本分类

**选型**：`cross-encoder/nli-MiniLM-L2-mnli`（HuggingFace）

| 指标 | 值 |
|------|-----|
| 模型大小 | ~90 MB |
| 推理速度 | ~50ms/次（CPU） |
| 新增依赖 | 无（复用现有 `transformers`） |
| 离线可用 | 是（下载后） |
| 准确率 | 约 85%（学科分类任务） |

### 2.3 学科→来源权重映射表

```python
# 基于各数据源实际覆盖率设计的权重
DISCIPLINE_WEIGHTS = {
    "computer science":  {"arxiv": 0.65, "openalex": 0.35},
    "physics":           {"arxiv": 0.75, "openalex": 0.25},
    "mathematics":       {"arxiv": 0.70, "openalex": 0.30},
    "statistics":        {"arxiv": 0.65, "openalex": 0.35},
    "biology":           {"arxiv": 0.20, "openalex": 0.80},
    "medicine":          {"arxiv": 0.10, "openalex": 0.90},
    "chemistry":         {"arxiv": 0.30, "openalex": 0.70},
    "economics":         {"arxiv": 0.35, "openalex": 0.65},
    "social science":    {"arxiv": 0.15, "openalex": 0.85},
    "engineering":       {"arxiv": 0.40, "openalex": 0.60},
    "default":           {"arxiv": 0.50, "openalex": 0.50},
}
```

### 2.4 分类与权重融合流程

```
用户输入: "CRISPR gene editing in cancer therapy"
    |
    v
TopicClassifier.classify(query)
    |
    v
zero-shot 分类（候选标签 = DISCIPLINE_WEIGHTS 的 key）
    |
    v
结果: {"biology": 0.61, "medicine": 0.28, "chemistry": 0.08, ...}
    |
    v
加权融合（取 top-2 学科，按置信度加权平均）:
  biology(0.61)  x {arxiv:0.20, openalex:0.80}
+ medicine(0.28) x {arxiv:0.10, openalex:0.90}
= {arxiv: 0.61x0.20 + 0.28x0.10, openalex: 0.61x0.80 + 0.28x0.90}
= {arxiv: 0.15, openalex: 0.74}  -> 归一化 -> {arxiv: 0.17, openalex: 0.83}
    |
    v
MultiSourceAggregator.search_all(query, weights={"arxiv": 0.17, "openalex": 0.83})
    |
    v
arXiv 抓取:    200 x 0.17 = 34 篇
OpenAlex 抓取: 200 x 0.83 = 166 篇
```

### 2.5 核心类设计

```python
# src/core/nlp/topic_classifier.py

class TopicClassifier:
    MODEL_NAME = "cross-encoder/nli-MiniLM-L2-mnli"

    def __init__(self):
        self._pipeline = None  # 懒加载，首次调用时初始化

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.MODEL_NAME,
                device=-1  # CPU
            )

    def classify(self, query: str, top_k: int = 2) -> Dict[str, float]:
        """
        对查询文本进行学科分类

        参数:
            query: 用户输入的主题/关键词
            top_k: 取置信度最高的 k 个学科参与权重融合

        返回:
            归一化的学科置信度字典
            示例: {"biology": 0.68, "medicine": 0.32}
        """
        self._load()
        labels = [k for k in DISCIPLINE_WEIGHTS if k != "default"]
        result = self._pipeline(query, candidate_labels=labels)
        top_labels = result["labels"][:top_k]
        top_scores = result["scores"][:top_k]
        total = sum(top_scores)
        return {label: score/total for label, score in zip(top_labels, top_scores)}

    def get_source_weights(self, query: str) -> Dict[str, float]:
        """
        直接返回各数据源的抓取权重

        返回:
            数据源权重字典，值之和为 1.0
            示例: {"arxiv": 0.17, "openalex": 0.83}
        """
        discipline_scores = self.classify(query)
        merged = {}
        for discipline, score in discipline_scores.items():
            weights = DISCIPLINE_WEIGHTS.get(discipline, DISCIPLINE_WEIGHTS["default"])
            for source, w in weights.items():
                merged[source] = merged.get(source, 0.0) + score * w
        total = sum(merged.values())
        return {k: v/total for k, v in merged.items()}
```

### 2.6 与 MultiSourceAggregator 的集成

修改 `src/core/sources/aggregator.py` 的 `search_all()` 方法，新增 `weights` 参数：

```python
def search_all(
    self,
    query: str,
    max_results: int = 200,
    weights: Optional[Dict[str, float]] = None,  # 新增参数
    oversample_ratio: float = 1.5,
    min_tolerance: float = 0.9
) -> List[Paper]:
    """
    weights 示例: {"arxiv": 0.3, "openalex": 0.7}
    若 weights 为 None，则每个来源平均分配（保持原有行为，向后兼容）
    """
    if weights is None:
        per_source = int(max_results * oversample_ratio)
        source_limits = {s.get_source_name(): per_source for s in self.sources}
    else:
        total_with_oversample = int(max_results * oversample_ratio)
        source_limits = {
            s.get_source_name(): max(
                10,  # 每个来源至少抓取 10 篇，避免某来源完全被忽略
                int(total_with_oversample * weights.get(s.get_source_name(), 1/len(self.sources)))
            )
            for s in self.sources
        }
    # 后续逻辑不变，只是每个 source 的请求数量不同
```

### 2.7 前端展示

在 Streamlit 搜索结果页面新增"来源分配"信息卡：

```
+------------------------------------------+
| 搜索: "CRISPR gene editing"              |
|                                          |
| 智能来源分配（基于主题分析）              |
|   检测到学科: 生物学(68%) + 医学(32%)    |
|   arXiv:    ████░░░░░░  17%  (34篇)      |
|   OpenAlex: ████████░░  83%  (166篇)     |
+------------------------------------------+
```

---

## 3. 架构集成方案

### 3.1 整体数据流（新增部分）

```
用户输入 query
    |
    v
TopicClassifier.get_source_weights(query)
    |  返回 weights = {"arxiv": 0.17, "openalex": 0.83}
    v
MultiSourceAggregator.search_all(query, weights=weights)
    |  按权重分配各源抓取数量
    v
List[Paper]（含 pdf_url 字段）
    |
    +---> 正常流程（向量化 -> 主题建模 -> 展示）
    |
    +---> PaperDownloader.download(paper)（用户点击下载时触发）
              |
              +-- Layer 1: 直接下载 pdf_url / arxiv pdf
              +-- Layer 2a: Unpaywall API
              +-- Layer 2b: Semantic Scholar API
              +-- Layer 3: 返回页面链接
```

### 3.2 与现有规划的关系

| 功能 | 建议加入阶段 | 与现有规划冲突 |
|------|------------|--------------|
| PDF 三层下载 | Stage 4（与收藏功能配合，"收藏并下载"） | 无冲突，互补 |
| 零样本来源路由 | Stage 3（改动小，可提前） | 无冲突，向后兼容 |

---

## 4. 新增文件清单

### 功能一：PDF 下载

| 操作 | 文件路径 | 说明 |
|------|---------|------|
| 新建 | `src/core/downloader/__init__.py` | 模块入口 |
| 新建 | `src/core/downloader/pdf_downloader.py` | `PaperDownloader` 主类（三层降级，约150行） |
| 新建 | `src/core/downloader/unpaywall.py` | Unpaywall API 适配器（约80行） |
| 新建 | `src/core/downloader/semantic_scholar_pdf.py` | Semantic Scholar PDF 查询（约60行） |
| 修改 | `src/core/config.py` | 新增 `PDF_DIR = DATA_DIR / "pdfs"` 和 `UNPAYWALL_EMAIL` |
| 修改 | `src/core/api/routes.py` | 新增 `/api/papers/{id}/download`、`/pdf`、`/pdf-status` 端点 |
| 修改 | `env.example` | 新增 `UNPAYWALL_EMAIL=your@email.com` |
| 修改 | `app.py` | 论文详情页新增「下载 PDF」按钮 |

### 功能二：智能来源路由

| 操作 | 文件路径 | 说明 |
|------|---------|------|
| 新建 | `src/core/nlp/topic_classifier.py` | `TopicClassifier` 类（零样本分类，约100行） |
| 修改 | `src/core/sources/aggregator.py` | `search_all()` 新增 `weights` 参数（约20行改动） |
| 修改 | `src/core/api/routes.py` | 搜索端点调用 `TopicClassifier` 并传入 weights |
| 修改 | `app.py` | 搜索结果页展示来源分配信息卡 |

---

## 5. 依赖变更

### 新增 Python 包

| 包名 | 用途 | 是否已有 |
|------|------|---------|
| `requests` | PDF 下载、Unpaywall/S2 API 调用 | 已有（间接依赖） |
| `transformers` | 零样本分类模型 | 已有 |

**结论：无需新增任何 Python 包**，完全复用现有依赖。

### 新增模型文件（首次运行自动下载）

| 模型 | 大小 | 下载来源 |
|------|------|---------|
| `cross-encoder/nli-MiniLM-L2-mnli` | ~90 MB | HuggingFace（支持镜像） |

### 新增数据目录

```
data/pdfs/    <- PDF 本地缓存目录（需在 config.py 配置）
```

---

## 6. 实施步骤清单

### 功能二（智能来源路由，建议先做，改动小）

- [ ] **Step 6.1** — 新建 `src/core/nlp/topic_classifier.py`，实现 `TopicClassifier` 类
- [ ] **Step 6.2** — 修改 `src/core/sources/aggregator.py`，`search_all()` 新增 `weights` 参数
- [ ] **Step 6.3** — 修改 `src/core/api/routes.py`，搜索端点集成 `TopicClassifier`
- [ ] **Step 6.4** — 修改 `app.py`，搜索结果展示来源分配信息卡
- [ ] **Step 6.5** — 新建 `src/test_stage6_routing.py`，测试分类准确性

### 功能一（PDF 下载，建议 Stage 4 时做）

- [ ] **Step 6.6** — 新建 `src/core/downloader/__init__.py`
- [ ] **Step 6.7** — 新建 `src/core/downloader/unpaywall.py`，实现 Unpaywall API 适配器
- [ ] **Step 6.8** — 新建 `src/core/downloader/semantic_scholar_pdf.py`，实现 S2 PDF 查询
- [ ] **Step 6.9** — 新建 `src/core/downloader/pdf_downloader.py`，实现三层降级主类
- [ ] **Step 6.10** — 修改 `src/core/config.py`，新增 `PDF_DIR` 和 `UNPAYWALL_EMAIL`
- [ ] **Step 6.11** — 修改 `env.example`，新增 `UNPAYWALL_EMAIL=` 配置项
- [ ] **Step 6.12** — 修改 `src/core/api/routes.py`，新增三个下载相关端点
- [ ] **Step 6.13** — 修改 `app.py`，论文详情页新增「下载 PDF」按钮
- [ ] **Step 6.14** — 新建 `src/test_stage6_download.py`，测试三层降级逻辑

**预估工作量**：
- 功能二（智能路由）：约 4-6 小时
- 功能一（PDF 下载）：约 8-10 小时
- 合计：约 2 个工作日

---

*文档编写时间：2026-04-29*
*基于 Stage 2 已完成的多源抓取架构设计*
