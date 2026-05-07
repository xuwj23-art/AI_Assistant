# 科研文献综述 AI 助手 — 后续阶段完整规划文档

> **版本**: v2.0 | **编写日期**: 2026-04-16 | **最后更新**: 2026-04-29
> **当前状态**: Stage 1 ✅ | Stage 2 ✅ | Stage 3 ✅ 完成
> **部署策略**: 本地部署（不使用服务器/Docker 部署）
> **本文档范围**: Stage 3、Stage 4、Stage 5 的完整规划说明

---

## 目录

1. [整体架构回顾（已完成部分）](#1-整体架构回顾已完成部分)
2. [Stage 3 — 可视化与体验升级](#2-stage-3--可视化与体验升级)
3. [Stage 4 — 个性化功能与深度分析](#3-stage-4--个性化功能与深度分析)
4. [Stage 5 — 智能化与生产部署](#4-stage-5--智能化与生产部署)
5. [各阶段依赖关系图](#5-各阶段依赖关系图)
6. [技术决策说明](#6-技术决策说明)
7. [优先级与时间估算](#7-优先级与时间估算)

---

## 1. 整体架构回顾（已完成部分）

### 已完成功能汇总

```
Stage 1 (Bug修复) ✅
├── requirements.txt 编码修复
├── train_topics.py 导入路径修复
├── build_vector_store.py 缺失 import 修复
└── 完整链路可运行：抓取 → 向量化 → 聚类 → 后端 → 前端

Stage 2 (核心功能升级) ✅
├── 统一 Paper 数据模型 (src/core/sources/base.py)
├── arXiv 适配器 (src/core/sources/arxiv_adapter.py)
├── OpenAlex 数据源 (src/core/openalex/client.py)
├── 多源聚合器 + 三级去重 (src/core/sources/aggregator.py)
├── ChromaDB 持久化存储 (src/core/storage/chroma_store.py)
├── 层级主题方法 (src/core/nlp/topic_modeling.py 新增2个方法)
└── 多源抓取 CLI (src/scripts/fetch_papers.py)

Stage 3 (可视化与体验升级) ✅
├── Plotly Sunburst 旭日图主题地图 (src/core/api/topic_routes.py)
├── 主题点击钻取 + 论文列表联动 (app.py)
├── 4 阶段实时进度条 (app.py)
├── 研究趋势折线图 (src/core/nlp/trend_analysis.py)
├── 相关主题推荐 (/api/topics/{id}/similar)
└── 集成测试 (src/test_stage3_viz.py)
```

### 当前系统能做什么

| 能力 | 状态 |
|------|------|
| 从 arXiv 抓取论文 | ✅ 完成 |
| 从 OpenAlex 抓取论文 | ✅ 完成（Stage 2 新增） |
| 多源并行抓取 + 三级去重 | ✅ 完成（Stage 2 新增） |
| 论文向量化（SentenceTransformer） | ✅ 完成 |
| ChromaDB 持久化缓存 | ✅ 完成（Stage 2 新增） |
| BERTopic 主题建模（一层） | ✅ 完成 |
| BERTopic 层级主题（多层树） | ✅ 完成（Stage 2 新增） |
| FastAPI 后端 API | ✅ 完成 |
| Streamlit 对话前端 | ✅ 完成（但界面较简单） |
| **交互式主题地图** | ✅ Stage 3 已完成 |
| **研究趋势时间轴** | ✅ Stage 3 已完成 |
| **论文收藏 / 笔记** | ❌ Stage 4 计划 |
| **BibTeX 导出** | ❌ Stage 4 计划 |
| **引用关系网络** | ❌ Stage 4 计划 |
| **LLM 智能主题命名** | ❌ Stage 5 计划 |

---

## 2. Stage 3 — 可视化与体验升级

### 为什么要做 Stage 3？

Stage 2 完成后，系统已经能"正确工作"了。但用户体验还很原始：

- **看不到**主题结构：主题只是文字列表，用户无法感知哪些方向规模大、哪些是子方向
- **等待时体验差**：搜索时界面卡住，用户不知道发生了什么
- **无法探索**：没有"点击某主题看更多"的交互，每次只能看全部结果

Stage 3 要把系统从"可用"变成"好用"。

### Stage 3 目标

> 让用户能**看到**论文分布，能**点击探索**主题，能**实时看到**抓取进度。

---

### 2.1 功能一：交互式主题地图（Plotly Sunburst）

#### 是什么？

Sunburst（旭日图）是一种圆形分层图表。中心圆代表"全部主题"，外圈的每个扇区代表一个研究主题，扇区的大小反映该主题的论文数量，用户可以点击某个扇区"钻入"查看子主题。

```
示意图（文字版）：

         ┌──────────────────────┐
         │   All Topics (200)   │
         │  ┌───┐  ┌──────────┐ │
         │  │NLP│  │CV (30篇) │ │
         │  │80 │  └──────────┘ │
         │  │篇 │  ┌──────────┐ │
         │  └───┘  │RL (45篇) │ │
         │         └──────────┘ │
         └──────────────────────┘
         
点击 "NLP 80篇" → 展开 NLP 子主题:
   ┌ Transformer (40篇)
   ├ BERT/GPT (25篇)
   └ 其他NLP (15篇)
```

#### 数据来源

[`TopicModeler.export_hierarchy_json()`](src/core/nlp/topic_modeling.py:277) 已经实现，输出树形 JSON。需要将其转换为 Plotly 所需的平铺格式（labels/parents/values 三个数组）。

#### 需要修改/新建的文件

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/core/api/topic_routes.py` | **修改** | 新增 `/api/topics/sunburst` 端点，将树形 JSON 转为平铺格式 |
| `app.py` | **修改** | 新增"主题地图"标签页，用 `plotly.graph_objects.Sunburst` 渲染 |
| `requirements.txt` | **修改** | 确认包含 `plotly>=5.18.0` |

#### 新增 API 端点设计

```
GET /api/topics/sunburst?query=deep+learning

返回格式（Plotly 直接使用）:
{
  "labels":  ["All Topics", "NLP",    "Transformer", "CV",    ...],
  "parents": ["",           "All Topics", "NLP",     "All Topics", ...],
  "values":  [0,            80,          40,          30,      ...],
  "ids":     ["root",       "topic_2",   "topic_7",  "topic_1", ...]
}
```

#### 前端交互设计

```
用户操作流程：
1. 进入"主题地图"页 → 输入关键词
2. 系统生成 Sunburst 图（每个扇区 = 一个研究主题）
3. 用户点击某扇区 → 图表自动聚焦该主题（Plotly 原生支持）
4. 页面右侧同步显示该主题下的论文列表
5. 用户点击某篇论文 → 展开详情（摘要 + AI总结 + 相关推荐）
```

---

### 2.2 功能二：主题点击钻取

#### 是什么？

"钻取"（Drill-down）是数据可视化中的标准术语，指从宏观视图"进入"到更细节的视图。

这是导师需求中"探索式研究导航"的核心。研究者不是被动接受一个固定结果列表，而是主动"漫游"研究领域地图。

#### 交互流程

```
搜索 "machine learning"
    ↓
Level 0: Sunburst 图显示 8 个大主题
    ↓
点击 "NLP/Language Models (80篇)"
    ↓
Level 1: 右侧显示该主题论文列表 + 页面内展示 3 个子主题
    ↓
点击 "Transformer Architecture (40篇)"  
    ↓
Level 2: 展示更细粒度论文 + 可选是否继续细分
    ↓
点击某篇具体论文
    ↓
论文详情面板：完整摘要 + AI总结 + "你可能还感兴趣" 推荐
```

#### 需要修改/新建的文件

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/core/api/topic_routes.py` | **修改** | 新增 `/api/topics/{id}/papers` 带分页/排序支持 |
| `src/core/api/topic_routes.py` | **修改** | 新增 `/api/topics/{id}/drill` 子主题重聚类 |
| `app.py` | **修改** | 实现点击→钻取的前端交互逻辑（用 `st.session_state` 跟踪层级） |

#### 新增 API 端点设计

```
GET /api/topics/{topic_id}/papers?sort_by=relevance&page=1&page_size=20

参数说明：
  sort_by: relevance（按相关性） | date（按时间） | citations（按引用数）
  page: 页码（从1开始）
  page_size: 每页论文数（5-100）

返回：
{
  "topic_name": "transformer_attention_model",
  "total": 40,
  "page": 1,
  "papers": [
    {
      "id": "arxiv_1706.03762",
      "title": "Attention Is All You Need",
      "authors": ["Vaswani, Ashish", "Shazeer, Noam", "..."],
      "year": 2017,
      "abstract": "...",
      "url": "https://arxiv.org/abs/1706.03762",
      "relevance_score": 0.95
    },
    ...
  ]
}
```

---

### 2.3 功能三：实时进度显示

#### 问题背景

当前搜索流程的用户体验：

```
用户点击"搜索" → 界面完全卡住 30-60 秒 → 突然显示结果
```

用户完全不知道：系统在抓取？在训练？出错了？

#### 解决方案

把搜索流程拆为 4 个阶段，每个阶段实时更新 Streamlit 进度条：

```
阶段 1/4: 抓取论文       [████░░░░░░] 40%
  "正在从 arXiv 和 OpenAlex 抓取，已获取 150/200 篇..."

阶段 2/4: 检查缓存       [█████░░░░░] 50%
  "缓存命中: 120 篇（跳过）| 需向量化: 30 篇（新论文）"

阶段 3/4: 生成向量       [████████░░] 80%
  "为 30 篇新论文生成向量，预计还需 15 秒..."

阶段 4/4: 主题建模       [██████████] 100%
  "完成！发现 8 个研究主题"
```

#### 关键优化点

**利用 ChromaDB 缓存跳过已处理论文**（Stage 2 的核心价值）：

```
第一次搜索 "deep learning"（200篇论文）：
  - 阶段3需要向量化: 200篇（全部新的）→ 约 60 秒

第二次搜索同样词（200篇论文）：
  - 阶段3需要向量化: 0篇（全部命中缓存）→ 约 2 秒！
```

#### 需要修改的文件

| 文件 | 操作 | 内容 |
|------|------|------|
| `app.py` | **修改** | 改造主搜索函数，用 `st.progress()` + `st.empty()` 实现进度显示 |

---

### 2.4 功能四：研究趋势时间轴

#### 是什么？

展示某研究领域中，各主题在不同年份的论文数量变化趋势（折线图）。

```
示例图（文字版）：

论文数
  120│                               ╭─── Transformer
   80│                        ╭─────╯
   60│              ╭────────╯           ╭── BERT/GPT
   40│       ╭─────╯     ╭─────────────╯
   20│  ╭───╯     ╭─────╯
    0└──────────────────────────────────── 年份
      2018  2019  2020  2021  2022  2023
```

通过折线图，研究者能立刻判断：哪个方向正在爆发？哪个方向已经成熟？

#### 数据基础

[`Paper.published_date`](src/core/sources/base.py:13) 字段已在 Stage 2 实现，arXiv 和 OpenAlex 都会填充这个字段，数据基础已具备。

#### 需要新建/修改的文件

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/core/nlp/trend_analysis.py` | **新建** | `compute_topic_trends()` 函数：按年份统计各主题论文数 |
| `src/core/nlp/trend_analysis.py` | **新建** | `get_trending_topics()` 函数：找出近两年增长最快的主题 |
| `src/core/api/topic_routes.py` | **修改** | 新增 `/api/topics/trends` 端点 |
| `app.py` | **修改** | 新增"研究趋势"标签页，用 `plotly.express.line` 渲染折线图 |

#### `trend_analysis.py` 核心逻辑说明

```
compute_topic_trends() 算法：

输入：
  - papers 列表（每篇论文有 published_date 和 topic_id）
  - topic_names 字典（topic_id → 主题名称）

处理：
  for 每篇论文:
    year = 论文发表年份
    topic = 该论文所属主题名称
    count[year][topic] += 1

输出：
  DataFrame，行=年份，列=主题，值=论文数量
  
  示例：
          NLP    CV    RL
  2020     28    15    10
  2021     45    20    18
  2022     78    22    25

get_trending_topics() 算法：

  增长率 = (最近2年平均论文数 + 1) / (前3年平均论文数 + 1)
  （+1 是为了避免除以零）
  返回增长率最高的 Top N 主题
```

---

### 2.5 Stage 3 实施步骤清单（✅ 已全部完成）

- [x] **Step 3.1** — 新建 `src/core/nlp/trend_analysis.py`（趋势分析工具函数）
- [x] **Step 3.2** — 修改 `src/core/api/topic_routes.py`，新增 `/api/topics/sunburst` 端点
- [x] **Step 3.3** — 修改 `src/core/api/topic_routes.py`，新增 `/api/topics/trends` 端点
- [x] **Step 3.4** — 修改 `src/core/api/topic_routes.py`，完善 `/api/topics/{id}/papers` 支持分页排序
- [x] **Step 3.5** — 修改 `app.py`，新增"主题地图"标签页（Sunburst 图）
- [x] **Step 3.6** — 修改 `app.py`，实现主题钻取交互（点击扇区→右侧论文列表联动）
- [x] **Step 3.7** — 修改 `app.py`，新增"研究趋势"标签页（折线图）
- [x] **Step 3.8** — 修改 `app.py`，改造搜索流程，加入 4 阶段进度条
- [x] **Step 3.9** — 修改 `requirements.txt`，确认 `plotly>=5.18.0` 已包含
- [x] **Step 3.10** — 新建 `src/test_stage3_viz.py`，集成测试可视化 API

**实际完成时间**: 2026-04-29（详见 [`docs/STAGE3_UPGRADE_SUMMARY.md`](docs/STAGE3_UPGRADE_SUMMARY.md)）

---

## 3. Stage 4 — 个性化功能与深度分析

### 为什么要做 Stage 4？

Stage 3 完成后，系统已经很"好用"了。Stage 4 要让系统变得"有记忆"和"有深度"：

- **有记忆**：研究者读过的论文、写的笔记不会消失，下次打开还在
- **可导出**：研究成果能融入现有的学术工作流（BibTeX、Markdown笔记）
- **有深度**：看到论文之间的引用关系，发现领域"奠基论文"

### Stage 4 目标

> 让研究者能"收藏"感兴趣的内容，能"导出"到自己的工作流，能看到"引用关系"。

---

### 3.1 功能一：论文收藏与个人笔记

#### 是什么？

类似浏览器的"书签"功能，但专门针对学术论文设计，额外支持：
- 给每篇论文写笔记（"这篇的贡献是..."、"和我的研究的关联是..."）
- 给笔记打标签（#重要、#方法创新、#必读）
- 按标签筛选收藏的论文

#### 技术方案：SQLite 本地数据库

选择 SQLite 的原因：
- Python **内置**，无需安装任何额外包
- **轻量**：对个人使用完全够用（10万篇论文也只有几十MB）
- **文件型**：整个数据库就是一个 `data/user_library.db` 文件，方便备份和迁移

#### 数据库表结构

```
表1: saved_papers（收藏的论文）
┌─────────────────┬──────────────────────────────────────────┐
│ 字段名          │ 说明                                     │
├─────────────────┼──────────────────────────────────────────┤
│ id              │ 主键，来自 paper.get_unique_id()          │
│ title           │ 论文标题                                 │
│ authors         │ 作者列表（JSON格式）                     │
│ abstract        │ 摘要                                     │
│ source          │ 来源（arxiv / openalex）                 │
│ doi             │ DOI 编号                                 │
│ arxiv_id        │ arXiv ID                                 │
│ published_date  │ 发表日期                                 │
│ url             │ 论文链接                                 │
│ saved_at        │ 收藏时间                                 │
└─────────────────┴──────────────────────────────────────────┘

表2: notes（笔记）
┌─────────────────┬──────────────────────────────────────────┐
│ 字段名          │ 说明                                     │
├─────────────────┼──────────────────────────────────────────┤
│ id              │ 自增主键                                 │
│ paper_id        │ 关联哪篇论文（外键→ saved_papers.id）    │
│ content         │ 笔记正文                                 │
│ tags            │ 标签列表（JSON格式，如["重要","方法"]）  │
│ created_at      │ 创建时间                                 │
│ updated_at      │ 最后修改时间                             │
└─────────────────┴──────────────────────────────────────────┘
```

#### 需要新建/修改的文件

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/core/storage/user_library.py` | **新建** | `UserLibrary` 类，封装所有 SQLite 操作 |
| `src/core/config.py` | **修改** | 新增 `USER_LIBRARY_DB = DATA_DIR / "user_library.db"` |
| `src/core/api/library_routes.py` | **新建** | FastAPI 路由：收藏/笔记 CRUD 操作 |
| `app.py` | **修改** | 新增"我的收藏"页面 |

#### `UserLibrary` 类的主要方法

```
UserLibrary 类方法列表：

save_paper(paper: Paper) → bool
  作用：收藏一篇论文
  返回：True=新增成功，False=已收藏过

remove_paper(paper_id: str) → bool
  作用：取消收藏
  返回：True=删除成功，False=不存在

get_saved_papers(tag_filter=None) → List[dict]
  作用：获取所有收藏（可按标签过滤）

add_note(paper_id, content, tags=[]) → int
  作用：给论文添加笔记
  返回：新笔记的 ID

update_note(note_id, content, tags) → bool
  作用：修改笔记

delete_note(note_id) → bool
  作用：删除笔记

get_notes(paper_id) → List[dict]
  作用：获取某篇论文的所有笔记

is_saved(paper_id) → bool
  作用：检查论文是否已收藏（用于显示"收藏"按钮状态）

export_bibtex(paper_ids=None) → str
  作用：导出为 BibTeX 格式（None=导出全部）

export_markdown(paper_ids=None) → str
  作用：导出为 Markdown 综述笔记（None=导出全部）
```

---

### 3.2 功能二：BibTeX / Markdown 导出

#### 为什么重要？

研究者的最终产出是论文。他们需要把在本系统中找到的论文引用到自己的论文里。BibTeX 是学术界标准的引用格式，支持 LaTeX、Word（通过 Mendeley/Zotero）等所有主流写作工具。

Markdown 导出则方便研究者生成结构化的**阅读笔记文档**，可以直接同步到 Notion、Obsidian 等知识管理工具。

#### BibTeX 导出格式示例

```bibtex
@article{vaswani2017,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and ...},
  year      = {2017},
  journal   = {arXiv preprint arXiv:1706.03762},
  doi       = {10.48550/arXiv.1706.03762},
  url       = {https://arxiv.org/abs/1706.03762}
}

@article{devlin2019,
  title     = {BERT: Pre-training of Deep Bidirectional Transformers...},
  author    = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and ...},
  year      = {2019},
  ...
}
```

**BibTeX key 生成规则**：第一作者姓氏（小写） + 发表年份。若冲突则加字母后缀（如 `vaswani2017a`、`vaswani2017b`）。

#### Markdown 导出格式示例

```markdown
# 文献综述笔记

> 导出时间：2026-04-16 18:30
> 论文总数：12 篇

---

## 1. Attention Is All You Need

**作者**: Vaswani, Ashish; Shazeer, Noam et al.
**年份**: 2017 | **来源**: arXiv | **DOI**: 10.48550/arXiv.1706.03762
**链接**: https://arxiv.org/abs/1706.03762

### 摘要
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks...（前500字）

### 我的笔记
- 提出了完全基于注意力机制的 Transformer 架构，放弃了 RNN/CNN
- 并行计算能力显著优于 RNN，训练速度快很多
- 标签: #基础架构 #必读 #transformer

---

## 2. BERT: Pre-training of Deep Bidirectional Transformers...

...（后续论文同样格式）
```

#### 前端下载按钮设计

```
"我的收藏"页面布局：

┌────────────────────────────────────────┐
│ 我的论文收藏                           │
│                                        │
│ 共收藏 12 篇 | 标签筛选: [全部 ▼]     │
│                                        │
│  [导出 BibTeX]  [导出 Markdown]        │
│                                        │
│ ┌──────────────────────────────────┐   │
│ │ ☆ Attention Is All You Need     │   │
│ │   Vaswani et al. | 2017 | arXiv  │   │
│ │   [查看笔记]  [编辑笔记]  [删除] │   │
│ └──────────────────────────────────┘   │
│ ...（论文列表）                        │
└────────────────────────────────────────┘
```

---

### 3.3 功能三：相关主题自动推荐

#### 是什么？

当用户正在浏览某个主题时，系统自动推荐"与该主题最相关的其他主题"。

例如：用户在看"Transformer Architecture"主题时，系统推荐：
1. "BERT/Pre-training"（相关性 0.91）
2. "Attention Mechanism"（相关性 0.88）
3. "Large Language Models"（相关性 0.85）

#### 技术原理

每个主题可以用该主题下所有论文向量的**平均值**来表示（称为"主题中心向量"）。主题之间的相关性 = 主题中心向量之间的余弦相似度。

已实现的 [`ChromaStore`](src/core/storage/chroma_store.py:15) 的 `get_all_embeddings()` 方法可以获取所有向量，计算主题中心向量只需要几行代码。

#### 需要新建/修改的文件

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/core/nlp/topic_similarity.py` | **新建** | `compute_topic_centroids()` + `find_similar_topics()` 函数 |
| `src/core/api/topic_routes.py` | **修改** | 新增 `/api/topics/{id}/similar` 端点 |
| `app.py` | **修改** | 在主题详情页右侧显示"相关主题"推荐卡片 |

#### 核心算法说明

```
step 1: 计算每个主题的中心向量
  topic_centroid[topic_id] = 该主题下所有论文向量的平均值

step 2: 对于目标主题 A，计算 A 与所有其他主题的

---

## 4. Stage 5 — 智能化与生产部署

### 为什么要做 Stage 5？

Stage 3 让系统好用，Stage 4 让系统有深度，Stage 5 让系统更智能——用 LLM 生成自然语言主题名称，并完善本地运行体验。

> **部署策略说明**：本项目采用**本地部署**方式运行，不使用 Docker 容器化或服务器部署。所有功能均在本地 conda 环境中运行。

### Stage 5 目标

> 用 LLM 提升智能感，优化本地运行体验，达到可演示的完整产品状态。

| 功能模块 | 说明 | 优先级 |
|----------|------|--------|
| LLM 智能主题命名 | GPT-4o-mini/Qwen 为主题生成自然语言名称 | 高 |
| 查询结果缓存 | 相同关键词结果缓存 24 小时，秒级响应 | 中 |
| 统一日志系统 | 替换 print()，支持日志级别控制 | 中 |
| 异步并行抓取 | asyncio 替代 ThreadPoolExecutor | 中 |
| ~~Docker 容器化~~ | ~~一键部署，无需手动配置 conda~~ | ~~低~~ **已取消（本地部署）** |
| 单元/集成测试 | 提高代码稳健性 | 低 |

---

### 4.1 功能一：LLM 智能主题命名

#### 当前问题

BERTopic 用关键词拼接生成主题名（如 `transformer_attention_model`），用户难以理解这是什么研究方向。

LLM 可以生成更直观的名称：

```
当前（关键词拼接）：      期望（LLM生成）：
  transformer_attention  ->  Transformer 与注意力机制
  training_loss_gradient ->  模型训练优化方法
  image_detection_object ->  目标检测与图像识别
```

#### 技术方案

调用 GPT-4o-mini（每次约 $0.001）或 Qwen-turbo（国内可用），输入主题关键词 + 代表性摘要，输出"英文名 | 中文名"格式。

Prompt 模板：

```
以下是一个研究主题的关键词和代表性论文摘要：
关键词：{keywords}
摘要1：{abstract1[:200]}
摘要2：{abstract2[:200]}

请为这个研究主题生成简洁准确的名称：
- 英文：不超过6个词的短语
- 中文：不超过8个汉字
格式：英文名 | 中文名
示例输出：Parameter-Efficient Fine-Tuning | 参数高效微调
```

#### 成本估算（以10个主题为例）

```
每个主题：输入约 500 token + 输出约 20 token
GPT-4o-mini 单价：$0.15/1M 输入 + $0.60/1M 输出
10 个主题合计：约 $0.001（人民币约 0.007 元）
-> 每次搜索命名成本可忽略不计
```

#### 需要新建/修改的文件

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/core/nlp/topic_namer.py` | 新建 | `TopicNamer` 类，支持 openai/qwen/ollama 三种 provider |
| `env.example` | 修改 | 新增 `OPENAI_API_KEY=`、`LLM_PROVIDER=openai` 配置项 |
| `src/core/config.py` | 修改 | 新增 LLM 相关配置读取 |
| `src/core/nlp/topic_modeling.py` | 修改 | `TopicModeler` 新增 `name_topics_with_llm(namer)` 方法 |

#### TopicNamer 接口设计

```
TopicNamer(provider="openai", model="gpt-4o-mini")

name_topic(keywords, sample_abstracts)
  输入：关键词列表 + 代表性摘要片段
  输出：{"en": "Attention Mechanism", "zh": "注意力机制"}

name_all_topics(topic_modeler)
  批量命名所有主题，自动结果缓存（避免重复调用 API）
  输出：{0: {"en": "...", "zh": "..."}, 1: {...}}
```

#### 三种 LLM Provider 对比

| 模型 | 成本/次 | 质量 | 国内可用 | 适用场景 |
|------|---------|------|---------|---------|
| GPT-4o-mini | ~$0.001 | 优秀 | 需梯子 | 国际用户首选 |
| Qwen-turbo | ~¥0.001 | 良好 | 直连 | 国内用户首选 |
| Ollama 本地 | 免费 | 良好 | 完全离线 | 无网络环境备选 |

---

### 4.2 功能二：查询结果缓存

#### 为什么需要？

同一关键词多次搜索会重复抓取+训练，浪费时间和 API 次数。
缓存后，相同关键词第二次搜索从 60 秒降至 1 秒。

#### 缓存设计方案

```
Key：MD5(query + sources + max_results)
Value：{
  "papers": [...],           # 论文列表
  "topics": [...],           # 主题分配结果
  "hierarchy_json": {...},   # 层级结构
  "cached_at": "2026-04-16T10:00:00"
}
过期时间：24 小时（可通过 CACHE_TTL_HOURS 配置）
最大缓存条数：10 条（防止内存占用过大）
```

#### 需要新建/修改的文件

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/core/cache/query_cache.py` | 新建 | `QueryCache` 类，LRU 缓存 + TTL 过期检查 |
| `src/core/api/routes.py` | 修改 | 搜索 API 加入缓存逻辑（先查缓存，未命中再处理） |
| `src/core/config.py` | 修改 | 新增 `CACHE_TTL_HOURS = 24` 配置项 |

---

### 4.3 功能三：统一日志系统

#### 当前问题

代码中大量 `print()` 语句：
1. 无法控制输出详细程度（开发 vs 生产模式）
2. 不保存日志文件（重启后丢失）
3. 格式混乱，无时间戳、无模块名

#### 解决方案：Python 标准 logging 模块

统一输出格式：

```
2026-04-16 18:30:15 [INFO ] aggregator.py:45 - 开始抓取 200 篇论文
2026-04-16 18:30:16 [DEBUG] chroma_store.py:78 - 缓存命中: arxiv_1706.03762
2026-04-16 18:30:45 [WARN ] topic_modeling.py:125 - 主题数量较少(3)，建议增加论文数
```

通过环境变量 `LOG_LEVEL=DEBUG/INFO/WARNING` 控制输出详细程度。

#### 需要新建/修改的文件

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/core/logging_config.py` | 新建 | `setup_logging(level, log_file)` 函数 |
| `src/core/` 下所有模块 | 修改 | 将 `print()` 替换为 `logger.info/debug/warning()` |

---

### 4.4 功能四：异步并行抓取优化（可选）

#### 当前 vs 优化对比

```
当前（ThreadPoolExecutor，Stage 2 已实现）：
  线程池并行，有线程切换开销，但对 2-3 个数据源已足够

优化（asyncio + httpx 原生异步）：
  协程并发，无线程开销
  内存减少约 50%，速度提升约 10-30%
```

#### 影响范围

| 文件 | 改动内容 |
|------|---------|
| `src/core/sources/base.py` | `search()` 改为 `async def search()` |
| `src/core/sources/arxiv_adapter.py` | `search()` 改为异步 |
| `src/core/openalex/client.py` | `search()` 改为异步（httpx 原生支持 async） |
| `src/core/sources/aggregator.py` | `ThreadPoolExecutor` 替换为 `asyncio.gather()` |

> 注意：这是破坏性改动，影响所有调用方。建议等其他功能稳定后再做此重构。

---

### 4.5 ~~功能五：Docker 容器化部署~~（已取消）

> **说明**：本项目决定采用**本地部署**方式，不进行 Docker 容器化。
> 本地启动方式见 [`QUICKSTART.md`](../QUICKSTART.md)：
> ```bash
> # 后端
> conda activate literature_review && cd src && python main.py
> # 前端
> conda activate literature_review && streamlit run app.py
> ```

---

### 4.6 Stage 5 实施步骤清单

| 步骤 | 任务 | 涉及文件 | 工作量 |
|------|------|----------|--------|
| 5.1 | LLM 主题命名模块 | `src/core/nlp/topic_namer.py`（新建） | 3小时 |
| 5.2 | 配置 LLM 密钥 | `env.example`、`src/core/config.py`（修改） | 0.5小时 |
| 5.3 | 接入命名到建模流程 | `src/core/nlp/topic_modeling.py`（修改） | 1小时 |
| 5.4 | 查询缓存模块 | `src/core/cache/query_cache.py`（新建） | 2小时 |
| 5.5 | 缓存接入搜索 API | `src/core/api/routes.py`（修改） | 1小时 |
| 5.6 | 统一日志配置 | `src/core/logging_config.py`（新建）+ 多文件修改 | 3小时 |
| 5.7 | 异步抓取重构（可选） | `aggregator.py` 等 4 个文件 | 5小时 |
| ~~5.8~~ | ~~Docker 容器化~~ | ~~已取消，本地部署~~ | ~~4小时~~ |
| 5.8 | 测试覆盖补充 | `tests/` 目录完善 | 4小时 |
| 5.9 | 文档更新 | `README.md`、`QUICKSTART.md`（修改） | 2小时 |

**Stage 5 合计预估工作量**：约 21.5 小时（3 个工作日，已去除 Docker 部分）

---

## 5. 各阶段依赖关系图

```
Stage 1 (已完成) --+
                   +--> Stage 2 (已完成) --+
                                          +--> Stage 3 --> Stage 4 --> Stage 5
                                          |    (可视化)   (个性化)   (智能化)
                                          |
                                          +--> Stage 5.1 LLM命名（可提前做，不依赖3/4）
```

| Stage | 必须前置 | 可跳过先做 |
|-------|---------|-----------|
| Stage 3 ✅ | Stage 2（ChromaDB + 层级主题） | Stage 4、5 |
| Stage 4 | Stage 2（Paper统一模型 + ChromaDB） | Stage 3 部分功能 |
| Stage 5 LLM命名 | Stage 2（主题建模已完成） | Stage 3、4 |
| ~~Stage 5 Docker~~ | ~~Stage 3+4（所有功能完成）~~ | 已取消，本地部署 |

**建议执行顺序**：~~Stage 3（全部）~~✅ -> Stage 4（收藏/导出优先）-> Stage 5（LLM命名优先）-> 其余按需实施

---

## 6. 技术决策说明

### 决策1：可视化库为什么选 Plotly？

| 库 | 优势 | 劣势 | 结论 |
|----|------|------|------|
| Plotly | Streamlit 原生集成；交互式；支持 Sunburst/Treemap | 需加载 JS | 首选 |
| Matplotlib | 简单成熟 | 静态图，不支持交互 | 仅后台分析 |
| ECharts | 功能强大 | 需额外集成 | 备选 |
| D3.js | 最灵活 | 需写 JavaScript | 不考虑 |

### 决策2：用户收藏为什么用 SQLite？

| 方案 | 优势 | 劣势 | 结论 |
|------|------|------|------|
| SQLite | Python内置；单文件；零配置 | 不支持高并发写 | 首选（个人使用） |
| PostgreSQL | 功能完整；支持并发 | 需安装服务 | 多用户场景再考虑 |
| JSON 文件 | 最简单 | 无查询能力 | 不适合 |

### 决策3：LLM 主题命名用哪个模型？

| 模型 | 成本/次 | 质量 | 国内可用 | 推荐 |
|------|---------|------|---------|------|
| GPT-4o-mini | ~$0.001 | 优秀 | 需梯子 | 国际首选 |
| Qwen-turbo | ~¥0.001 | 良好 | 直连 | 国内首选 |
| Ollama 本地 | 免费 | 良好 | 完全离线 | 无网络备选 |

通过环境变量 `LLM_PROVIDER` 配置，代码同时支持三种，用户按需选择。

### 决策4：异步改造是否必须？

**结论：不是必须，Stage 5 末期再做**

`ThreadPoolExecutor` 对 2-3 个数据源已经足够。asyncio 改造需修改 4-5 个核心文件，风险较高。建议等功能完整稳定后再做此性能优化。

---

## 7. 优先级与时间估算

### 总工作量汇总

| 阶段 | 主要交付物 | 预估工作量 | 建议周期 |
|------|-----------|-----------|---------|
| Stage 3 ✅ | Sunburst图、趋势图、进度条、主题钻取 | ~~24.5 小时~~ **已完成** | 第3周 |
| Stage 4 | 收藏/笔记、BibTeX导出、相关推荐、引用网络 | 26 小时 | 第4-5周 |
| Stage 5 | LLM命名、缓存、日志（~~Docker已取消~~） | 21.5 小时 | 第6-7周 |
| 合计 | | 约 47.5 小时（剩余） | 约4周 |

### 最小可演示版本（MVP，1周内可完成）

```
必做（演示价值最高）：
  Stage 3 - Sunburst 主题地图     最直观，导师最关注
  Stage 3 - 进度条改造            用户体验立竿见影
  Stage 5.1 - LLM 主题命名        命名质量跨越式提升

可选（时间允许再做）：
  Stage 3 - 研究趋势图
  Stage 4 - 论文收藏功能
  Stage 4 - BibTeX 导出
```

### 功能价值/成本分析

```
高价值、低成本（优先做）：
  [推荐] Plotly Sunburst 主题地图  视觉冲击力强，代码量少
  [推荐] 进度条显示               用户体验大幅改善
  [推荐] LLM 主题命名             命名质量跨越式提升，只需新建1个文件
  [推荐] BibTeX 导出              融入学术工作流，实用价值高

高价值、较高成本（次优先）：
  [建议] 主题钻取交互             探索式导航核心，需前后端配合
  [建议] 论文收藏+笔记            长期使用价值，需新建DB+前端页面
  [建议] 研究趋势时间轴           可视化加分项

较高成本、中等价值（按需实施）：
  [可选] 引用关系网络             需要 Semantic Scholar API
  [可选] 异步抓取重构             性能优化，改动范围大
  [可选] Docker 容器化            部署便利性，不影响功能
```

---

## 附录：完整文件变更清单

### Stage 3 涉及文件

| 操作 | 文件路径 |
|------|---------|
| 新建 | `src/core/nlp/trend_analysis.py` |
| 修改 | `src/core/api/topic_routes.py` |
| 修改 | `app.py` |
| 修改 | `requirements.txt`（加 plotly>=5.18.0） |
| 新建 | `src/test_stage3_viz.py` |

### Stage 4 涉及文件

| 操作 | 文件路径 |
|------|---------|
| 新建 | `src/core/storage/user_library.py` |
| 新建 | `src/core/api/library_routes.py` |
| 新建 | `src/core/nlp/topic_similarity.py` |
| 新建 | `src/core/sources/semantic_scholar.py` |
| 新建 | `src/core/nlp/citation_graph.py` |
| 修改 | `src/core/config.py` |
| 修改 | `src/main.py` |
| 修改 | `src/core/api/topic_routes.py` |
| 修改 | `app.py` |
| 修改 | `requirements.txt`（加 pyvis>=0.3.2） |
| 新建 | `src/test_stage4_library.py` |

### Stage 5 涉及文件

| 操作 | 文件路径 |
|------|---------|
| 新建 | `src/core/nlp/topic_namer.py` |
| 新建 | `src/core/cache/query_cache.py` |
| 新建 | `src/core/logging_config.py` |
| ~~新建~~ | ~~`docker/Dockerfile.backend`~~ （已取消，本地部署） |
| ~~新建~~ | ~~`docker/Dockerfile.frontend`~~ （已取消，本地部署） |
| ~~新建~~ | ~~`docker-compose.yml`~~ （已取消，本地部署） |
| 修改 | `src/core/config.py` |
| 修改 | `src/core/nlp/topic_modeling.py` |
| 修改 | `src/core/api/routes.py` |
| 修改 | `env.example` |
| 修改 | `README.md`、`QUICKSTART.md` |

---

*规划文档编写时间：2026-04-16*

---

## Stage 6 — PDF 下载 与 智能来源路由（已确认方案）

> 详细设计文档见 [`docs/STAGE6_PDF_AND_SMART_ROUTING.md`](docs/STAGE6_PDF_AND_SMART_ROUTING.md)

### 功能一：论文 PDF 三层降级下载

采用"直接下载 → Unpaywall API → Semantic Scholar API"三层降级策略：

| 层级 | 方式 | 预期覆盖率 |
|------|------|-----------|
| Layer 1 | 直接下载 `pdf_url` / 构造 arXiv PDF URL | arXiv 论文 100% |
| Layer 2a | Unpaywall API（免费合法，需配置邮箱） | 开放获取论文 ~50% |
| Layer 2b | Semantic Scholar API（免费，有速率限制） | ~30% |
| Layer 3 | 失败，返回论文页面链接 | — |

**综合覆盖率估算：60-75%**（不集成 Sci-Hub，避免法律风险和维护成本）

新增文件：`src/core/downloader/` 模块（4个文件）
建议实施阶段：**Stage 4**（与收藏功能配合，"收藏并下载"）

### 功能二：零样本分类智能来源路由

使用 `cross-encoder/nli-MiniLM-L2-mnli`（~90MB）对用户输入主题进行零样本学科分类，自动计算各数据源的最优抓取比例：

```
示例：输入 "CRISPR gene editing in cancer therapy"
  → 检测到：生物学(68%) + 医学(32%)
  → 自动分配：arXiv 17% (34篇) | OpenAlex 83% (166篇)
```

**无需新增任何 Python 包**，复用现有 `transformers` 依赖。

新增文件：`src/core/nlp/topic_classifier.py`
修改文件：`src/core/sources/aggregator.py`（新增 `weights` 参数，向后兼容）
建议实施阶段：**Stage 3**（改动小，可提前实施）
