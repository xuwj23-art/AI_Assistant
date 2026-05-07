# Stage 3 升级总结 — 可视化与体验升级

> **版本**: v3.0 | **完成日期**: 2026-04-29
> **前置依赖**: Stage 1（Bug修复）✅ | Stage 2（多源抓取 + ChromaDB + 层级主题）✅
> **本阶段状态**: ✅ 全部完成（10/10 步骤，5/5 单元测试通过）

---

## 目录

1. [Stage 3 目标回顾](#1-stage-3-目标回顾)
2. [新增功能详解](#2-新增功能详解)
3. [文件变更清单](#3-文件变更清单)
4. [API 端点说明](#4-api-端点说明)
5. [前端界面说明](#5-前端界面说明)
6. [测试方式](#6-测试方式)
7. [启动与使用](#7-启动与使用)
8. [已知限制与后续计划](#8-已知限制与后续计划)

---

## 1. Stage 3 目标回顾

Stage 2 完成后，系统已能正确工作，但用户体验较原始：

| 问题 | Stage 3 解决方案 |
|------|----------------|
| 主题只是文字列表，无法感知规模 | Plotly Sunburst 旭日图，扇区大小 = 论文数量 |
| 搜索时界面卡住，不知道进度 | 4 阶段实时进度条 |
| 无法探索子主题 | 点击主题 → 右侧论文列表联动 + 相关主题推荐 |
| 无法判断研究方向热度 | 研究趋势折线图 + 增长率自动计算 |

**核心目标**：让用户能**看到**论文分布，能**点击探索**主题，能**实时看到**抓取进度。

---

## 2. 新增功能详解

### 2.1 交互式主题地图（Sunburst 旭日图）

**位置**：前端"📊 主题地图"标签页

**功能描述**：
- 圆形分层图表，中心 = "All Topics"，外圈每个扇区 = 一个研究主题
- 扇区大小反映该主题的论文数量
- 颜色深浅表示相对规模（蓝色系渐变）
- 鼠标悬停显示主题名称和论文数

**交互流程**：
```
进入"主题地图"页
    ↓
点击"加载主题地图"按钮
    ↓
Sunburst 图渲染（每个扇区 = 一个研究主题）
    ↓
下拉框选择主题
    ↓
右侧显示该主题下的论文列表（支持分页/排序）
    ↓
底部显示"相关主题推荐"（点击可切换主题）
```

**数据来源**：`GET /api/topics/sunburst` 端点

---

### 2.2 主题钻取与论文列表

**位置**：主题地图页右侧面板

**功能描述**：
- 选择主题后，右侧实时加载该主题下的论文列表
- 支持两种排序方式：
  - **时间降序**（最新论文优先）
  - **相关性**（原始顺序，BERTopic 分配顺序）
- 支持分页（每页 10 篇，可翻页）
- 每篇论文展示：标题、作者、年份、摘要前 300 字、PDF 链接
- 底部显示 3 个相关主题推荐（基于 Jaccard 关键词相似度），点击可直接切换

**数据来源**：
- `GET /api/topics/{id}/papers?page=1&page_size=10&sort_by=date`
- `GET /api/topics/{id}/similar?top_n=3`

---

### 2.3 研究趋势时间轴

**位置**：前端"📈 研究趋势"标签页

**功能描述**：
- 折线图展示各研究主题在不同年份的论文数量变化
- 自动标注近期增长最快的主题（橙色标签，显示增长倍率）
- 可调节显示主题数量（3-20 个）
- 附原始数据表格（可展开查看）

**增长率算法**：
```
增长率 = (最近2年平均论文数 + 1) / (前3年平均论文数 + 1)
（+1 避免除以零）
```

**示例输出**：
```
热门方向: ↑6.2x Transformer  ↑3.1x BERT/GPT  ↑2.5x LLM

论文数
  120│                               ╭─── Transformer
   80│                        ╭─────╯
   60│              ╭────────╯           ╭── BERT/GPT
   40│       ╭─────╯     ╭─────────────╯
     └──────────────────────────────────── 年份
       2018  2019  2020  2021  2022  2023
```

**数据来源**：`GET /api/topics/trends?top_n=8`

---

### 2.4 实时 4 阶段进度条

**位置**：前端"💬 AI 对话"标签页

**功能描述**：
将原来"界面卡住 → 突然出结果"的体验改为 4 阶段实时反馈：

```
阶段 1/4: 理解问题并检索相关论文    [████░░░░░░] 25%
阶段 2/4: 从向量数据库检索文档      [█████░░░░░] 50%
阶段 3/4: 生成 AI 回答              [████████░░] 75%
阶段 4/4: 完成！                    [██████████] 100%
```

---

## 3. 文件变更清单

### 新建文件

| 文件路径 | 说明 |
|---------|------|
| [`src/core/nlp/trend_analysis.py`](../src/core/nlp/trend_analysis.py) | 趋势分析工具函数模块 |
| [`src/test_stage3_viz.py`](../src/test_stage3_viz.py) | Stage 3 集成测试（单元测试 + API 测试） |

### 修改文件

| 文件路径 | 修改内容 |
|---------|---------|
| [`src/core/api/models.py`](../src/core/api/models.py) | 新增 4 个响应模型：`SunburstResponse`、`TrendResponse`、`TrendTopicSeries`、`PaperListPagedResponse` |
| [`src/core/api/topic_routes.py`](../src/core/api/topic_routes.py) | 新增 3 个端点，升级 1 个端点（详见第 4 节） |
| [`app.py`](../app.py) | 全面重构为三标签页界面（详见第 5 节） |

---

## 4. API 端点说明

### 新增端点

#### `GET /api/topics/sunburst`

返回 Plotly Sunburst 图所需的平铺格式数据。

**响应示例**：
```json
{
  "labels":    ["All Topics", "Topic 0: transformer_attention", "Topic 1: bert_language"],
  "parents":   ["",           "All Topics",                    "All Topics"],
  "values":    [0,            80,                              45],
  "ids":       ["root",       "topic_0",                       "topic_1"],
  "topic_ids": [-1,           0,                               1]
}
```

**前端使用**：
```python
import plotly.graph_objects as go
fig = go.Figure(go.Sunburst(
    labels=data["labels"],
    parents=data["parents"],
    values=data["values"],
    ids=data["ids"],
    branchvalues="total"
))
```

---

#### `GET /api/topics/trends?top_n=10`

按年份统计各主题论文数量，返回趋势数据和增长最快的主题。

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `top_n` | int | 10 | 返回主题数量上限（1-30） |

**响应示例**：
```json
{
  "years": [2019, 2020, 2021, 2022, 2023],
  "topics": [
    {"name": "transformer_attention", "counts": [5, 12, 28, 45, 80]},
    {"name": "bert_language_model",   "counts": [3, 8,  20, 35, 60]}
  ],
  "trending": [
    {"name": "transformer_attention", "growth_rate": 2.5},
    {"name": "bert_language_model",   "growth_rate": 2.1}
  ]
}
```

---

#### `GET /api/topics/{topic_id}/similar?top_n=5`

基于关键词 Jaccard 相似度推荐相关主题。

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `topic_id` | int | 必填 | 目标主题 ID |
| `top_n` | int | 5 | 返回相关主题数量（1-20） |

**响应示例**：
```json
{
  "topic_id": 0,
  "topic_name": "transformer_attention",
  "similar_topics": [
    {"topic_id": 3, "topic_name": "attention_mechanism", "paper_count": 32, "similarity": 0.4286},
    {"topic_id": 1, "topic_name": "bert_language_model", "paper_count": 45, "similarity": 0.2857}
  ]
}
```

---

### 升级端点

#### `GET /api/topics/{topic_id}/papers`（升级）

**新增参数**（原来只有 `limit`）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `page` | int | 1 | 页码（从1开始） |
| `page_size` | int | 20 | 每页数量（5-100） |
| `sort_by` | str | `relevance` | 排序方式：`relevance` 或 `date` |

**响应格式变更**（从 `PaperListResponse` 升级为 `PaperListPagedResponse`）：
```json
{
  "topic_name": "transformer_attention",
  "total": 80,
  "page": 1,
  "page_size": 20,
  "sort_by": "date",
  "papers": [...]
}
```

---

## 5. 前端界面说明

### 整体布局

```
┌─────────────────────────────────────────────────────────┐
│  侧边栏                │  主界面                         │
│  ─────────────────     │  📚 AI 科研文献助手              │
│  后端状态: 🟢 在线     │                                  │
│  功能介绍              │  [📊 主题地图] [📈 研究趋势] [💬 AI对话] │
│  使用技巧              │                                  │
│                        │  （标签页内容）                  │
└─────────────────────────────────────────────────────────┘
```

### 标签页 1：主题地图

```
┌──────────────────────────────┬─────────────────────────┐
│  [加载主题地图] 按钮          │  主题论文列表            │
│                              │  当前主题: transformer   │
│  ┌────────────────────────┐  │  排序: [时间降序 ▼]      │
│  │   Plotly Sunburst 图   │  │  页码: [1]               │
│  │   （交互式旭日图）      │  │                          │
│  └────────────────────────┘  │  > 论文1 标题...         │
│                              │  > 论文2 标题...         │
│  选择主题: [Topic 0: ... ▼]  │  > 论文3 标题...         │
│                              │  ─────────────────────   │
│                              │  相关主题推荐:            │
│                              │  → attention_mechanism   │
│                              │  → bert_language_model   │
└──────────────────────────────┴─────────────────────────┘
```

### 标签页 2：研究趋势

```
┌─────────────────────────────────────────────────────────┐
│  显示主题数量: [────●────] 8                             │
│  [加载趋势数据] 按钮                                     │
│                                                         │
│  热门方向: [↑6.2x Transformer] [↑3.1x BERT/GPT]        │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Plotly 折线图                       │   │
│  │  论文数                                          │   │
│  │   120│                          ╭── Transformer  │   │
│  │    60│              ╭──────────╯                 │   │
│  │     0└──────────────────────────── 年份          │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ▶ 查看原始数据表格                                      │
└─────────────────────────────────────────────────────────┘
```

### 标签页 3：AI 对话

```
┌─────────────────────────────────────────────────────────┐
│  [用户消息]                                              │
│  What is the Transformer architecture?                  │
│                                                         │
│  [AI 回答]                                              │
│  The Transformer architecture...                        │
│  ▶ 查看 3 篇参考论文                                     │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│  输入你的问题（支持中英文）...              [发送]       │
│                                                         │
│  [🗑️ 清空对话历史]                                      │
└─────────────────────────────────────────────────────────┘
```

---

## 6. 测试方式

### 6.1 单元测试（无需后端，推荐先运行）

```bash
# 在项目根目录运行
cd e:/AIassistant_v2
python src/test_stage3_viz.py --unit
```

**测试覆盖内容**：

| 测试名称 | 测试内容 |
|---------|---------|
| `test_trend_analysis_basic` | `compute_topic_trends()` 基本功能：年份统计、噪声过滤 |
| `test_trend_analysis_empty` | 空输入处理：空列表 → 空 DataFrame |
| `test_get_trending_topics` | 增长率计算：快速增长 > 稳定 > 下降 |
| `test_extract_year_formats` | 日期解析：datetime/Timestamp/字符串/整数/None/NaT |
| `test_trend_df_to_api_format` | API 格式转换：DataFrame → {years, topics} |

**预期输出**：
```
============================================================
  Stage 3 单元测试（无需后端）
============================================================
  [OK]   DataFrame 形状: (6, 2)
  [OK]   年份范围: 2021 - 2026
  [OK]   compute_topic_trends 基本功能正常
  ...
============================================================
  单元测试结果: 5 通过 / 0 失败
============================================================
  [PASS] 所有测试通过！Stage 3 可视化功能就绪。
============================================================
```

---

### 6.2 API 集成测试（需要后端在线）

**前置条件**：
1. 已训练主题模型（运行过 `python src/scripts/train_topics.py`）
2. 后端服务已启动（`cd src && python main.py`）

```bash
# 运行全部测试（单元测试 + API 测试）
python src/test_stage3_viz.py
```

**API 测试覆盖内容**：

| 测试名称 | 测试端点 | 验证内容 |
|---------|---------|---------|
| `test_api_sunburst` | `GET /api/topics/sunburst` | 响应字段完整性、数组长度一致、根节点正确 |
| `test_api_trends` | `GET /api/topics/trends` | 响应字段完整性、年份/主题/趋势数据 |
| `test_api_topics_list` | `GET /api/topics` | 主题列表、total 字段 |
| `test_api_topic_papers_paged` | `GET /api/topics/{id}/papers` | 分页正确、排序参数、404 处理 |
| `test_api_similar_topics` | `GET /api/topics/{id}/similar` | 相关主题数量、相似度字段 |

---

### 6.3 手动 API 测试（curl）

```bash
# 测试 Sunburst 数据
curl http://127.0.0.1:8000/api/topics/sunburst

# 测试趋势数据（显示前8个主题）
curl "http://127.0.0.1:8000/api/topics/trends?top_n=8"

# 测试主题列表
curl http://127.0.0.1:8000/api/topics

# 测试分页论文列表（主题0，第1页，按时间排序）
curl "http://127.0.0.1:8000/api/topics/0/papers?page=1&page_size=10&sort_by=date"

# 测试相关主题推荐
curl "http://127.0.0.1:8000/api/topics/0/similar?top_n=5"
```

---

### 6.4 前端手动测试

```bash
# 启动后端
cd src && python main.py

# 新开终端，启动前端
cd e:/AIassistant_v2
streamlit run app.py
```

**手动测试检查清单**：

- [ ] 侧边栏显示"🟢 后端服务在线"
- [ ] "主题地图"标签页：点击"加载主题地图"后 Sunburst 图正常渲染
- [ ] 下拉框选择主题后，右侧论文列表正常加载
- [ ] 切换排序方式（时间/相关性），论文顺序变化
- [ ] 翻页功能正常（第2页显示不同论文）
- [ ] 相关主题推荐按钮可点击切换主题
- [ ] "研究趋势"标签页：点击"加载趋势数据"后折线图正常渲染
- [ ] 热门主题标签正确显示增长倍率
- [ ] 展开"查看原始数据表格"正常显示
- [ ] "AI 对话"标签页：发送问题时显示 4 阶段进度条
- [ ] 对话历史正常保存，"清空对话历史"按钮有效

---

## 7. 启动与使用

### 完整启动流程

```bash
# 步骤 1：在线抓取论文（首次使用必须运行）
# ⚠️ 注意：fetch_papers.py 会实时联网从 arXiv / OpenAlex 在线抓取论文元数据，
# 无需手动下载任何文件，运行时自动完成在线下载并缓存到本地 ChromaDB。
python src/scripts/fetch_papers.py --query "deep learning" --max-results 200

# 步骤 1b：训练主题模型（依赖步骤 1 抓取的数据）
python src/scripts/train_topics.py

# 步骤 2：启动后端
cd src
conda activate literature_review
python main.py
# 后端运行在 http://127.0.0.1:8000

# 步骤 3：启动前端（新终端）
cd e:/AIassistant_v2
streamlit run app.py
# 前端运行在 http://localhost:8501
```

### 快速验证

```bash
# 验证后端健康
curl http://127.0.0.1:8000/health

# 验证 Stage 3 新端点
curl http://127.0.0.1:8000/api/topics/sunburst | python -m json.tool | head -20
```

---

## 8. 已知限制与后续计划

### 当前限制

| 限制 | 说明 | 后续解决方案 |
|------|------|------------|
| Sunburst 只有一层 | 当前只展示顶层主题，无子主题层级 | Stage 4 接入 BERTopic `hierarchical_topics()` 实现多层 |
| 相关主题用关键词相似度 | Jaccard 相似度较粗糙 | Stage 4 改用向量余弦相似度（`topic_similarity.py`） |
| 趋势图依赖 `published` 字段 | 若数据缺少日期则趋势图为空 | 确保抓取时填充 `published_date` |
| 进度条为模拟进度 | 4 阶段进度是估算，非真实流式 | Stage 5 改用 SSE/WebSocket 实现真实流式进度 |

### Stage 4 计划（下一阶段）

- 论文收藏与个人笔记（SQLite 本地数据库）
- BibTeX / Markdown 导出
- 相关主题推荐（向量余弦相似度升级版）
- Semantic Scholar 引用关系网络

---

*文档编写时间：2026-04-29*
