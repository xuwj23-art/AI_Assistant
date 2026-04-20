# 科研文献综述 AI 助手 — 升级方案与路线图

> 基于导师需求（多来源、在线下载、层级主题分类、探索式导航）整理
> 参考文件：Interactive AI Assistant for Scientific Literature Review.pdf

---

## 一、导师原始需求分析

导师 Dr. XU Lingling 的项目说明书核心工作流为：

**Search → Ingest → Cluster → Explore → Summarize & Synthesize**

在现有 Demo 基础上，导师希望进一步实现：
1. **多来源论文**（不只是 arXiv）
2. **在线实时下载**（非离线数据集）
3. **层级主题分类**（大领域 → 子主题 → 更细子主题）
4. **探索式研究导航**（研究人员可以"漫游"主题地图）

---

## 二、导师方案可行性评估

**结论：技术上完全可行，但有关键性能瓶颈需提前规避。**

| 挑战 | 风险等级 | 说明 |
|------|----------|------|
| 多来源 API 速率限制 | 🟡 中 | Semantic Scholar 免费 API 限速 100 req/5min |
| 实时聚类性能 | 🔴 高 | BERTopic 在 500+ 篇论文上训练需 30-60 秒，用户等待体验差 |
| 层级聚类一致性 | 🟡 中 | 多次独立训练的二级聚类与一级可能不连贯 |
| 重复论文去重 | 🟡 中 | 多来源抓取同一论文需 DOI/标题相似度去重 |

---

## 三、整体架构升级方案

### 推荐新架构（流式检索 + 向量缓存）

```
用户输入关键词
    ↓
多源并行异步抓取 (arXiv + OpenAlex + Semantic Scholar)
    ↓
DOI / 标题去重
    ↓
向量化 (SentenceTransformer)
    ↓
ChromaDB 持久化存储 ←→ 缓存命中检查
    ↓
BERTopic 层级聚类 (一次训练, 多层结果)
    ↓
LLM 自动命名主题 (GPT-4o-mini / 本地 Qwen)
    ↓
Plotly Sunburst 主题地图 (交互式)
    ↓
用户点击某主题
    ↓
├── 递归细分 (对子集用更小 min_cluster_size 再聚类)
├── 主题综述生成 (RAG + LLM)
└── 相关主题推荐 (向量余弦相似度)
```

**关键架构转变**：从"批量训练时聚类"改为"查询时向量检索 + 动态分组"，解决实时性瓶颈。

---

## 四、六大升级维度详解

### 升级 1：多来源数据抓取

**推荐数据源优先级**：

| 来源 | API | 优势 | 建议 |
|------|-----|------|------|
| **OpenAlex** | REST（完全免费，无限制） | 覆盖所有学科，数据最全，含引用关系 | ⭐ 首选接入 |
| **arXiv** | 官方 SDK（已有） | CS/AI 领域最全，预印本最新 | ✅ 保留 |
| **Semantic Scholar** | REST（免费 100req/5min） | 引用网络、作者影响力分析 | 🟡 第二优先 |
| **PubMed** | NCBI E-utilities | 医学/生物领域权威 | 🟢 按需接入 |
| CrossRef | REST（免费） | DOI 权威来源 | 🟢 辅助去重用 |

**推荐实现：统一 PaperSource 接口**

```python
# src/core/sources/base.py
class PaperSource(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int) -> List[Paper]:
        ...

# src/core/sources/arxiv_source.py
class ArxivSource(PaperSource): ...

# src/core/sources/openalex_source.py
class OpenAlexSource(PaperSource): ...

# src/core/sources/aggregator.py
class MultiSourceAggregator:
    """并行抓取多源，自动去重"""
    async def search_all(self, query, max_results):
        tasks = [source.search(query, max_results) for source in self.sources]
        results = await asyncio.gather(*tasks)
        return self._deduplicate(flatten(results))
```

---

### 升级 2：层级主题探索

**推荐方案：BERTopic 原生层级模式（一次训练，多层结果）**

BERTopic 原生支持 `hierarchical_topics()` 方法，无需多次训练：

```python
# 一次训练即可获取层级结构
topics, probs = topic_model.fit_transform(docs, embeddings)
hierarchical_topics = topic_model.hierarchical_topics(docs)

# 可视化层级树
fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

# 获取某层级的子主题
# 用 parent_id 过滤即可实现导航
```

**备选方案：动态递归细分（更灵活但更慢）**

```python
def drill_down(papers_subset: List[str], level: int = 1):
    min_size = max(3, 10 // level)  # 越深层越细粒度
    sub_model = BERTopic(min_topic_size=min_size, calculate_probabilities=False)
    sub_topics, _ = sub_model.fit_transform(papers_subset)
    return sub_model, sub_topics
```

**建议**：先用 BERTopic 层级模式，层数不够再用递归细分补充。

---

### 升级 3：实时性能优化

| 策略 | 效果 | 实现难度 |
|------|------|----------|
| **ChromaDB 向量缓存**：持久化已抓取论文向量 | 相同主题第二次查询瞬间返回 | 🟢 低（已有目录） |
| **异步并行抓取**：`asyncio + httpx` | 多来源同时抓取，速度提升 3-5x | 🟡 中 |
| **流式前端**：先展示已加载论文，后台继续 | 用户感知等待降低 80% | 🟡 中 |
| **轻量聚类**：UMAP 降维从 5 维改为 2 维 | 速度提升 2x，精度略降 | 🟢 低 |
| **结果缓存**：相同查询词的聚类结果缓存 24h | 热门搜索秒级响应 | 🟢 低 |

---

### 升级 4：LLM 智能主题命名

当前用关键词拼接命名（如 `transformer_attention_model`），可用 LLM 生成自然语言名称：

```python
async def generate_topic_name(keywords: List[str], sample_abstracts: List[str]) -> str:
    prompt = f"""
    以下是一组相关论文的关键词：{keywords[:10]}
    以下是其中两篇论文的摘要片段：
    - {sample_abstracts[0][:200]}
    - {sample_abstracts[1][:200]}
    
    请为这个研究主题生成一个简洁准确的名称：
    - 英文：不超过 6 个词的短语
    - 中文：不超过 8 个汉字
    格式：英文名 | 中文名
    """
    # 例如输出：Parameter-Efficient Fine-Tuning | 参数高效微调
```

---

### 升级 5：知识图谱可视化（探索式体验核心）

这是区别普通文献检索和"探索式导航"的最关键差异，推荐用 **Plotly Sunburst/Treemap**：

```python
import plotly.express as px

# Sunburst 图（层级主题地图）
fig = px.sunburst(
    topic_df,
    path=['domain', 'topic_level1', 'topic_level2'],
    values='paper_count',
    color='avg_year',         # 用颜色表示研究时期新旧
    hover_data=['top_keywords']
)
st.plotly_chart(fig, use_container_width=True)

# 或用 Treemap
fig = px.treemap(topic_df, path=['level1', 'level2'], values='paper_count')
```

**更进阶**：用 `pyvis` 构建论文引用关系网络图。

---

### 升级 6：研究人员个性化功能

| 功能 | 研究价值 | 实现复杂度 |
|------|----------|------------|
| **研究趋势分析**：按年份展示主题热度曲线 | 识别新兴/衰退研究方向 | 🟡 中 |
| **相关主题推荐**：当前主题向量最近邻 | 发现跨领域交叉点 | 🟢 低 |
| **引用网络**：借助 Semantic Scholar / OpenAlex 引用数据 | 定位领域奠基论文 | 🟡 中 |
| **收藏与笔记**：用户标记感兴趣论文 | 持续使用价值 | 🟡 中 |
| **导出功能**：BibTeX / Markdown 综述报告 | 直接融入科研工作流 | 🟢 低 |
| **论文推荐**：基于已收藏论文的相似推荐 | 个性化发现 | 🟡 中 |

---

## 五、推荐优先级路线图

### 阶段 1（立即执行）：修复 + 稳定现有 Demo
- [ ] 修复 `requirements.txt` 编码问题
- [ ] 修复 `train_topics.py` import 路径
- [ ] 修复 `build_vector_store.py` 缺失 import
- [ ] 跑通完整链路：抓取 → 向量化 → 聚类 → 后端 → 前端
- [ ] 更新 `docs/BACKEND_README.md` 使其与代码同步

### 阶段 2（核心功能升级）：多来源 + 层级主题
- [ ] 实现 `PaperSource` 抽象接口
- [ ] 接入 **OpenAlex API**（优先，免费无限制）
- [ ] 启用 **BERTopic 层级模式** `hierarchical_topics()`
- [ ] 实现 **ChromaDB 向量持久化**（缓存已抓取论文）
- [ ] 主题命名改用 LLM（可选 GPT-4o-mini 或本地模型）

### 阶段 3（体验升级）：可视化 + 在线化
- [ ] Plotly **Sunburst/Treemap** 主题地图
- [ ] 异步并行抓取（`asyncio + httpx`）
- [ ] 流式前端进度展示（Streamlit `st.progress`）
- [ ] 研究趋势时间轴可视化

### 阶段 4（深度功能）：个性化 + 导出
- [ ] 接入 Semantic Scholar 引用网络数据
- [ ] 论文收藏与个人笔记功能
- [ ] 导出 BibTeX / Markdown 综述报告
- [ ] 相关主题自动推荐

---

## 六、关键技术决策建议

### 决策 1：ChromaDB 还是 FAISS？

| | ChromaDB | FAISS（当前使用）|
|--|----------|-----------------|
| 持久化 | ✅ 原生支持 | ❌ 需手动保存 |
| 元数据过滤 | ✅ 支持按字段过滤 | ❌ 不支持 |
| 多来源管理 | ✅ Collection 概念自然映射 | ❌ 需自行管理 |
| 性能 | 🟡 比 FAISS 慢 | ✅ 最快 |
| 建议 | 用于主数据库 | 可保留用于纯速度场景 |

**建议**：升级为以 ChromaDB 为核心，FAISS 作为高速检索层。

### 决策 2：LLM 摘要用本地还是 API？

| | 本地 T5/BART（当前）| GPT-4o-mini API |
|--|---------------------|-----------------|
| 成本 | 免费 | 约 $0.001/次 |
| 质量 | 一般 | 优秀 |
| 速度 | 慢（CPU 30s/篇）| 快（2-3s/篇）|
| 离线 | ✅ | ❌ |
| 建议 | 备用方案 | **主推方案** |

**建议**：用 GPT-4o-mini 或 Qwen-turbo（国内可用）做摘要和主题命名，T5 作为离线降级备选。

### 决策 3：BERTopic 层级 vs 递归训练？

**强烈建议用 BERTopic 原生层级模式**，原因：
- 一次训练获得所有层级，速度快 3-10x
- 层级之间语义一致性更好
- 官方维护，API 稳定

---

## 七、与学术界前沿对标

以下是类似定位的已有工具，可作为参考和差异化方向：

| 工具 | 核心功能 | 本项目差异化方向 |
|------|----------|-----------------|
| Semantic Scholar | 论文搜索 + 引用图 | 自动主题聚类 + 综述生成 |
| Connected Papers | 引用关系可视化 | 主题层级探索 + 对话问答 |
| Elicit | AI 辅助文献综述 | 开源自建 + 多来源 + 可定制 |
| Research Rabbit | 论文推荐 | 基于 NLP 聚类而非引用 |
| Litmaps | 文献地图 | 更注重内容语义而非引用结构 |

**本项目最大差异化优势**：
1. 自动主题聚类 + 层级导航（其他工具没有）
2. 基于论文内容语义（而非引用关系）的主题发现
3. 可完全本地化部署（数据隐私）
4. 开源可定制

---

*报告生成时间：2026-04-16*
