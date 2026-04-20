# Stage 2 核心升级总结文档

> **版本**: v2.0 | **日期**: 2026-04-16 | **状态**: 已完成并通过全部测试

---

## 目录

1. [为什么要做这次升级？](#1-为什么要做这次升级)
2. [升级了哪些内容？文件清单](#2-升级了哪些内容文件清单)
3. [新增模块详细说明](#3-新增模块详细说明)
4. [修改的原有文件说明](#4-修改的原有文件说明)
5. [核心设计思路（适合小白）](#5-核心设计思路适合小白)
6. [如何使用这些新功能](#6-如何使用这些新功能)
7. [测试结果](#7-测试结果)
8. [常见问题与解决方法](#8-常见问题与解决方法)
9. [环境配置说明](#9-环境配置说明)

---

## 1. 为什么要做这次升级？

### 1.1 原有架构的问题

Stage 1 结束后，项目可以：
- 从 **arXiv** 单一来源抓取论文
- 对论文做主题建模（BERTopic）
- 用 FAISS 做向量相似度搜索

但存在以下局限：

| 问题 | 影响 |
|------|------|
| **只有 arXiv 一个来源** | 覆盖率低，很多重要论文（尤其是发表在期刊的）抓不到 |
| **主题建模只有一层** | 无法展示主题之间的层级关系（比如"AI"下面有"NLP"、"CV"等子方向） |
| **FAISS 不能持久化** | 每次重启程序都要重新建索引，浪费时间 |
| **重复向量化** | 抓过的论文再次处理时会重复计算向量，浪费资源 |

### 1.2 Stage 2 的三大升级目标

```
升级1: 多源论文库
  旧: 只有 arXiv
  新: arXiv + OpenAlex（并行抓取，去重合并）

升级2: 层级主题建模
  旧: 只有一层平铺主题
  新: 多层树状结构（BERTopic 原生支持）

升级3: 持久化向量存储
  旧: FAISS（内存中，重启就丢失）
  新: ChromaDB（存到硬盘，永久保存）
```

---

## 2. 升级了哪些内容？文件清单

### 2.1 新增文件（13个）

```
src/
├── core/
│   ├── sources/                        ← 新增整个目录
│   │   ├── __init__.py                 ← 模块入口
│   │   ├── base.py                     ← 统一论文数据模型
│   │   ├── arxiv_adapter.py            ← arXiv 适配器
│   │   └── aggregator.py              ← 多源聚合器（去重逻辑在这里）
│   │
│   ├── openalex/                       ← 新增整个目录
│   │   ├── __init__.py                 ← 模块入口
│   │   └── client.py                   ← OpenAlex API 客户端
│   │
│   └── storage/                        ← 新增整个目录
│       ├── __init__.py                 ← 模块入口
│       └── chroma_store.py             ← ChromaDB 持久化存储
│
├── scripts/
│   └── fetch_papers.py                 ← 新增：多源抓取命令行工具
│
└── test_stage2_integration.py          ← 新增：完整链路集成测试
```

### 2.2 修改的原有文件（3个）

```
src/core/config.py              ← 新增 CHROMA_DB_DIR 路径配置
src/core/nlp/topic_modeling.py  ← 新增层级主题方法 + 修复一个Bug
requirements.txt                ← 新增 chromadb、httpx 依赖
```

---

## 3. 新增模块详细说明

### 3.1 `src/core/sources/base.py` — 统一论文数据模型

**这个文件解决了什么问题？**

不同数据库返回的论文格式不一样：
- arXiv 返回的有 `arxiv_id`
- OpenAlex 返回的有 `openalex_id` 和 `doi`

如果不统一格式，后续代码就要写很多"判断来源，分别处理"的逻辑，非常麻烦。

**解决方案：定义一个通用的 `Paper` 类**

```python
class Paper(BaseModel):
    # 用于去重的ID字段（不同来源填不同的）
    doi: Optional[str]          # 如 "10.48550/arXiv.1706.03762"
    arxiv_id: Optional[str]     # 如 "1706.03762"
    openalex_id: Optional[str]  # 如 "W2741809807"
    
    # 所有来源都有的基础字段
    title: str          # 论文标题
    abstract: str       # 摘要
    authors: List[str]  # 作者列表
    source: str         # 来源标识："arxiv" 或 "openalex"
    
    # 额外信息（有就填，没有就 None）
    citations_count: Optional[int]  # 引用次数（OpenAlex 提供）
    venue: Optional[str]            # 发表的期刊/会议
    ...
```

**`get_unique_id()` 方法的作用**

这是去重的核心！不同论文用不同的方式生成唯一ID：

```
优先级：DOI > arXiv ID > OpenAlex ID > 标题哈希

例如：
- 有 DOI 的论文：  "doi:10.48550/arXiv.1706.03762"
- 只有 arXiv ID：  "arxiv:1706.03762"
- 只有 OpenAlex： "openalex:W2741809807"
- 都没有：         "title:a3f9b12c8d4e"（标题的MD5哈希）
```

**为什么这样设计？**

因为同一篇论文可能同时出现在 arXiv 和 OpenAlex 中，如果没有统一的去重键，就会把同一篇论文存两次。

---

**`PaperSource` 抽象类的作用**

这是一个"接口规范"——规定所有数据源都必须实现哪些方法：

```python
class PaperSource(ABC):
    def search(self, query, max_results) -> List[Paper]:
        """必须实现：搜索论文，返回统一的 Paper 列表"""
        
    def get_source_name(self) -> str:
        """必须实现：返回数据源名称"""
```

好处是：未来添加新数据源（比如 Semantic Scholar），只需要写一个新的类实现这两个方法，其他代码不用改。

---

### 3.2 `src/core/sources/arxiv_adapter.py` — arXiv 适配器

**这个文件解决了什么问题？**

项目原来有 `src/core/arxiv/client.py`，可以从 arXiv 抓论文。但它返回的是 `ArxivPaper` 对象，不是新的统一 `Paper` 模型。

这个适配器的作用就是"翻译"——把 arXiv 原来的格式转成统一的 `Paper` 格式。

**关键逻辑：arXiv ID 的提取**

arXiv 返回的 `entry_id` 是一个完整 URL，比如：
```
http://arxiv.org/abs/1706.03762v5
```

需要从中提取出纯粹的 arXiv ID：
```
1706.03762
```

代码处理方式：
1. 找到 `arxiv.org/abs/` 后面的部分
2. 去掉版本号（`v5` 这种后缀）

---

### 3.3 `src/core/openalex/client.py` — OpenAlex 客户端

**OpenAlex 是什么？**

OpenAlex 是一个完全免费、无需注册的学术论文数据库，特点：
- 覆盖 2.5 亿+ 篇学术文献（远超 arXiv）
- 提供引用次数、发表期刊等丰富信息
- 无速率限制，可以快速抓取大量数据
- API 文档：https://docs.openalex.org/

**最关键的技术点：摘要重建**

OpenAlex 的摘要不是直接存储的文本，而是用"**倒排索引**"格式：

```json
{
    "The": [0, 15, 30],
    "dominant": [1],
    "sequence": [2],
    "transduction": [3],
    "models": [4, 18],
    ...
}
```

意思是：单词 "The" 出现在第0、15、30个位置；"dominant" 出现在第1个位置，以此类推。

**重建算法**（`_reconstruct_abstract` 方法）：

```python
# 第一步：把字典翻转，变成"位置 -> 单词"
position_to_word = {0: "The", 1: "dominant", 2: "sequence", ...}

# 第二步：按位置排序
sorted_positions = [0, 1, 2, 3, 4, ...]

# 第三步：按顺序拼接
abstract = "The dominant sequence transduction models ..."
```

**为什么 OpenAlex 用这种格式存摘要？**

这是搜索引擎常用的技术，倒排索引的优势是检索很快——给定一个词，可以立即找到它在哪些位置出现。但对我们来说，需要先转换回普通文本。

---

### 3.4 `src/core/sources/aggregator.py` — 多源聚合器

**这是 Stage 2 最核心的模块！**

**功能总览**：

```
用户说："我要 200 篇论文"
    ↓
聚合器：好的，我来两个数据源各请求 300 篇（1.5倍过采样）
    ↓
同时（并行）向 arXiv 和 OpenAlex 发请求
    arXiv  → 返回 300 篇
    OpenAlex → 返回 300 篇
    ↓
合计 600 篇，但其中有重复的
    ↓
去重（三个步骤）
    ↓
剩下约 580 篇（重复率约 3%）
    ↓
取前 200 篇返回给用户
```

**并行抓取：ThreadPoolExecutor**

传统方式是顺序请求（先等 arXiv 完成，再请求 OpenAlex）：
```
等 arXiv 30秒 → 等 OpenAlex 30秒 = 总共 60秒
```

并行方式（同时发请求）：
```
arXiv 和 OpenAlex 同时请求 → 总共约 30秒
```

代码实现：
```python
with ThreadPoolExecutor(max_workers=2) as executor:
    # 同时提交两个任务
    future_arxiv   = executor.submit(arxiv_source.search, query, 300)
    future_openalex = executor.submit(openalex_source.search, query, 300)
    
    # 等两个都完成
    arxiv_papers   = future_arxiv.result()
    openalex_papers = future_openalex.result()
```

**三级去重策略**

去重按优先级从高到低进行：

```
第一级：DOI 去重（最可靠）
  → 同一篇论文在不同数据库里 DOI 是一样的
  → 如果 DOI 相同，直接认为是同一篇

第二级：arXiv ID 去重
  → arXiv 上的论文有唯一的 arXiv ID
  → 用于处理 OpenAlex 里也收录了 arXiv 论文的情况

第三级：标题归一化去重（兜底方案）
  → 把标题统一处理：转小写、去标点、去常用词
  → "Attention Is All You Need!" == "attention all you need"
  → 如果处理后标题相同，认为是同一篇
```

**为什么要过采样（请求 1.5 倍数量）？**

因为去重会损失一部分论文：
```
目标 200 篇
每源请求 200 × 1.5 = 300 篇
两源合计 600 篇
去重后约 580 篇
取前 200 篇 ✓
```

如果不过采样，去重后可能只剩 190 篇，达不到目标数量。

---

### 3.5 `src/core/storage/chroma_store.py` — ChromaDB 存储

**ChromaDB 是什么？**

ChromaDB 是一个专门用来存储向量（embeddings）的数据库，可以把数据存到硬盘上。

**为什么要用 ChromaDB 替换 FAISS？**

| 对比 | FAISS（旧） | ChromaDB（新） |
|------|------------|----------------|
| 数据保存方式 | 存在内存，程序关闭就丢失 | 存在硬盘，永久保留 |
| 增量添加 | 不支持，要重建整个索引 | 支持，随时添加新论文 |
| 元数据（标题、作者等） | 不存储，需要额外维护 | 内置存储 |
| 去重 | 需要自己实现 | 基于 ID 自动去重 |

简单说：**ChromaDB 就像一个带有搜索功能的数据库，而 FAISS 只是一个搜索引擎，没有存储能力。**

**主要方法说明**

```python
# 1. 写入论文（自动去重，已存在的会跳过）
store.upsert_papers(papers, embeddings)

# 2. 检查某篇论文是否已经存储
if store.is_cached(paper.get_unique_id()):
    print("已存在，跳过向量化")

# 3. 语义搜索（用向量找相似论文）
results = store.search(query_embedding, top_k=10)

# 4. 获取统计信息
stats = store.get_stats()
# 返回：{"total_papers": 500, "source_distribution": {"arxiv": 300, "openalex": 200}}
```

**数据存储位置**

ChromaDB 的数据存在 `data/chroma_db/` 目录下：

```
data/chroma_db/
└── papers/                  ← 论文集合
    ├── chroma.sqlite3       ← 元数据数据库（标题、作者等）
    └── <uuid>/              ← 向量索引文件（HNSW算法）
        ├── data_level0.bin
        ├── header.bin
        ├── length.bin
        └── link_lists.bin
```

---

### 3.6 `src/scripts/fetch_papers.py` — 多源抓取命令行工具

这个脚本把整个多源抓取流程封装成一个命令行工具，方便直接在终端使用。

**支持的参数**

```bash
python src/scripts/fetch_papers.py \
  --sources arxiv openalex \   # 选择数据源（可以只选一个）
  --query "transformer" \      # 搜索关键词
  --max-results 200 \          # 目标论文数量
  --output data/papers.csv \   # 输出文件路径
  --generate-embeddings        # 是否同时生成向量并存到 ChromaDB
```

**脚本执行流程**

```
1. 解析命令行参数
2. 初始化选定的数据源
3. 创建多源聚合器，执行并行抓取
4. 去重，截断到目标数量
5. 导出到 CSV 文件
6. （可选）生成向量，写入 ChromaDB
```

---

### 3.7 `src/test_stage2_integration.py` — 集成测试

这个文件用来验证整个 Stage 2 流程是否正常工作。

**测试内容**

```
测试1: 多源抓取 + 去重
  - 从 arXiv 和 OpenAlex 各抓取论文
  - 验证去重后无重复

测试2: ChromaDB 存储
  - 把论文写入 ChromaDB
  - 验证能正确存储和读取

测试3: 层级主题聚类
  - 训练 BERTopic 模型
  - 验证能生成层级主题结构
  - 验证能导出 JSON 格式
```

**运行方式**

```bash
conda activate literature_review
python src/test_stage2_integration.py
```

---

## 4. 修改的原有文件说明

### 4.1 `src/core/config.py`

**修改内容**：新增第13行和第21行

```python
# 新增这一行（第13行）
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# 新增这一行（第21行，确保目录存在）
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
```

**为什么要改**：ChromaDB 需要知道把数据存在哪里，所以统一在配置文件里定义路径。

---

### 4.2 `src/core/nlp/topic_modeling.py`

**修改内容1**：新增 `get_hierarchical_topics()` 方法（第253-275行）

```python
def get_hierarchical_topics(self, documents: List[str]) -> pd.DataFrame:
    """
    获取层级主题结构（BERTopic 原生支持）
    
    BERTopic 内部会使用层次聚类算法，把相似的主题合并成树状结构
    比如：
      根节点
      ├── AI技术类（主题0、1、2 的父节点）
      │   ├── 主题0: attention_transformer_vision
      │   └── 主题1: language_models_bert
      └── 应用类（主题3、4 的父节点）
          ├── 主题3: medical_diagnosis
          └── 主题4: autonomous_driving
    """
    return self.topic_model.hierarchical_topics(documents)
```

**修改内容2**：新增 `export_hierarchy_json()` 方法（第277-332行）

这个方法把层级结构转成前端可以直接用的 JSON 格式：

```json
{
  "name": "All Topics",
  "children": [
    {
      "name": "Topic 0: attention_transformer",
      "value": 25,
      "topic_id": 0
    },
    {
      "name": "Topic 1: language_bert_gpt",
      "value": 18,
      "topic_id": 1
    }
  ]
}
```

前端可以用 Plotly 的 Sunburst 图或 Treemap 直接渲染这个数据。

**修改内容3（Bug 修复）**：`fit()` 方法的返回值问题

**原来的代码（有Bug）**：

```python
def fit(self, documents, embeddings):
    topics, probs = self.topic_model.fit_transform(documents, embeddings)
    
    if self.verbose:
        print("训练完成!")
        print(f"主题数: {n_topics}")
        return topics, probs   # ← 问题在这里！return 在 if 块里
    # 如果 verbose=False，这里没有 return，函数返回 None！
```

**修复后的代码**：

```python
def fit(self, documents, embeddings):
    topics, probs = self.topic_model.fit_transform(documents, embeddings)
    
    if self.verbose:
        print("训练完成!")
        print(f"主题数: {n_topics}")
    
    return topics, probs   # ← 正确：无论 verbose 是什么，都会返回
```

**这个 Bug 的影响**：当调用 `topic_modeler.fit(documents, embeddings)` 时，如果 `verbose=True`（打印日志），函数正常返回；但如果 `verbose=False`，函数会返回 `None`，导致 `topics, probs = modeler.fit(...)` 报错。

---

### 4.3 `requirements.txt`

**修改内容**：

```diff
- chromadb==0.4.22
+ chromadb>=1.5.7
+ httpx>=0.27.0
+ pyarrow>=14.0.0
```

**为什么升级 chromadb？**

原来的版本 0.4.22 与 numpy 2.x 不兼容（会报 `AttributeError: np.float_ was removed`），必须升级到支持 numpy 2.x 的 1.5.7 版本。

**为什么新增 httpx？**

OpenAlex 客户端使用 `httpx` 库发送 HTTP 请求（相比 `requests`，`httpx` 支持异步，性能更好）。

**为什么新增 pyarrow？**

ChromaDB 内部使用 pyarrow 做数据序列化，是 chromadb 的依赖。

---

## 5. 核心设计思路（适合小白）

### 5.1 为什么要定义"接口"（抽象类）？

想象你在餐厅点餐，服务员不需要知道厨师怎么做菜，只需要把订单传给厨房，然后把菜端出来。`PaperSource` 就像这个"接口"：

```
你（调用方） → 说"我要搜 'AI' 相关的论文"
                    ↓
             PaperSource 接口（统一格式）
            /                          \
    ArxivSource                   OpenAlexSource
  （去 arXiv 找）               （去 OpenAlex 找）
```

这样的好处是：将来想加新数据源（比如 Google Scholar），只需要再写一个实现了 `PaperSource` 接口的类，其他代码完全不用动。

---

### 5.2 并行 vs 顺序：形象理解

**顺序方式**（原来）：

```
你去图书馆A借书，等借完 → 再去图书馆B借书 → 回家
总时间：30分钟 + 30分钟 = 60分钟
```

**并行方式**（现在）：

```
你让朋友去图书馆A借书
自己去图书馆B借书
两人同时出发，同时回到约定地点汇合
总时间：max(30分钟, 30分钟) = 30分钟
```

这就是 `ThreadPoolExecutor` 做的事情——同时派出多个"线程"去不同数据源抓数据。

---

### 5.3 去重：为什么需要三级？

打个比方：你收集了两份名单（arXiv 和 OpenAlex），需要合并去重：

```
第一步：看身份证号（DOI）
  → 最准确，但不是每篇论文都有 DOI

第二步：看 arXiv 编号
  → arXiv 上的论文都有，但 OpenAlex 里非 arXiv 的论文没有

第三步：比较名字（标题）
  → 兜底方案，处理前两步都无法匹配的情况
  → 要先"归一化"（去掉大小写差异、标点差异）再比较
```

---

### 5.4 ChromaDB：向量数据库是什么？

**普通数据库**（如 MySQL）：

```sql
SELECT * FROM papers WHERE title = '关注力机制'
```
→ 只能精确匹配

**向量数据库**（ChromaDB）：

```python
results = store.search(query_embedding)  # "人工智能"的向量
```
→ 能找到"机器学习"、"深度学习"等**相关**论文（语义相似）

ChromaDB 把每篇论文的摘要转成一个向量（几百个数字的数组），然后存起来。搜索时，把查询词也转成向量，找最相近的那些向量对应的论文。

---

## 6. 如何使用这些新功能

### 6.1 多源抓取（命令行方式）

```bash
# 激活环境
conda activate literature_review

# 进入项目目录
cd e:/AIassistant_v2

# 基础用法：只抓取 CSV
python src/scripts/fetch_papers.py \
  --sources arxiv openalex \
  --query "transformer attention" \
  --max-results 100

# 高级用法：同时生成向量存入 ChromaDB
python src/scripts/fetch_papers.py \
  --sources arxiv openalex \
  --query "large language models" \
  --max-results 200 \
  --generate-embeddings \
  --output data/papers.csv
```

### 6.2 在 Python 代码中使用多源抓取

```python
from core.sources.arxiv_adapter import ArxivSource
from core.openalex.client import OpenAlexSource
from core.sources.aggregator import MultiSourceAggregator

# 1. 初始化数据源
arxiv_source = ArxivSource()
openalex_source = OpenAlexSource()

# 2. 创建聚合器
aggregator = MultiSourceAggregator(
    sources=[arxiv_source, openalex_source],
    max_workers=2
)

# 3. 抓取论文（自动并行+去重）
papers = aggregator.search_all(
    query="deep learning",
    max_results=200,
    oversample_ratio=1.5  # 过采样倍数
)

print(f"获取到 {len(papers)} 篇论文")
for paper in papers[:3]:
    print(f"- [{paper.source}] {paper.title}")
```

### 6.3 使用 ChromaDB 缓存

```python
from core.storage.chroma_store import ChromaStore
from core.nlp.embeddings import EmbeddingGenerator

# 1. 初始化存储
store = ChromaStore(persist_directory="data/chroma_db/papers")

# 2. 生成向量
embedder = EmbeddingGenerator()
embeddings = embedder.encode_papers(papers)

# 3. 写入（自动去重，重复的会跳过）
store.upsert_papers(papers, embeddings)

# 4. 下次运行时检查是否已缓存
for paper in papers:
    if store.is_cached(paper.get_unique_id()):
        print(f"已缓存: {paper.title[:50]}")

# 5. 语义搜索
query_embed = embedder.encode_texts(["自注意力机制"])[0]
results = store.search(query_embed, top_k=5)
for r in results:
    print(f"相似度: {r['distance']:.3f} | {r['metadata']['title'][:50]}")
```

### 6.4 训练层级主题模型

```python
from core.nlp.topic_modeling import TopicModeler
import pandas as pd

# 1. 准备文档
df = pd.read_csv("data/papers.csv")
documents = df["abstract"].tolist()

# 2. 训练模型
modeler = TopicModeler(min_topic_size=10)
topics, probs = modeler.fit(documents, embeddings)

# 3. 获取普通主题信息
topic_info = modeler.get_topic_info()
print(topic_info[["Topic", "Count", "Name"]].head())

# 4. 获取层级结构（新增功能）
hierarchical = modeler.get_hierarchical_topics(documents)
print(hierarchical.head())

# 5. 导出 JSON（前端直接可用）
import json
hierarchy_json = modeler.export_hierarchy_json(documents)
with open("data/hierarchy.json", "w", encoding="utf-8") as f:
    json.dump(hierarchy_json, f, ensure_ascii=False, indent=2)
print("层级主题已导出到 data/hierarchy.json")
```

---

## 7. 测试结果

### 7.1 集成测试输出（实际运行结果）

```
测试1: 多源抓取 + 去重
  [OK] arxiv          :  75 篇
  [OK] openalex       :  75 篇
  [去重前] 总计: 150 篇
  [去重后] 剩余: 145 篇（重复率: 3.3%）
  [最终结果] 返回: 50 篇论文
  [OK] 测试通过: 抓取 50 篇论文，无重复

测试2: ChromaDB 存储
  [ChromaStore] 初始化完成
    持久化目录: data/chroma_db/test
    集合名称: test_papers
    已存储论文数: 0
  [ChromaStore] 新增 50 篇论文到缓存
  [OK] 测试通过: ChromaDB 存储 50 篇论文

测试3: 层级主题聚类
  [OK] 层级主题节点数: 1
  [OK] JSON 导出成功，根节点: All Topics
  [OK] 子主题数: 2
  主题统计:
    主题数量: 3
    前3个主题:
      Topic 0: attention_transformer_vision
      Topic 1: attention_language_models

[OK] 所有测试通过！
```

### 7.2 性能参考数据

| 操作 | 耗时 | 说明 |
|------|------|------|
| 多源抓取 150 篇 | 30-60秒 | 网络速度影响较大 |
| 向量化 50 篇 | 10-20秒 | 首次需下载模型(约400MB) |
| ChromaDB 写入 50 篇 | <3秒 | 本地操作，很快 |
| BERTopic 训练 50 篇 | 15-30秒 | 依赖数据量和主题数 |

---

## 8. 常见问题与解决方法

### Q1：OpenAlex 抓取失败，提示 "HTTP 错误"

**原因**：网络无法访问 `api.openalex.org`

**解决方法**：
```bash
# 如果无法连接 OpenAlex，只使用 arXiv
python src/scripts/fetch_papers.py --sources arxiv --query "AI"
```

---

### Q2：BERTopic 报错 `max_df corresponds to < documents than min_df`

**原因**：训练文档数量太少（需要至少 30-50 篇）

**解决方法**：
```python
# 增加论文数量（改成 100 篇或更多）
papers = aggregator.search_all(query, max_results=100)
```

---

### Q3：ChromaDB 报错相关

**清理方法**：
```python
# 方法1：调用 reset() 清空集合
store = ChromaStore(persist_directory="data/chroma_db/test")
store.reset()

# 方法2：删除目录（彻底清除）
import shutil
shutil.rmtree("data/chroma_db/test")
```

---

### Q4：`conda activate` 在 PowerShell 里没有效果

**解决方法**：
```powershell
# 第一步（仅首次执行，需要管理员权限）
conda init powershell

# 第二步：关闭并重开终端

# 第三步：激活环境
conda activate literature_review

# 验证：查看 Python 路径（应该包含 literature_review）
python -c "import sys; print(sys.executable)"
```

---

### Q5：`np.float_` 相关错误（NumPy 版本问题）

**原因**：chromadb 旧版本（0.4.22）不兼容 numpy 2.x

**解决方法**：
```bash
conda activate literature_review
pip install "chromadb>=1.5.7"
```

---

## 9. 环境配置说明

### 9.1 所需依赖版本

| 库 | 版本要求 | 用途 |
|----|---------|------|
| chromadb | >= 1.5.7 | 向量持久化存储 |
| httpx | >= 0.27.0 | OpenAlex HTTP 客户端 |
| pyarrow | >= 14.0.0 | ChromaDB 内部依赖 |
| bertopic | == 0.16.0 | 主题建模 |
| sentence-transformers | >= 2.3.1 | 文本向量化 |

### 9.2 安装命令

```bash
# 激活 conda 环境
conda activate literature_review

# 安装所有依赖
pip install -r requirements.txt

# 或单独安装 Stage 2 新增依赖
pip install "chromadb>=1.5.7" "httpx>=0.27.0" "pyarrow>=14.0.0"
```

### 9.3 验证安装

```bash
conda activate literature_review
python -c "
import chromadb, httpx, pyarrow
print('chromadb:', chromadb.__version__)
print('httpx:', httpx.__version__)
print('pyarrow:', pyarrow.__version__)
print('所有依赖安装正常!')
"
```

---

## 附录：数据流程图

```
用户输入关键词
      |
      v
+-----------------------------+
|    MultiSourceAggregator    |
|  +-----------+ +---------+  |
|  |ArxivSource| |OpenAlex |  |  <- 并行抓取
|  +-----------+ |Source   |  |
|       |        +---------+  |
|       +--------+            |
|           v                 |
|       三级去重               |
|           |                 |
+-----------|------------------+
            |  统一 Paper 列表
            v
   +-------------------+
   | EmbeddingGenerator|  <- 生成向量
   +--------+----------+
            |  embeddings
            v
   +-------------------+
   |   ChromaStore     |  <- 持久化存储
   | (data/chroma_db/) |
   +--------+----------+
            |  documents
            v
   +-------------------+
   |   TopicModeler    |  <- 主题建模
   |   (BERTopic)      |
   +--------+----------+
            |  hierarchy_json
            v
        前端展示
```

---

*本文档由 AI Assistant 生成，最后更新：2026-04-16*