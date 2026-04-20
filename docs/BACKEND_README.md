# 后端 API 文档


---

## 目录

1. [项目概述](#1-项目概述)
2. [快速开始](#2-快速开始)
3. [API 端点总览](#3-api-端点总览)
4. [详细接口文档](#4-详细接口文档)
5. [数据模型](#5-数据模型)

---

## 1. 项目概述

### 1.1 功能简介


-  **论文管理**：获取、搜索、查询论文
-  **主题管理**：获取主题列表、主题详情、主题下的论文
-  **统计分析**：论文数量、日期范围等统计信息
-  **RAG 对话**（待实现）：基于论文的智能问答

### 1.2 技术架构

```
FastAPI 应用
    ↓
核心服务层 (PaperService)
    ↓
本地数据 (CSV 文件)
```

**特点：**
-  所有数据预先下载到本地（本地预下载模式）
-  快速响应（无需联网查询）
-  类型安全（Pydantic 数据验证）
-  自动生成 API 文档（Swagger UI）

---

## 2. 快速开始

### 2.1 环境要求

- Python 3.10+
- Anaconda
- Git（用于克隆项目）
- 依赖包见 `requirements.txt`

### 2.2 从 GitHub 克隆项目

**步骤 1：克隆仓库**

```bash
# 克隆项目到本地
git clone https://github.com/xuwj23-art/AI_Assistant.git

# 进入项目目录
cd AI_Assistant
```

**步骤 2：查看项目结构**

```
AI_Assistant/
├── src/                  # 源代码
│   ├── core/            # 核心模块
│   ├── scripts/         # 脚本工具
│   └── main.py          # FastAPI 主程序
├── data/                # 数据目录
│   └── raw/            # 原始论文数据
├── models/              # 模型目录
├── tests/               # 测试代码
└── requirements.txt     # 依赖列表
```

### 2.3 安装步骤

**方式 1：使用 Conda（推荐）**

```bash
# 1. 创建虚拟环境
conda create -n literature_review python=3.10 -y

# 2. 激活环境
conda activate literature_review

# 3. 安装依赖
pip install -r requirements.txt
```

**方式 2：使用 pip + venv**

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活环境（Windows）
venv\Scripts\activate
# 或（Linux/Mac）
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

**验证安装：**

```bash
# 检查关键包是否安装成功
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
```

### 2.4 下载论文数据（首次使用）

在启动服务前，需要先下载一些论文数据：

```bash
# 进入 src 目录
cd src

# 下载论文（示例：下载 Transformer 相关论文）
python -m scripts.fetch_and_save_arxiv --query "Transformer" --max-results 100

# 下载更多主题的论文（可选）
python -m scripts.fetch_and_save_arxiv --query "machine learning" --max-results 50
python -m scripts.fetch_and_save_arxiv --query "deep learning" --max-results 50
```

**参数说明：**
- `--query`: 搜索关键词
- `--max-results`: 下载数量（建议 50-500）
- `--output-dir`: 输出目录（默认 `data/raw`）

**下载信息：**
```
============================================================
arXiv 论文抓取工具
============================================================
关键词：Transformer
最大数量：100
...
[success] 已经成功抓取100篇论文
[success] 已经保存到：data/raw/arxiv_Transformer.csv
```

### 2.5 启动后端服务

```bash
# 确保在 src 目录下
cd src

# 启动服务（开发模式）
uvicorn main:app --reload
```

**启动成功后，会看到：**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**说明：**
- 服务默认运行在 `http://127.0.0.1:8000`
- `--reload` 参数启用热重载（代码修改后自动重启）
- 生产环境请去掉 `--reload` 参数

**访问 API 文档：**
-  **Swagger UI**：http://localhost:8000/docs （推荐，可交互测试）
-  **ReDoc**：http://localhost:8000/redoc （更美观的文档）
-  **API 根路径**：http://localhost:8000/

### 2.6 快速测试

**测试方式 1：使用浏览器**

直接在浏览器中访问：
- http://localhost:8000/health
- http://localhost:8000/docs（可视化测试所有接口）

**测试方式 2：使用 PowerShell（Windows）**

```powershell
# 测试 1：健康检查
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET

# 预期响应：
# status
# ------
# healthy

# 测试 2：获取统计信息
Invoke-RestMethod -Uri "http://localhost:8000/api/stats" -Method GET

# 预期响应：
# total_papers date_range
# ------------ ----------
# 500          @{earliest=2026-01-18T20:25:37+00:00; latest=2026-01-26T18:59:48+00:00}

# 测试 3：搜索论文
Invoke-RestMethod -Uri "http://localhost:8000/api/papers/search?query=transformer&max_results=3" -Method GET

# 测试 4：获取主题列表（需要先运行主题建模）
Invoke-RestMethod -Uri "http://localhost:8000/api/topics" -Method GET | ConvertTo-Json -Depth 3

# 测试 5：获取主题详情
Invoke-RestMethod -Uri "http://localhost:8000/api/topics/0" -Method GET | ConvertTo-Json -Depth 3

# 测试 6：获取主题下的论文
Invoke-RestMethod -Uri "http://localhost:8000/api/topics/0/papers?limit=5" -Method GET
```

**测试方式 3：使用 curl（Linux/Mac/Git Bash）**

```bash
# 测试 1：健康检查
curl http://localhost:8000/health

# 预期响应：
# {"status":"healthy"}

# 测试 2：获取根路径信息
curl http://localhost:8000/

# 预期响应：
# {
#   "name": "论文助手 API",
#   "version": "1.0.0",
#   "docs": "/docs",
#   "endpoints": {...}
# }

# 测试 3：获取统计信息
curl http://localhost:8000/api/stats

# 预期响应：
# {
#   "total_papers": 500,
#   "date_range": {
#     "earliest": "2026-01-18T20:25:37+00:00",
#     "latest": "2026-01-26T18:59:48+00:00"
#   }
# }

# 测试 4：搜索论文
curl "http://localhost:8000/api/papers/search?query=transformer&max_results=3"

# 测试 5：获取主题列表
curl http://localhost:8000/api/topics | jq .

# 测试 6：获取主题详情
curl http://localhost:8000/api/topics/0 | jq .

# 测试 7：获取主题下的论文
curl "http://localhost:8000/api/topics/0/papers?limit=5" | jq .
```

**注意：** 主题管理接口（测试 4-7）需要先运行主题建模脚本生成主题数据：

```bash
# 生成嵌入向量
cd src
python -m scripts.generate_embeddings --input ../data/raw/arxiv_Transformer.csv --output ../data/processed/embeddings.npy

# 训练主题模型
python -m scripts.train_topics --csv ../data/raw/arxiv_Transformer.csv --embeddings ../data/processed/embeddings.npy --output ../models/bertopic_model
```

## 3. API 端点总览

### 3.1 系统端点

| 端点 | 方法 | 功能 | 状态 |
|------|------|------|------|
| `/` | GET | 欢迎页面 | 已实现 |
| `/health` | GET | 健康检查 |  已实现 |

### 3.2 论文管理端点

| 端点 | 方法 | 功能 | 状态 |
|------|------|------|------|
| `/api/papers` | GET | 获取论文列表（分页） |  已实现 |
| `/api/papers/search` | GET | 搜索论文 |  已实现 |
| `/api/papers/{id}` | GET | 获取单篇论文详情 |  已实现 |
| `/api/stats` | GET | 获取统计信息 |  已实现 |

### 3.3 主题管理端点

| 端点 | 方法 | 功能 | 状态 |
|------|------|------|------|
| `/api/topics` | GET | 获取主题列表 | ✅ 已实现 |
| `/api/topics/{id}` | GET | 获取主题详情 | ✅ 已实现 |
| `/api/topics/{id}/papers` | GET | 获取主题下的论文 | ✅ 已实现 |

### 3.4 RAG 对话端点（待实现）

| 端点 | 方法 | 功能 | 状态 |
|------|------|------|------|
| `/api/chat/init` | POST | 初始化对话 |  待实现 |
| `/api/chat/message` | POST | 发送消息 |  待实现 |
| `/api/chat/history` | GET | 获取对话历史 |  待实现 |

---

## 4. 详细接口文档

### 4.1 系统端点

#### 4.1.1 GET `/` - 欢迎页面

**功能：** 返回 API 基本信息

**请求参数：** 无

**响应示例：**

```json
{
  "name": "论文助手 API",
  "version": "1.0.0",
  "docs": "/docs",
  "endpoints": {
    "papers": "/api/papers",
    "search": "/api/papers/search",
    "stats": "/api/stats",
    "topics": "/api/topics"
  }
}
```

---

#### 4.1.2 GET `/health` - 健康检查

**功能：** 检查服务状态

**请求参数：** 无

**响应示例：**

```json
{
  "status": "healthy"
}
```

**使用场景：**
- 监控服务是否正常运行
- 容器健康检查
- 负载均衡器健康检查

---

### 4.2 论文管理端点

#### 4.2.1 GET `/api/papers` - 获取论文列表

**功能：** 获取所有论文，支持分页

**请求参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `limit` | int | 否 | 10 | 返回数量（1-100） |

**请求示例：**

```bash
# 获取 10 篇论文
curl "http://localhost:8000/api/papers?limit=10"

# 获取 50 篇论文
curl "http://localhost:8000/api/papers?limit=50"
```

**响应示例：**

```json
{
  "total": 10,
  "papers": [
    {
      "id": "2301.00001",
      "title": "BERT: Pre-training of Deep Bidirectional Transformers",
      "abstract": "We introduce a new language representation model called BERT...",
      "authors": ["Alice Smith", "Bob Johnson"],
      "published": "2023-01-01T00:00:00",
      "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
      "categories": ["cs.CL", "cs.AI"]
    },
    ...
  ]
}
```

**错误响应：**

| 状态码 | 说明 |
|--------|------|
| 400 | 参数错误（如 limit > 100） |
| 500 | 服务器内部错误 |

---

#### 4.2.2 GET `/api/papers/search` - 搜索论文

**功能：** 根据关键词搜索论文

**请求参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | str | 是 | - | 搜索关键词 |
| `max_results` | int | 否 | 10 | 最多返回数量（1-100） |
| `fields` | str | 否 | "title,abstract" | 搜索字段（逗号分隔） |

**可用的搜索字段：**
- `title` - 标题
- `abstract` - 摘要
- `authors` - 作者

**请求示例：**

```bash
# 在标题和摘要中搜索 "BERT"
curl "http://localhost:8000/api/papers/search?query=BERT&max_results=10"

# 只在标题中搜索
curl "http://localhost:8000/api/papers/search?query=transformer&fields=title"

# 在标题、摘要、作者中搜索
curl "http://localhost:8000/api/papers/search?query=Alice&fields=title,abstract,authors"
```

**响应示例：**

```json
{
  "total": 5,
  "papers": [
    {
      "id": "2301.00001",
      "title": "BERT: Pre-training of Deep Bidirectional Transformers",
      "abstract": "We introduce BERT, a new language model...",
      "authors": ["Alice Smith", "Bob Johnson"],
      "published": "2023-01-01T00:00:00",
      "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
      "categories": ["cs.CL"]
    },
    ...
  ]
}
```

**搜索逻辑：**
- 不区分大小写
- 支持部分匹配（如搜索 "bert" 可以匹配 "BERT" 或 "BERTopic"）
- 在指定字段中查找包含关键词的论文

**错误响应：**

| 状态码 | 说明 |
|--------|------|
| 400 | 参数错误（query 为空或 max_results 超出范围） |
| 500 | 服务器内部错误 |

---

#### 4.2.3 GET `/api/papers/{paper_id}` - 获取单篇论文详情

**功能：** 根据 ID 获取论文详细信息

**路径参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `paper_id` | str | 是 | 论文的 arXiv ID |

**请求示例：**

```bash
# 获取 ID 为 2301.00001 的论文
curl "http://localhost:8000/api/papers/2301.00001"
```

**响应示例：**

```json
{
  "id": "2301.00001",
  "title": "BERT: Pre-training of Deep Bidirectional Transformers",
  "abstract": "We introduce a new language representation model called BERT...",
  "authors": ["Alice Smith", "Bob Johnson", "Charlie Brown"],
  "published": "2023-01-01T00:00:00",
  "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
  "categories": ["cs.CL", "cs.AI"]
}
```

**错误响应：**

| 状态码 | 说明 | 响应示例 |
|--------|------|---------|
| 404 | 论文不存在 | `{"detail": "Paper not found: 2301.00001"}` |
| 500 | 服务器内部错误 | `{"detail": "Internal server error"}` |

---

#### 4.2.4 GET `/api/stats` - 获取统计信息

**功能：** 获取论文库的统计数据

**请求参数：** 无

**请求示例：**

```bash
curl "http://localhost:8000/api/stats"
```

**响应示例：**

```json
{
  "total_papers": 1000,
  "date_range": {
    "earliest": "2020-01-01",
    "latest": "2024-12-31"
  },
  "categories": {
    "cs.CL": 450,
    "cs.AI": 350,
    "cs.LG": 200
  }
}
```

**响应字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_papers` | int | 论文总数 |
| `date_range.earliest` | str | 最早论文发布日期 |
| `date_range.latest` | str | 最新论文发布日期 |
| `categories` | dict | 各分类的论文数量 |

---

### 4.3 主题管理端点

#### 4.3.1 GET `/api/topics` - 获取主题列表

**功能：** 获取所有主题及其统计信息

**状态：** ✅ 已实现

**请求参数：** 无

**请求示例：**

```bash
# 获取所有主题
curl "http://localhost:8000/api/topics"
```

**响应示例：**

```json
{
  "total": 5,
  "topics": [
    {
      "topic_id": 0,
      "topic_name": "transformer_attention_mechanism",
      "keywords": [
        {"word": "transformer", "score": 0.85},
        {"word": "attention", "score": 0.72},
        {"word": "mechanism", "score": 0.68}
      ],
      "paper_count": 109
    },
    {
      "topic_id": 1,
      "topic_name": "neural_network_architecture",
      "keywords": [
        {"word": "neural", "score": 0.82},
        {"word": "network", "score": 0.75}
      ],
      "paper_count": 95
    }
    // ... 更多主题
  ]
}
```

**响应字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `total` | int | 主题总数 |
| `topics` | array | 主题列表 |
| `topics[].topic_id` | int | 主题ID |
| `topics[].topic_name` | str | 主题名称 |
| `topics[].keywords` | array | 关键词列表（按分数降序） |
| `topics[].keywords[].word` | str | 关键词 |
| `topics[].keywords[].score` | float | 关键词得分（0-1） |
| `topics[].paper_count` | int | 该主题下的论文数量 |

**测试方法：**

<details>
<summary>点击展开测试步骤</summary>

**方式 1：使用浏览器**
```
直接访问：http://localhost:8000/api/topics
```

**方式 2：使用 PowerShell（Windows）**
```powershell
# 获取主题列表
Invoke-RestMethod -Uri "http://localhost:8000/api/topics" -Method GET | ConvertTo-Json -Depth 3

# 查看主题总数
$result = Invoke-RestMethod -Uri "http://localhost:8000/api/topics" -Method GET
Write-Host "主题总数: $($result.total)"

# 查看第一个主题的信息
$result.topics[0] | Format-List
```

**方式 3：使用 curl（Linux/Mac/Git Bash）**
```bash
# 获取主题列表
curl http://localhost:8000/api/topics | jq .

# 只显示主题数量
curl -s http://localhost:8000/api/topics | jq '.total'

# 显示所有主题名称
curl -s http://localhost:8000/api/topics | jq '.topics[].topic_name'
```

**方式 4：使用 Swagger UI**
```
1. 访问 http://localhost:8000/docs
2. 找到 "主题管理" 部分
3. 点击 GET /api/topics
4. 点击 "Try it out"
5. 点击 "Execute"
6. 查看响应结果
```

**预期结果：**
- 返回状态码：200
- 返回主题列表，按论文数量降序排列
- 每个主题包含 ID、名称、关键词和论文数量

</details>

---

#### 4.3.2 GET `/api/topics/{topic_id}` - 获取主题详情

**功能：** 获取单个主题的详细信息

**状态：** ✅ 已实现

**路径参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `topic_id` | int | 是 | 主题 ID |

**请求示例：**

```bash
# 获取 ID 为 0 的主题
curl "http://localhost:8000/api/topics/0"
```

**响应示例：**

```json
{
  "topic_id": 0,
  "topic_name": "transformer_attention_mechanism",
  "keywords": [
    {"word": "transformer", "score": 0.85},
    {"word": "attention", "score": 0.72},
    {"word": "mechanism", "score": 0.68},
    {"word": "model", "score": 0.65},
    {"word": "language", "score": 0.58}
  ],
  "paper_count": 109
}
```

**错误响应：**

| 状态码 | 说明 | 响应示例 |
|--------|------|---------|
| 404 | 主题不存在 | `{"detail": "Topic not found: 999"}` |
| 500 | 服务器内部错误 | `{"detail": "Internal server error"}` |

**测试方法：**

<details>
<summary>点击展开测试步骤</summary>

**方式 1：使用浏览器**
```
访问：http://localhost:8000/api/topics/0
访问：http://localhost:8000/api/topics/1
```

**方式 2：使用 PowerShell（Windows）**
```powershell
# 获取主题 0 的详情
Invoke-RestMethod -Uri "http://localhost:8000/api/topics/0" -Method GET | ConvertTo-Json -Depth 3

# 查看主题名称和论文数量
$topic = Invoke-RestMethod -Uri "http://localhost:8000/api/topics/0" -Method GET
Write-Host "主题名称: $($topic.topic_name)"
Write-Host "论文数量: $($topic.paper_count)"
Write-Host "关键词: $($topic.keywords | ForEach-Object { $_.word })"

# 测试不存在的主题（应返回404）
try {
    Invoke-RestMethod -Uri "http://localhost:8000/api/topics/999" -Method GET
} catch {
    Write-Host "状态码: $($_.Exception.Response.StatusCode.value__)"
    Write-Host "错误信息: $($_.ErrorDetails.Message)"
}
```

**方式 3：使用 curl（Linux/Mac/Git Bash）**
```bash
# 获取主题详情
curl http://localhost:8000/api/topics/0 | jq .

# 只显示关键词
curl -s http://localhost:8000/api/topics/0 | jq '.keywords'

# 显示主题名称
curl -s http://localhost:8000/api/topics/0 | jq -r '.topic_name'

# 测试404错误
curl -i http://localhost:8000/api/topics/999
```

**方式 4：使用 Python**
```python
import requests

# 获取主题详情
response = requests.get("http://localhost:8000/api/topics/0")
topic = response.json()

print(f"主题ID: {topic['topic_id']}")
print(f"主题名称: {topic['topic_name']}")
print(f"论文数量: {topic['paper_count']}")
print(f"关键词: {[kw['word'] for kw in topic['keywords']]}")

# 测试404
response = requests.get("http://localhost:8000/api/topics/999")
print(f"状态码: {response.status_code}")  # 应为404
```

**预期结果：**
- 成功请求返回 200 状态码
- 返回指定主题的完整信息
- 不存在的主题返回 404 错误

</details>

---

#### 4.3.3 GET `/api/topics/{topic_id}/papers` - 获取主题下的论文

**功能：** 获取某个主题下的所有论文

**状态：** ✅ 已实现

**路径参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `topic_id` | int | 是 | 主题 ID |

**查询参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `limit` | int | 否 | 50 | 返回数量（1-100） |

**请求示例：**

```bash
# 获取主题 0 下的前 10 篇论文
curl "http://localhost:8000/api/topics/0/papers?limit=10"

# 获取主题 1 下的前 50 篇论文（默认）
curl "http://localhost:8000/api/topics/1/papers"
```

**响应示例：**

```json
{
  "total": 10,
  "papers": [
    {
      "id": "2301.00001",
      "title": "Attention Is All You Need",
      "abstract": "The dominant sequence transduction models...",
      "authors": ["Ashish Vaswani", "Noam Shazeer"],
      "published": "2023-01-01T00:00:00",
      "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
      "categories": ["cs.CL", "cs.AI"]
    }
    // ... 更多论文
  ]
}
```

**错误响应：**

| 状态码 | 说明 | 响应示例 |
|--------|------|---------|
| 404 | 主题不存在 | `{"detail": "Topic not found: 999"}` |
| 400 | 参数错误 | `{"detail": "limit must be between 1 and 100"}` |
| 500 | 服务器内部错误 | `{"detail": "Internal server error"}` |

**测试方法：**

<details>
<summary>点击展开测试步骤</summary>

**方式 1：使用浏览器**
```
访问：http://localhost:8000/api/topics/0/papers?limit=5
访问：http://localhost:8000/api/topics/1/papers?limit=10
```

**方式 2：使用 PowerShell（Windows）**
```powershell
# 获取主题 0 下的前 5 篇论文
$result = Invoke-RestMethod -Uri "http://localhost:8000/api/topics/0/papers?limit=5" -Method GET
Write-Host "返回论文数: $($result.total)"
$result.papers | ForEach-Object { 
    Write-Host "[$($_.id)] $($_.title)" 
}

# 查看第一篇论文的详细信息
$result.papers[0] | Format-List

# 获取所有主题的论文统计
0..4 | ForEach-Object {
    $papers = Invoke-RestMethod -Uri "http://localhost:8000/api/topics/$_/papers?limit=1" -Method GET
    Write-Host "主题 $_: $($papers.total) 篇论文"
}
```

**方式 3：使用 curl（Linux/Mac/Git Bash）**
```bash
# 获取论文列表
curl "http://localhost:8000/api/topics/0/papers?limit=5" | jq .

# 只显示论文标题
curl -s "http://localhost:8000/api/topics/0/papers?limit=5" | jq '.papers[].title'

# 显示论文数量
curl -s "http://localhost:8000/api/topics/0/papers" | jq '.total'

# 获取并保存论文列表到文件
curl -s "http://localhost:8000/api/topics/0/papers?limit=10" | jq . > topic_0_papers.json
```

**方式 4：使用 Python**
```python
import requests

# 获取主题下的论文
response = requests.get(
    "http://localhost:8000/api/topics/0/papers",
    params={"limit": 5}
)
data = response.json()

print(f"返回论文数: {data['total']}")
for paper in data['papers']:
    print(f"[{paper['id']}] {paper['title']}")
    print(f"  作者: {', '.join(paper['authors'][:3])}")
    print(f"  PDF: {paper['pdf_url']}")
    print()

# 批量获取所有主题的论文统计
for topic_id in range(5):
    response = requests.get(f"http://localhost:8000/api/topics/{topic_id}/papers?limit=1")
    count = response.json()['total']
    print(f"主题 {topic_id}: {count} 篇论文")
```

**方式 5：使用 Swagger UI**
```
1. 访问 http://localhost:8000/docs
2. 找到 GET /api/topics/{topic_id}/papers
3. 点击 "Try it out"
4. 输入 topic_id（如 0）
5. 输入 limit（如 10）
6. 点击 "Execute"
7. 查看响应结果
```

**预期结果：**
- 成功请求返回 200 状态码
- 返回指定数量的论文列表
- 每篇论文包含完整的元数据
- 不存在的主题返回 404 错误
- 超出范围的 limit 参数被限制在 1-100 之间

</details>

---

### 4.4 RAG 对话端点（待实现）

#### 4.4.1 POST `/api/chat/init` - 初始化对话

**功能：** 初始化一个新的对话会话

**状态：**  待实现

**请求体：**

```json
{
  "topic_id": 0  // 可选，限定主题范围
}
```

**预期响应示例：**

```json
{
  "session_id": "abc123def456",
  "topic_id": 0,
  "topic_name": "medical_imaging_diagnosis",
  "message": "对话已初始化，您可以开始提问了"
}
```

---

#### 4.4.2 POST `/api/chat/message` - 发送消息

**功能：** 向 RAG 系统发送问题并获取答案

**状态：**  待实现

**请求体：**

```json
{
  "session_id": "abc123def456",
  "message": "医学影像诊断的最新进展有哪些？"
}
```

**预期响应示例：**

```json
{
  "answer": "根据相关论文，医学影像诊断的最新进展包括：1. 深度学习在影像分类中的应用...",
  "sources": [
    {
      "title": "Deep Learning for Medical Imaging",
      "id": "2301.00001",
      "relevance": 0.92
    },
    {
      "title": "AI-Assisted Diagnosis in Radiology",
      "id": "2302.00050",
      "relevance": 0.88
    }
  ],
  "question": "医学影像诊断的最新进展有哪些？"
}
```

---

#### 4.4.3 GET `/api/chat/history` - 获取对话历史

**功能：** 获取某个会话的对话历史

**状态：**  待实现

**查询参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | str | 是 | 会话 ID |

**预期响应示例：**

```json
{
  "session_id": "abc123def456",
  "history": [
    {
      "role": "user",
      "content": "医学影像诊断的最新进展有哪些？",
      "timestamp": "2024-01-01T10:00:00"
    },
    {
      "role": "assistant",
      "content": "根据相关论文，医学影像诊断的最新进展包括...",
      "timestamp": "2024-01-01T10:00:05"
    }
  ]
}
```

---

## 5. 数据模型

### 5.1 PaperResponse - 论文响应模型

**用途：** 单个论文的完整信息

**字段说明：**

| 字段 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| `id` | str | 是 | 论文 ID | `"2301.00001"` |
| `title` | str | 是 | 论文标题 | `"BERT: Pre-training..."` |
| `abstract` | str | 是 | 论文摘要 | `"We introduce..."` |
| `authors` | List[str] | 是 | 作者列表 | `["Alice", "Bob"]` |
| `published` | datetime | 是 | 发布日期 | `"2023-01-01T00:00:00"` |
| `pdf_url` | str | 否 | PDF 下载链接 | `"https://arxiv.org/pdf/..."` |
| `categories` | List[str] | 是 | 论文分类 | `["cs.CL", "cs.AI"]` |

**完整示例：**

```json
{
  "id": "2301.00001",
  "title": "BERT: Pre-training of Deep Bidirectional Transformers",
  "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers...",
  "authors": ["Alice Smith", "Bob Johnson", "Charlie Brown"],
  "published": "2023-01-01T00:00:00",
  "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
  "categories": ["cs.CL", "cs.AI"]
}
```

---

### 5.2 PaperListResponse - 论文列表响应模型

**用途：** 包含多个论文的响应

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `total` | int | 是 | 论文总数 |
| `papers` | List[PaperResponse] | 是 | 论文列表 |

**示例：**

```json
{
  "total": 2,
  "papers": [
    {
      "id": "2301.00001",
      "title": "Paper 1",
      ...
    },
    {
      "id": "2301.00002",
      "title": "Paper 2",
      ...
    }
  ]
}
```

---

### 5.3 SearchRequest - 搜索请求模型

**用途：** 搜索论文的请求参数

**字段说明：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | str | 是 | - | 搜索关键词 |
| `max_results` | int | 否 | 10 | 最多返回数量（1-100） |
| `fields` | List[str] | 否 | `["title", "abstract"]` | 搜索字段 |

**示例：**

```json
{
  "query": "BERT",
  "max_results": 20,
  "fields": ["title", "abstract", "authors"]
}
```

---

### 5.4 ErrorResponse - 错误响应模型

**用途：** API 错误时的统一响应格式

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `detail` | str | 是 | 错误详情 |

**示例：**

```json
{
  "detail": "Paper not found: 2301.00001"
}
```

---