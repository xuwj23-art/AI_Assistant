# 后端 API 常见问题（FAQ）

本文档补充 `BACKEND_README.md`，提供常见问题的解答和最佳实践指南。

---

## 目录

1. [主题管理相关](#1-主题管理相关)
2. [测试和调试](#2-测试和调试)
3. [性能和优化](#3-性能和优化)
4. [数据格式](#4-数据格式)
5. [生产部署](#5-生产部署)

---

## 1. 主题管理相关

### Q1: 为什么 `/api/topics` 返回的主题列表为空？

**A:** 主题列表为空可能有以下几种原因：

#### 原因 1：未运行主题建模

需要先运行 BERTopic 训练脚本生成主题数据：

```bash
# 步骤 1: 生成嵌入向量
cd src
python -m scripts.generate_embeddings --input ../data/raw/arxiv_Transformer.csv --output ../data/processed/embeddings.npy

# 步骤 2: 训练主题模型
python -m scripts.train_topics \
  --csv ../data/raw/arxiv_Transformer.csv \
  --embeddings ../data/processed/embeddings.npy \
  --output ../models/bertopic_model
```

#### 原因 2：数据文件路径不正确

检查 `src/core/api/topic_routes.py` 中的数据文件路径：
- 服务优先查找 `data/processed/arxiv_Transformer_with_topics.csv`
- 如果不存在，则使用 `data/raw/arxiv_Transformer.csv`

#### 原因 3：CSV 文件缺少主题列

确保 CSV 文件包含以下列：
- `topic_id` - 主题ID（整数）
- `topic_name` - 主题名称（字符串）

**验证方法：**
```python
import pandas as pd
df = pd.read_csv('data/processed/arxiv_Transformer_with_topics.csv')
print('必须包含的列:', 'topic_id' in df.columns, 'topic_name' in df.columns)
print('前5行主题信息:')
print(df[['id', 'topic_id', 'topic_name']].head())
```

---

### Q2: 如何理解主题的关键词分数？

**A:** 关键词分数（score）表示该词与主题的相关性：

- **分数范围：** 0.0 - 1.0
- **含义：** 分数越高，表示该词越能代表这个主题
- **计算方式：** 由 BERTopic 的 c-TF-IDF 算法计算得出
  - c-TF-IDF 衡量词在主题中的重要性
  - 考虑词频和主题内的分布

**降级方案：**
如果未使用 BERTopic 模型，分数会根据词的位置递减：
- 第一个词：1.0
- 第二个词：0.9
- 第三个词：0.8
- 依此类推（最低 0.3）

**示例：**
```json
{
  "topic_id": 0,
  "keywords": [
    {"word": "transformer", "score": 0.85},  // 最能代表主题
    {"word": "attention", "score": 0.72},     // 次重要
    {"word": "mechanism", "score": 0.68}      // 相关性较低
  ]
}
```

---

### Q3: 主题 ID 为 -1 代表什么？

**A:** 主题 ID 为 -1 表示"噪声"或"离群点"：

- **含义：** 这些论文无法明确归类到任何主题
- **原因：**
  - 论文内容过于独特或小众
  - 与其他论文的相似度较低
  - 不满足 BERTopic 的聚类阈值

- **处理方式：**
  - 在 `/api/topics` 接口中，离群点会被自动过滤
  - 如需查看离群点，可使用 `/api/papers` 并筛选 `topic_id == -1`

**查看离群点数量：**
```python
import pandas as pd
df = pd.read_csv('data/processed/arxiv_Transformer_with_topics.csv')
outliers = df[df['topic_id'] == -1]
print(f"离群点数量: {len(outliers)}")
print(f"占比: {len(outliers) / len(df) * 100:.2f}%")
```

---

### Q4: 如何调整主题数量？

**A:** 在训练主题模型时使用 `--n-topics` 参数：

```bash
# 将主题合并至 5 个
python -m scripts.train_topics \
  --csv ../data/raw/arxiv_Transformer.csv \
  --embeddings ../data/processed/embeddings.npy \
  --output ../models/bertopic_model \
  --n-topics 5
```

**参数说明：**
- `--n-topics 5` - 最终保留 5 个主题
- BERTopic 会合并最相似的主题
- 建议范围：3-15 个主题

**选择建议：**
- 数据量 < 100 篇：3-5 个主题
- 数据量 100-500 篇：5-10 个主题
- 数据量 > 500 篇：10-15 个主题

---

### Q5: 为什么某些主题的论文数量差异很大？

**A:** 这是正常现象，原因包括：

1. **语义相似度分布不均**
   - BERTopic 基于论文的实际语义相似度进行聚类
   - 某些研究方向天然更集中

2. **数据源的偏差**
   - 不同研究方向的论文发表数量本身就不均衡

3. **参数影响**
   - `min_topic_size` 参数决定最小主题大小
   - 过小的簇会被标记为离群点

**解决方案：**

**方案 1：使用 `reduce_topics` 合并小主题**
```bash
python -m scripts.train_topics \
  --n-topics 8  # 合并至8个主题
```

**方案 2：调整 `min_topic_size` 参数**
```python
# 在 src/core/nlp/topic_modeling.py 中修改
topic_model = TopicModeler(
    min_topic_size=20  # 增大最小主题大小
)
```

**方案 3：使用平衡采样**
```python
# 每个主题采样相同数量的论文
import pandas as pd
df = pd.read_csv('data/processed/arxiv_Transformer_with_topics.csv')
balanced = df.groupby('topic_id').apply(lambda x: x.sample(min(50, len(x))))
```

---

## 2. 测试和调试

### Q6: 如何在 Swagger UI 中测试接口？

**A:** 详细步骤：

**步骤 1：启动服务**
```bash
cd src
uvicorn main:app --reload
```

**步骤 2：访问 Swagger UI**
打开浏览器访问：http://localhost:8000/docs

**步骤 3：测试接口**
1. 找到要测试的接口（如 `GET /api/topics`）
2. 点击该接口展开详情
3. 点击右侧的 "Try it out" 按钮
4. 填写参数（如果有）
5. 点击 "Execute" 按钮
6. 查看下方的 "Response body" 和 "Response headers"

**步骤 4：查看响应**
- **Status code:** 200 表示成功
- **Response body:** JSON 格式的响应数据
- **Response headers:** HTTP 头信息

**快捷提示：**
- 使用 "Schema" 标签查看数据结构
- 使用 "Example Value" 查看示例数据
- 点击 "Download" 下载响应数据

---

### Q7: 如何查看服务的日志输出？

**A:** 日志会输出到启动服务的终端：

```bash
cd src
uvicorn main:app --reload
# 日志会显示在此终端中
```

**关键日志示例：**

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
[TopicService] 加载了 500 篇论文数据
[TopicService] 已加载 BERTopic 模型: ../models/bertopic_model
INFO:     127.0.0.1:12345 - "GET /api/topics HTTP/1.1" 200 OK
```

**日志含义：**
- `[TopicService] 加载了 N 篇论文数据` - 数据加载成功
- `[TopicService] 已加载 BERTopic 模型` - 模型加载成功
- `[TopicService] 警告: 数据中没有 topic_id 列` - 需要运行主题建模
- `200 OK` - 请求成功
- `404 Not Found` - 资源不存在
- `500 Internal Server Error` - 服务器错误

**调试技巧：**
```bash
# 增加日志详细程度
uvicorn main:app --reload --log-level debug

# 将日志保存到文件
uvicorn main:app --reload 2>&1 | tee app.log
```

---

### Q8: 如何重新加载数据而不重启服务？

**A:** 有两种方式：

**方式 1：手动重启服务**
- 在终端按 `Ctrl+C` 停止服务
- 重新运行 `uvicorn main:app --reload`

**方式 2：触发自动重载**
FastAPI 的 `--reload` 模式会在代码修改时自动重启：
1. 打开任意 `.py` 文件（如 `src/main.py`）
2. 添加一个空格或注释
3. 保存文件
4. 服务会自动重启并加载新数据

**注意：** 数据文件修改不会触发自动重启，只有 Python 代码修改才会。

---

## 3. 性能和优化

### Q9: 接口响应速度慢怎么办？

**A:** 优化建议：

#### 1. 使用分页限制返回数量

```bash
# 只返回 10 条结果（而非默认的 50 条）
curl "http://localhost:8000/api/topics/0/papers?limit=10"
```

#### 2. 利用缓存机制

服务使用单例模式缓存数据，首次请求后会更快：

```python
# 在 dependencies.py 中，服务只初始化一次
_topic_service: TopicService = None  # 全局缓存

def get_topic_service():
    global _topic_service
    if _topic_service is None:
        _topic_service = TopicService(...)  # 只执行一次
    return _topic_service
```

#### 3. 优化 CSV 文件

- **删除不必要的列：** 只保留 API 需要的字段
- **压缩文件：** 使用 gzip 压缩（Pandas 自动支持）

```python
# 保存为压缩文件
df.to_csv('data.csv.gz', compression='gzip', index=False)

# 读取压缩文件（自动解压）
df = pd.read_csv('data.csv.gz')
```

#### 4. 使用数据库

对于大规模数据（>10,000 篇论文），考虑使用数据库：

```python
# 使用 SQLite（轻量级）
import sqlite3
conn = sqlite3.connect('papers.db')
df.to_sql('papers', conn, if_exists='replace', index=False)

# 或使用 PostgreSQL（生产环境）
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@localhost/dbname')
df.to_sql('papers', engine, if_exists='replace', index=False)
```

#### 5. 添加索引

如果使用数据库，为常查询的字段添加索引：

```sql
CREATE INDEX idx_topic_id ON papers(topic_id);
CREATE INDEX idx_published ON papers(published);
```

---

### Q10: 如何处理大量论文数据？

**A:** 针对大规模数据（>10,000 篇）的建议：

#### 1. 分批处理

将论文分成多个主题领域，分别建模：

```bash
# 为不同主题分别训练
python -m scripts.fetch_and_save_arxiv --query "computer vision" --max-results 500
python -m scripts.fetch_and_save_arxiv --query "natural language processing" --max-results 500
```

#### 2. 增量更新

定期更新主题模型而非全量重训：

```python
# 使用 BERTopic 的增量学习功能
topic_model = BERTopic.load('models/bertopic_model')
new_topics = topic_model.update_topics(new_docs, new_embeddings)
```

#### 3. 异步处理

使用 FastAPI 的后台任务处理耗时操作：

```python
from fastapi import BackgroundTasks

@app.post("/api/topics/train")
async def train_topics(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_topic_model)
    return {"message": "训练已开始，请稍后查看结果"}
```

#### 4. 使用向量数据库

存储嵌入向量以加速语义搜索：

```bash
pip install chromadb

# 使用 ChromaDB
from chromadb import Client
client = Client()
collection = client.create_collection("papers")
collection.add(embeddings=embeddings, documents=abstracts, ids=paper_ids)
```

#### 5. 分布式部署

使用多个服务实例处理请求：

```bash
# 使用 Gunicorn 启动 4 个 worker
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

---

## 4. 数据格式

### Q11: CSV 文件应该包含哪些列？

**A:** 完整的列表：

#### 基础论文数据（必需）

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `id` | str | 论文 ID | `"2301.00001"` |
| `title` | str | 标题 | `"Attention Is All You Need"` |
| `abstract` | str | 摘要 | `"The dominant sequence..."` |
| `authors` | str/list | 作者列表 | `["Ashish Vaswani", "Noam Shazeer"]` |
| `published` | datetime | 发布日期 | `"2023-01-01T00:00:00"` |
| `categories` | str/list | 分类 | `["cs.CL", "cs.AI"]` |
| `url` | str | 论文链接 | `"https://arxiv.org/abs/2301.00001"` |
| `pdf_url` | str | PDF 链接 | `"https://arxiv.org/pdf/2301.00001.pdf"` |

#### 主题数据（可选，由训练脚本生成）

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `topic_id` | int | 主题 ID | `0` |
| `topic_name` | str | 主题名称 | `"transformer_attention"` |

#### 扩展字段（可选）

| 列名 | 类型 | 说明 |
|------|------|------|
| `updated` | datetime | 更新日期 |
| `comment` | str | 备注信息 |
| `journal_ref` | str | 期刊引用 |
| `doi` | str | DOI 标识 |

---

### Q12: 如何将论文导出为其他格式？

**A:** 使用 API 获取 JSON 数据后转换：

#### 导出为 CSV

```python
import requests
import pandas as pd

# 获取论文数据
response = requests.get("http://localhost:8000/api/topics/0/papers?limit=100")
papers = response.json()['papers']

# 转换为 DataFrame
df = pd.DataFrame(papers)

# 保存为 CSV（支持中文）
df.to_csv('topic_0_papers.csv', index=False, encoding='utf-8-sig')
```

#### 导出为 Excel

```python
# 安装依赖：pip install openpyxl
df.to_excel('topic_0_papers.xlsx', index=False)
```

#### 导出为 JSON

```bash
# 直接使用 curl 保存
curl "http://localhost:8000/api/topics/0/papers?limit=100" > papers.json

# 或使用 jq 美化输出
curl "http://localhost:8000/api/topics/0/papers" | jq . > papers_pretty.json
```

#### 导出为 Markdown

```python
import requests

response = requests.get("http://localhost:8000/api/topics/0/papers?limit=10")
papers = response.json()['papers']

with open('papers.md', 'w', encoding='utf-8') as f:
    f.write('# 论文列表\n\n')
    for paper in papers:
        f.write(f"## [{paper['title']}]({paper['pdf_url']})\n\n")
        f.write(f"**作者：** {', '.join(paper['authors'][:3])}\n\n")
        f.write(f"**摘要：** {paper['abstract'][:200]}...\n\n")
        f.write('---\n\n')
```

#### 导出为 BibTeX

```python
def to_bibtex(paper):
    authors = ' and '.join(paper['authors'])
    year = paper['published'][:4]
    return f"""@article{{{paper['id'].replace('.', '_')},
  title={{{paper['title']}}},
  author={{{authors}}},
  journal={{arXiv preprint arXiv:{paper['id']}}},
  year={{{year}}}
}}
"""

# 批量导出
with open('papers.bib', 'w', encoding='utf-8') as f:
    for paper in papers:
        f.write(to_bibtex(paper) + '\n')
```

---

## 5. 生产部署

### Q13: 如何部署到生产环境？

**A:** 生产部署建议：

#### 1. 使用 Gunicorn + Uvicorn

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动 4 个 worker 进程
cd src
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

**参数说明：**
- `-w 4` - 4 个 worker 进程（建议：CPU 核心数 * 2 + 1）
- `-k uvicorn.workers.UvicornWorker` - 使用 Uvicorn worker
- `--bind 0.0.0.0:8000` - 监听所有网络接口的 8000 端口

#### 2. 配置 CORS

在 `main.py` 中限制允许的域名：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # 替换 ["*"]
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # 限制允许的方法
    allow_headers=["*"],
)
```

#### 3. 使用环境变量

创建 `.env` 文件：

```bash
# .env
DATA_PATH=/app/data/papers.csv
MODEL_PATH=/app/models/bertopic_model
DEBUG=false
LOG_LEVEL=info
```

在代码中读取：

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    data_path: str
    model_path: str
    debug: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()
```

#### 4. 添加认证

使用 API Key 保护接口：

```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY = "your-secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.get("/api/topics", dependencies=[Depends(verify_api_key)])
def get_topics():
    ...
```

#### 5. 使用反向代理（Nginx）

Nginx 配置示例：

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### 6. 容器化（Docker）

创建 `Dockerfile`：

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

构建和运行：

```bash
# 构建镜像
docker build -t literature-api .

# 运行容器
docker run -d -p 8000:8000 --name literature-api literature-api
```

使用 Docker Compose：

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - LOG_LEVEL=info
```

---

### Q14: 如何监控服务健康状态？

**A:** 使用健康检查端点和监控工具：

#### 1. 定时检查

```bash
# 每分钟检查一次（Linux/Mac）
watch -n 60 'curl -s http://localhost:8000/health'

# 或使用 cron 定时任务
* * * * * curl -s http://localhost:8000/health || echo "Service down!" | mail -s "Alert" admin@example.com
```

#### 2. 使用 Uptime Robot

免费的在线监控服务：https://uptimerobot.com/

配置：
- Monitor Type: HTTP(s)
- URL: http://yourdomain.com/health
- Monitoring Interval: 5 minutes

#### 3. 使用 Prometheus + Grafana

安装 Prometheus 客户端：

```bash
pip install prometheus-fastapi-instrumentator
```

在 `main.py` 中添加：

```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)
```

访问指标：http://localhost:8000/metrics

#### 4. 日志监控

使用 ELK Stack（Elasticsearch + Logstash + Kibana）：

```python
import logging
from pythonjsonlogger import jsonlogger

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)
```

#### 5. 应用性能监控（APM）

使用 New Relic 或 DataDog：

```bash
pip install newrelic

# 启动时添加 agent
NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program uvicorn main:app
```

---

## 附录

### 相关资源

- **FastAPI 文档：** https://fastapi.tiangolo.com/
- **BERTopic 文档：** https://maartengr.github.io/BERTopic/
- **Pandas 文档：** https://pandas.pydata.org/docs/
- **Docker 文档：** https://docs.docker.com/

### 更新日志

- **v1.0** (2026-03-11) - 初版发布

---

**返回主文档：** [BACKEND_README.md](./BACKEND_README.md)
