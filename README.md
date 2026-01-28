# 科研文献综述 AI 助手 (Scientific Literature Review Assistant)

## 项目简介

这是一个交互式 AI 应用,旨在帮助研究者快速理解和整理大量科研文献。系统通过以下流程工作:

1. **Search & Ingest**: 从 arXiv API 获取论文数据
2. **Cluster**: 使用 BERTopic 自动发现主题
3. **Synthesize**: 使用 LLM 生成主题综述和论文摘要
4. **Explore**: 通过 Streamlit UI 可视化和交互
5. **Chat (RAG)**: 与特定主题进行对话问答

## 技术栈

- **Language**: Python 3.10+
- **Backend**: FastAPI + Pydantic
- **AI/NLP**: BERTopic, LangChain, OpenAI API
- **Data**: Pandas, ChromaDB
- **Frontend**: Streamlit

## 项目结构

```
.
├── src/
│   ├── backend/              # FastAPI 后端
│   ├── core/                 # 核心算法库
│   ├── frontend/             # Streamlit 前端
│   └── scripts/              # 命令行工具
├── data/                     # 数据存储
├── models/                   # 模型存储
├── notebooks/                # 实验和分析
├── tests/                    # 单元测试
└── requirements.txt
```

## 快速开始

### 1. 环境准备（使用 Anaconda）

```bash
# 创建 Conda 环境
conda create -n literature_review python=3.10 -y

# 安装核心包（Conda）
conda install -n literature_review -c conda-forge hdbscan umap-learn scikit-learn numpy pandas -y

# 激活环境
conda activate literature_review

# 安装其他包（pip）
pip install -r requirements-conda.txt
```

**详细安装指南：** 查看 `INSTALL_MANUAL.md` 或 `QUICKSTART.md`

### 2. Phase 1: 数据获取

```bash
cd src
python -m scripts.fetch_and_save_arxiv --query "large language models" --max-results 100
```

### 3. Phase 2: 主题聚类 (即将实现)

```bash
python -m scripts.train_bertopic --input-csv data/raw/arxiv_large_language_models.csv
```

### 4. Phase 3: 启动后端

```bash
cd src
uvicorn backend.main:app --reload
```

### 5. Phase 4: 启动前端

```bash
cd src
streamlit run frontend/app.py
```

## 学习路线图

### Phase 1: Data Ingestion ✅
- arXiv API 调用
- Pydantic 模型设计
- DataFrame 数据处理

### Phase 2: Embedding & BERTopic (进行中)
- 句向量生成
- BERTopic 主题建模
- 主题可视化

### Phase 3: FastAPI Backend
- REST API 设计
- 服务层封装
- 请求/响应模型

### Phase 4: Streamlit Frontend
- UI 组件设计
- API 调用封装
- 数据可视化

### Phase 5: RAG & Production
- ChromaDB 向量存储
- LangChain RAG 对话
- 日志、测试、配置管理

## 参考资料

- [BERTopic 论文](https://arxiv.org/abs/2203.05794) - Grootendorst, M. (2022)
- [BERT 论文](https://aclanthology.org/N19-1423/) - Devlin et al. (2019)
- [Transformer 论文](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017)

## 导师

Dr. XU Lingling

## License

MIT

