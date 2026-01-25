# Phase 1 实战指南：数据获取与探索

## 目标

✅ 从 arXiv 抓取 100+ 篇论文  
✅ 转换为标准化的 DataFrame  
✅ 保存为 CSV 文件  
✅ 进行初步数据探索

---

## 第一步：环境准备

### 1.1 创建虚拟环境

在项目根目录（`E:\AI_project\LiteratureReview_AIAssistant`）打开 PowerShell：

```powershell
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 如果遇到权限问题，运行：
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 1.2 安装依赖

```powershell
pip install -r requirements.txt
```

**注意：** 首次安装可能需要 5-10 分钟，特别是 `sentence-transformers` 等大型包。

---

## 第二步：运行数据抓取脚本

### 2.1 基本用法

```powershell
cd src
python -m scripts.fetch_and_save_arxiv --query "large language models" --max-results 100
```

### 2.2 其他示例

```powershell
# 抓取 PEFT 相关论文
python -m scripts.fetch_and_save_arxiv --query "parameter-efficient fine-tuning" --max-results 200

# 抓取 Transformer 相关论文
python -m scripts.fetch_and_save_arxiv --query "transformer" --max-results 150

# 抓取并自动清洗数据
python -m scripts.fetch_and_save_arxiv --query "LLM" --max-results 100 --clean
```

### 2.3 脚本输出

成功运行后，你会看到类似的输出：

```
============================================================
arXiv 论文抓取工具
============================================================
检索关键词: large language models
最大结果数: 100
输出目录: data/raw
数据清洗: 否
============================================================

[1/4] 正在从 arXiv 抓取论文...
✓ 成功抓取 100 篇论文

[2/4] 正在转换为 DataFrame...
✓ DataFrame 形状: (100, 9) (行数, 列数)

[3/4] 跳过数据清洗 (使用 --clean 启用)

[4/4] 正在保存到 CSV...
✓ 已保存到: E:\AI_project\LiteratureReview_AIAssistant\data\raw\arxiv_large_language_models.csv

============================================================
统计信息:
============================================================
论文总数: 100
作者数量: 456
发布年份范围: 2020-01-15 至 2024-12-20
平均摘要长度: 1245 字符
============================================================

✓ 完成! 你现在可以:
  1. 在 Excel/VSCode 中打开 arxiv_large_language_models.csv 查看数据
  2. 在 Jupyter Notebook 中加载: pd.read_csv('data/raw/arxiv_large_language_models.csv')
  3. 进入 Phase 2,对这些论文做 BERTopic 聚类
```

---

## 第三步：检查数据

### 3.1 在 VS Code 中查看

1. 打开 `data/raw/arxiv_large_language_models.csv`
2. VS Code 会自动用表格视图显示
3. 检查列：`id, title, abstract, authors, published, updated, categories, url`

### 3.2 在 Python 中查看

```python
import pandas as pd

# 加载数据
df = pd.read_csv('data/raw/arxiv_large_language_models.csv')

# 查看基本信息
print(df.shape)  # (100, 9)
print(df.columns)
print(df.head())

# 查看第一篇论文的摘要
print(df.iloc[0]['abstract'])
```

### 3.3 在 Jupyter Notebook 中探索

我们已经为你准备了一个探索模板（`notebooks/01_explore_arxiv.ipynb`），你可以：

```powershell
# 安装 jupyter (如果还没装)
pip install jupyter

# 启动 Jupyter
jupyter notebook
```

然后打开 `notebooks/01_explore_arxiv.ipynb`，运行其中的代码单元。

---

## 第四步：理解代码架构

### 4.1 模块划分

```
src/
├── core/
│   ├── arxiv/
│   │   ├── models.py       # 数据模型 (ArxivPaper)
│   │   ├── client.py       # API 调用逻辑
│   │   └── pipeline.py     # 数据转换与保存
│   └── config.py           # 全局配置
└── scripts/
    └── fetch_and_save_arxiv.py  # 命令行工具
```

### 4.2 数据流

```
arXiv API 
  → XML/JSON 
  → ArxivPaper (Pydantic 模型) 
  → List[ArxivPaper] 
  → pandas.DataFrame 
  → CSV 文件
```

### 4.3 关键设计

**为什么用 Pydantic？**
- 自动类型验证
- 防止脏数据进入系统
- 便于后续与 FastAPI 集成

**为什么用 DataFrame？**
- 统一的表格格式
- 丰富的数据操作接口
- 与 BERTopic 无缝集成

**为什么分成多个模块？**
- 类似 C 语言的 `.h` 和 `.c` 分离
- 便于测试、复用、维护
- 符合工业界的工程规范

---

## 常见问题

### Q1: 抓取速度很慢？

**原因：** arXiv API 有速率限制（建议每 3 秒 1 次请求）。  
**解决：** 这是正常的，耐心等待。100 篇论文通常需要 1-2 分钟。

### Q2: 报错 `ModuleNotFoundError: No module named 'arxiv'`？

**原因：** 虚拟环境未激活或依赖未安装。  
**解决：**

```powershell
.\.venv\Scripts\Activate.ps1
pip install arxiv
```

### Q3: 想抓取更多论文（超过 200 篇）？

**建议：** 分批抓取，避免单次请求过大导致超时。

```powershell
# 方法 1: 多次运行，用不同的关键词
python -m scripts.fetch_and_save_arxiv --query "LLM training" --max-results 200
python -m scripts.fetch_and_save_arxiv --query "LLM inference" --max-results 200

# 方法 2: 在代码中添加时间过滤（未来扩展）
```

### Q4: 如何处理 Rate Limit？

当前的 `arxiv` 包已经内置了合理的请求间隔，一般不会触发限流。如果遇到 429 错误，可以：

1. 减少 `max_results`
2. 使用 `fetch_arxiv_papers_with_retry()` 函数（已在 `client.py` 中提供）

---

## 验收标准（自查清单）

在进入 Phase 2 之前，确保：

- [ ] ✅ 成功安装所有依赖
- [ ] ✅ 成功运行脚本并生成 CSV
- [ ] ✅ CSV 文件包含至少 100 篇论文
- [ ] ✅ 能用 VS Code / Jupyter 打开并查看数据
- [ ] ✅ 理解了 Pydantic 模型的作用
- [ ] ✅ 理解了数据流：API → Model → DataFrame → CSV

---

## 下一步：Phase 2

现在你已经有了数据，接下来我们将：

1. **学习句向量（Sentence Embeddings）**
   - 什么是 embedding？
   - 为什么要把文本转成向量？
   - 如何选择 embedding 模型？

2. **使用 BERTopic 进行主题聚类**
   - BERTopic 的工作原理
   - 训练模型并提取主题
   - 可视化主题分布

3. **为每篇论文分配主题标签**
   - 将 `topic_id` 和 `topic_name` 回写到 DataFrame
   - 保存带主题标签的新 CSV

**准备好了吗？** 完成 Phase 1 后，告诉我，我将为你提供详细的 Phase 2 教程！

