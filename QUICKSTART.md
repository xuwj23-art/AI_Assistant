# 快速启动指南 (Quick Start)

## 前置要求

- Python 3.10 或更高版本
- Windows 10/11（其他操作系统请相应调整命令）
- 稳定的网络连接（用于下载依赖和调用 arXiv API）

---

## 第一步：环境准备（5 分钟）

### 1. 打开 Anaconda Prompt

在开始菜单搜索 "Anaconda Prompt" 并打开。

### 2. 进入项目目录

```bash
cd /d E:\AI_project\LiteratureReview_AIAssistant
```

### 3. 创建 Conda 环境

```bash
conda create -n literature_review python=3.10 -y
```

### 4. 安装核心包（用 Conda）

**重要：使用 `-n` 参数指定环境名！**

```bash
conda install -n literature_review -c conda-forge hdbscan umap-learn scikit-learn numpy pandas -y
```

这一步可能需要 5-10 分钟，请耐心等待。

### 5. 激活环境

```bash
conda activate literature_review
```

你应该看到命令行前面出现 `(literature_review)`。

### 6. 安装其他包（用 pip）

```bash
pip install -r requirements-conda.txt
```

这一步可能需要 5-10 分钟。

### 7. 验证安装

```bash
python -c "import hdbscan; print('hdbscan: OK')"
python -c "import bertopic; print('bertopic: OK')"
python -c "import fastapi; print('fastapi: OK')"
python -c "import streamlit; print('streamlit: OK')"
```

如果都输出 "OK"，说明安装成功！

---

## 第二步：配置环境变量（可选，Phase 3+ 需要）

> **重要：** Phase 1 和 Phase 2 **不需要** API Key，可以先跳过此步骤！

### 1. 复制环境变量模板

```bash
copy env.example .env
```

### 2. 编辑 `.env` 文件

用任何文本编辑器（如 VS Code、记事本）打开 `.env`，根据你的情况配置：

#### 选项 A：使用 OpenRouter（推荐，国内可用）

```env
OPENAI_API_KEY=sk-or-v1-你的OpenRouter密钥
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-3.5-turbo
```

#### 选项 B：使用 OpenAI 官方

```env
OPENAI_API_KEY=sk-你的OpenAI密钥
OPENAI_API_BASE=
OPENAI_MODEL=gpt-3.5-turbo
```

> **详细配置指南：** 查看 `docs/OPENROUTER_SETUP.md`

### 3. 为什么需要配置？

- **API Key**：用于调用 LLM 生成摘要和对话（Phase 3-5）
- **安全性**：密钥不会提交到 Git，保护你的账号
- **灵活性**：可以轻松切换不同的 LLM 提供商

### 4. 什么时候需要？

| Phase | 是否需要 API Key | 说明 |
|-------|-----------------|------|
| Phase 1 | ❌ 不需要 | 只抓取论文数据 |
| Phase 2 | ❌ 不需要 | 只做主题聚类 |
| Phase 3 | ✅ 需要 | 生成摘要功能 |
| Phase 4 | ✅ 需要 | UI 中的摘要功能 |
| Phase 5 | ✅ 需要 | RAG 对话功能 |

---

## 第三步：运行第一个脚本（2 分钟）

### 从 arXiv 抓取论文

```bash
cd src
python -m scripts.fetch_and_save_arxiv --query "large language models" --max-results 100
```
我们把它拆解成 **4 个部分** 来详细解析：

`python` + `-m` + `scripts.fetch_and_save_arxiv` + `--query ... --max-results ...`

---

### 1. `python` (召唤工头)
*   **含义**：呼叫 Python 解释器。
*   **作用**：告诉电脑：“接下来的内容，请用 Python 这种语言来运行。”
*   **注意**：在某些 Mac 或 Linux 系统上，可能需要写成 `python3`。

### 2. `-m` (运行模式)
*   **含义**：module（模块）的缩写。
*   **作用**：这是一个很关键的参数。它告诉 Python：“不要把后面的名字当成一个普通的文件路径（比如 `scripts/xxx.py`），而是把它当成一个**模块**来运行。”
*   **为什么要这样写？**
    *   如果不加 `-m`，你通常需要写成文件路径：`python scripts/fetch_and_save_arxiv.py`。
    *   **加了 `-m` 的好处**：它能更好地处理项目中的“引用关系”（import）。比如，如果脚本里引用了项目里其他的代码，用 `-m` 运行通常不会报错，而直接运行文件路径经常会报“找不到模块”的错误。

### 3. `scripts.fetch_and_save_arxiv` (具体的任务书)
*   **含义**：这是你要运行的**代码文件的位置**。
*   **对应关系**：
    *   `scripts` 对应文件夹名。
    *   `.` (点) 对应目录分隔符（就像文件夹里的 `/`）。
    *   `fetch_and_save_arxiv` 对应文件名（去掉了 `.py` 后缀）。
*   **实际文件**：这条指令实际上是在运行 `src/scripts/fetch_and_save_arxiv.py` 这个文件里的代码。

### 4. `--query "large language models" --max-results 100` (任务参数)
这是传递给脚本的**具体要求**（参数）。脚本里写了代码来接收这些参数：

*   **`--query "large language models"`**：
    *   **`--query`**：这是脚本里定义好的一个开关，意思是“我要搜索的关键词是...”。
    *   **`"large language models"`**：这是具体的值。**为什么要加引号？** 因为中间有空格。如果不加引号，电脑会以为 `large` 是一个参数，`language` 是另一个参数，程序就会乱套。
*   **`--max-results 100`**：
    *   **`--max-results`**：这是另一个开关，意思是“最大下载数量”。
    *   **`100`**：告诉脚本“只要前 100 篇，多了不要”。

---


**预期输出：**

```
============================================================
arXiv 论文抓取工具
============================================================
检索关键词: large language models
最大结果数: 100
...
✓ 已保存到: E:\AI_project\LiteratureReview_AIAssistant\data\raw\arxiv_large_language_models.csv
```

### 查看结果

在 VS Code 中打开 `data/raw/arxiv_large_language_models.csv`，你会看到一个包含 100 篇论文的表格。


### 1. arXiv 是什么？（AI 界的“免费图书馆”）

**arXiv** (读音类似 "archive"，/ɑːrˈkaɪv/) 是一个由**康奈尔大学**运营的网站。

*   **地位：** 它是全球最重要、最权威的**学术论文预印本（Pre-print）数据库**。
*   **内容：** 里面存放了物理学、数学、**计算机科学（特别是人工智能）**等领域的数百万篇论文。
*   **特点：**
    1.  **免费：** 任何人都可以免费阅读和下载。
    2.  **极快：** 传统的学术期刊发表论文可能需要几个月甚至一年，但 AI 发展太快了，等不及。所以，OpenAI、Google、Meta 以及全球的顶尖大学研究员，都会把写好的论文**第一时间**上传到 arXiv。
    3.  **源头：** 你听过的 GPT-4、Transformer、Stable Diffusion 等几乎所有重大 AI 突破的论文，最初都是发在这里的。

**总结：** arXiv 就是 AI 知识的“源头活水”。

### 2. 为什么可以“抓取”？（官方允许的“自动借书”）

*   **API（应用程序接口）：** arXiv 非常开放，它专门为开发者提供了一个**官方的、合法的**数据接口（API）。
*   **原理：**
    1.  你的 Python 脚本（`fetch_and_save_arxiv.py`）就像一个**“自动化的图书管理员”**。
    2.  它向 arXiv 的服务器发送了一条指令：“请给我关于 'large language models'（大语言模型）的最新 100 篇论文的信息。”
    3.  arXiv 的服务器收到指令后，非常配合地打包好数据（标题、作者、摘要、发布时间、下载链接），发送回你的电脑。
    4.  你的脚本把这些数据整理成一行行整齐的文字，存进了 Excel 表格（`.csv` 文件）里。

### 3. 这些论文的来源是什么？

*   **生产者：** 全球的科研人员（教授、博士生、企业研究员）。
*   **流程：**
    1.  研究员写完论文。
    2.  研究员主动登录 arXiv 网站上传 PDF 文件。
    3.  arXiv 进行简单的审核（确保不是垃圾广告）。
    4.  论文上线，对全世界公开。

### 这个步骤在项目中的意义

你正在做的这个项目叫“文献综述助手”。
*   **第一步（现在做的）：** 你需要先有“米”才能下锅。这个脚本就是去把“米”（最新的 AI 论文数据）从仓库（arXiv）里搬运到你的电脑硬盘里。
*   **第二步（之后要做的）：** 你的电脑里有了这 100 篇论文的摘要后，我们就可以用你刚刚配置好的 **OpenRouter (AI 大脑)** 来阅读这些摘要，让 AI 帮你总结：“这 100 篇论文主要在讲什么？有哪些创新点？”

**简单来说：** 你现在是在指挥 Python 帮你“跑腿”，去图书馆把书单拿回来，为后面让 AI 帮你“读书”做准备。

---

## 第四步：探索数据（可选）

### 方法 1：在 Python 中查看

```python
import pandas as pd

df = pd.read_csv('../data/raw/arxiv_large_language_models.csv')
print(df.head())
print(f"\n总共 {len(df)} 篇论文")
```

### 方法 2：使用 Jupyter Notebook

```bash
# 确保在项目根目录
cd ..  # 如果你在 src/ 下

# 安装 Jupyter（如果还没装）
conda install jupyter -y

# 启动 Jupyter
jupyter notebook
```

然后在 Jupyter 中创建新的 Notebook 进行数据探索。

---

## 常见问题排查

### 问题 1：`ModuleNotFoundError: No module named 'arxiv'`

**原因：** Conda 环境未激活或依赖未安装。

**解决：**
```bash
conda activate literature_review
pip install arxiv
```

### 问题 2：Conda 环境激活失败

**原因：** Conda 未正确初始化。

**解决：**
```bash
conda init
# 然后重启 Anaconda Prompt
conda activate literature_review
```

### 问题 3：arXiv API 请求超时

**原因：** 网络问题或 arXiv 服务器繁忙。

**解决：**
- 减少 `--max-results`（例如改为 50）
- 稍后重试
- 检查网络连接

### 问题 4：pip 安装速度很慢

**解决：** 使用国内镜像源
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 5：安装 hdbscan 时报错（编译失败）

**错误信息：**
```
Building wheel for hdbscan (pyproject.toml) ... error
fatal error C1083: 无法打开包括文件: "io.h"
```

**原因：** Windows 上 pip 安装 hdbscan 需要编译 C++ 代码。

**解决方案：**

**方法 1：使用 Conda 安装（推荐）**
```bash
# 删除失败的安装
pip uninstall hdbscan -y

# 用 Conda 安装预编译版本
conda install -c conda-forge hdbscan -y
```

**方法 2：使用自动安装脚本**
```bash
# 删除环境重新安装
conda deactivate
conda remove -n literature_review --all -y

# 在 Anaconda Prompt 中运行
cd E:\AI_project\LiteratureReview_AIAssistant
setup_env.bat
```

**方法 3：手动分步安装**
```bash
# 先安装需要编译的包（指定环境名）
conda install -n literature_review -c conda-forge hdbscan umap-learn scikit-learn numpy pandas -y

# 激活环境
conda activate literature_review

# 再安装其他包
pip install -r requirements-conda.txt
```

> **详细说明：** 查看 `install_guide_conda.md`

### 问题 6：想在 VS Code 中使用 Conda 环境

**解决：**
1. 在 VS Code 中按 `Ctrl+Shift+P`
2. 输入 "Python: Select Interpreter"
3. 选择 `literature_review` 环境（路径类似 `C:\Users\YourName\anaconda3\envs\literature_review\python.exe`）

---

## 下一步

✅ **恭喜！** 你已经完成了 Phase 1 的第一部分。

**接下来你可以：**

1. **深入学习 Phase 1：** 阅读 `docs/PHASE1_GUIDE.md`
2. **查看完整计划：** 阅读 `docs/MASTER_PLAN.md`
3. **尝试不同关键词：**
   ```bash
   python -m scripts.fetch_and_save_arxiv --query "transformer" --max-results 150
   python -m scripts.fetch_and_save_arxiv --query "BERT" --max-results 200
   ```
4. **阅读源代码：** 打开 `src/core/arxiv/client.py`，理解抓取逻辑

---

## 项目结构速览

```
.
├── src/
│   ├── core/              # 核心算法（Phase 1-2）
│   │   └── arxiv/         # arXiv 数据获取
│   ├── backend/           # FastAPI 后端（Phase 3）
│   ├── frontend/          # Streamlit 前端（Phase 4）
│   └── scripts/           # 命令行工具
│
├── data/                  # 数据存储
│   ├── raw/               # 原始 CSV ← 你的数据在这里
│   └── processed/         # 处理后数据
│
├── docs/                  # 文档
│   ├── MASTER_PLAN.md     # 完整学习计划
│   └── PHASE1_GUIDE.md    # Phase 1 详细指南
│
└── requirements.txt       # 依赖列表
```

---

## 获取帮助

- **阅读文档：** `docs/` 目录下有详细指南
- **查看代码注释：** 每个函数都有详细的 docstring
- **运行测试：** `pytest tests/`（Phase 5 会添加测试）

**准备好进入 Phase 2 了吗？** 阅读 `docs/PHASE2_GUIDE.md`（即将创建）！

