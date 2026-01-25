# 快速启动指南 (Quick Start)

## 前置要求

- Python 3.10 或更高版本
- Windows 10/11（其他操作系统请相应调整命令）
- 稳定的网络连接（用于下载依赖和调用 arXiv API）

---

## 第一步：环境准备（5 分钟）

### 1. 克隆/打开项目

```powershell
cd E:\AI_project\LiteratureReview_AIAssistant
```

### 2. 创建虚拟环境

```powershell
python -m venv .venv
```

### 3. 激活虚拟环境

```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1

# 如果遇到权限错误，先运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

你应该看到命令行前面出现 `(.venv)` 提示符。

### 4. 升级 pip

```powershell
python -m pip install --upgrade pip
```

### 5. 安装依赖

```powershell
pip install -r requirements.txt
```

⏰ **预计时间：** 5-10 分钟（取决于网络速度）

> **提示：** 如果下载速度很慢，可以使用清华镜像源：
> ```powershell
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 第二步：配置环境变量（1 分钟）

### 1. 复制环境变量模板

```powershell
copy .env.example .env
```

### 2. 编辑 `.env` 文件

用任何文本编辑器打开 `.env`，填入你的 OpenAI API Key：

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

> **注意：** Phase 1 和 Phase 2 不需要 OpenAI API Key，可以暂时跳过此步骤。

---

## 第三步：运行第一个脚本（2 分钟）

### 从 arXiv 抓取论文

```powershell
cd src
python -m scripts.fetch_and_save_arxiv --query "large language models" --max-results 100
```

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

```powershell
# 确保在项目根目录
cd ..  # 如果你在 src/ 下

# 启动 Jupyter
jupyter notebook
```

然后打开 `notebooks/01_explore_arxiv.ipynb`（需要先创建）。

---

## 常见问题排查

### 问题 1：`ModuleNotFoundError: No module named 'arxiv'`

**原因：** 虚拟环境未激活或依赖未安装。

**解决：**
```powershell
.\.venv\Scripts\Activate.ps1
pip install arxiv
```

### 问题 2：`Permission Denied` 运行 PowerShell 脚本

**原因：** Windows 默认禁止运行未签名脚本。

**解决：**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 问题 3：arXiv API 请求超时

**原因：** 网络问题或 arXiv 服务器繁忙。

**解决：**
- 减少 `--max-results`（例如改为 50）
- 稍后重试
- 检查网络连接

### 问题 4：pip 安装速度很慢

**解决：** 使用国内镜像源
```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 下一步

✅ **恭喜！** 你已经完成了 Phase 1 的第一部分。

**接下来你可以：**

1. **深入学习 Phase 1：** 阅读 `docs/PHASE1_GUIDE.md`
2. **查看完整计划：** 阅读 `docs/MASTER_PLAN.md`
3. **尝试不同关键词：**
   ```powershell
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

