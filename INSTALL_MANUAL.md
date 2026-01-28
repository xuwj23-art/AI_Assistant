# 手动安装指南（推荐）

如果自动脚本出现问题，请按以下步骤手动安装。

## 完整步骤

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

### 8. 运行 Phase 1

```bash
cd src
python -m scripts.fetch_and_save_arxiv --query "large language models" --max-results 100
```

---

## 常见问题

### Q: 第 4 步很慢怎么办？

**A:** 换成清华源：

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
```

然后重新运行第 4 步。

### Q: 第 6 步很慢怎么办？

**A:** 使用清华 pip 源：

```bash
pip install -r requirements-conda.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 验证时提示 ModuleNotFoundError？

**A:** 确认：

1. 环境已激活（命令行前有 `(literature_review)`）
2. 重新运行安装命令
3. 检查是否有错误信息

### Q: 如何重新安装？

```bash
# 1. 退出环境
conda deactivate

# 2. 删除环境
conda remove -n literature_review --all -y

# 3. 从第 3 步重新开始
```

---

## 完整命令（复制粘贴版）

```bash
# 进入目录
cd /d E:\AI_project\LiteratureReview_AIAssistant

# 创建环境
conda create -n literature_review python=3.10 -y

# 安装核心包
conda install -n literature_review -c conda-forge hdbscan umap-learn scikit-learn numpy pandas -y

# 激活环境
conda activate literature_review

# 安装其他包
pip install -r requirements-conda.txt

# 验证
python -c "import hdbscan, bertopic, fastapi, streamlit; print('All packages installed successfully!')"

# 运行 Phase 1
cd src
python -m scripts.fetch_and_save_arxiv --query "large language models" --max-results 100
```

---

**预计总时间：** 15-20 分钟（取决于网络速度）

**下一步：** 查看 `QUICKSTART.md` 继续学习！

