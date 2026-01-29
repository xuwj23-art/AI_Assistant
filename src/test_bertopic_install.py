# test_bertopic_install.py
"""
验证 BERTopic 环境配置
"""
# import os
# os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'E:/AI_project/models/sentence_transformers'

print("=" * 70)
print("验证 BERTopic 环境")
print("=" * 70)

# 测试 1：导入核心库
print("\n[测试 1] 导入核心库...")
try:
    import bertopic
    print(f"  [success] bertopic {bertopic.__version__}")
except ImportError as e:
    print(f"  [failed] bertopic 导入失败: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print(f"  [success] sentence-transformers 已安装")
except ImportError as e:
    print(f"  [failed] sentence-transformers 导入失败: {e}")

try:
    import umap
    print(f"  [success] umap-learn 已安装")
except ImportError as e:
    print(f"  [failed] umap-learn 导入失败: {e}")

try:
    import hdbscan
    print(f"  [success] hdbscan 已安装")
except ImportError as e:
    print(f"  [failed] hdbscan 导入失败: {e}")

try:
    import plotly
    print(f"  [success] plotly {plotly.__version__}")
except ImportError as e:
    print(f"  [failed] plotly 导入失败: {e}")

# 测试 2：创建简单的 BERTopic 模型
print("\n[测试 2] 创建 BERTopic 模型...")
try:
    from bertopic import BERTopic
    model = BERTopic()
    print("  [success] BERTopic 模型创建成功")
except Exception as e:
    print(f"  [failed] BERTopic 模型创建失败: {e}")

# 测试 3：加载预训练模型（会触发下载）
print("\n[测试 3] 加载 Sentence Transformer 模型...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')
    print("  [success] 模型加载成功")
    
    # 测试编码
    test_text = "This is a test sentence."
    embedding = model.encode(test_text)
    print(f"  [success] 文本编码成功，向量维度: {len(embedding)}")
except Exception as e:
    print(f"  [failed] 模型加载失败: {e}")

# 测试 4：简单的主题建模测试
print("\n[测试 4] 简单主题建模测试...")
try:
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    import numpy as np
    
    # 扩展测试数据（至少需要 20-30 个样本才能正常聚类）
    docs = [
        "I love programming in Python",
        "Python is great for data science",
        "Machine learning is fascinating",
        "Deep learning uses neural networks",
        "Neural networks are powerful tools",
        "I enjoy cooking Italian food",
        "Pizza is my favorite dish",
        "Pasta is delicious",
        "JavaScript is used for web development",
        "React is a popular frontend framework",
        "Vue.js is easy to learn",
        "I like watching movies",
        "Science fiction films are interesting",
        "Action movies are exciting",
        "Data analysis with pandas is efficient",
        "NumPy is essential for numerical computing",
        "Matplotlib creates beautiful visualizations",
        "Natural language processing is important",
        "BERT revolutionized NLP tasks",
        "Transformers are state-of-the-art models",
    ]
    
    # 🔧 配置适合小数据集的参数
    umap_model = UMAP(
        n_components=2,       # 降到 2 维（便于小数据集）
        n_neighbors=5,        # 减少邻居数（原默认 15）
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=2,   # 最小簇大小（原默认 10）
        min_samples=1,        # 最小样本数
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=False
    )
    topics, probs = topic_model.fit_transform(docs)
    
    print(f"  [success] 主题建模成功")
    print(f"  发现 {len(set(topics))} 个主题")
    print(f"  主题分布: {dict(zip(*np.unique(topics, return_counts=True)))}")
    
except Exception as e:
    print(f"  [failed] 主题建模失败: {e}")
    import traceback
    print(f"  详细错误信息:")
    traceback.print_exc()

print("\n" + "=" * 70)
print("环境验证完成！")
print("=" * 70)