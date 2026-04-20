from sentence_transformers import SentenceTransformer

# 1. 加载模型（首次会自动下载）
print("加载模型...")
model = SentenceTransformer('all-mpnet-base-v2')
print("模型加载完成！")

# 2. 准备文本
sentences = [
    "深度学习是机器学习的一个分支",
    "深度学习属于机器学习领域",
    "今天天气真好"
]

# 3. 生成向量
print("\n生成向量...")
embeddings = model.encode(sentences)

# 4. 查看结果
print(f"\n向量形状: {embeddings.shape}")
print(f"第一个句子的向量（前10维）: {embeddings[0][:10]}")

# 5. 计算相似度
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)
print("\n相似度矩阵:")
print(similarity_matrix)
print(f"\n句子1和句子2的相似度: {similarity_matrix[0][1]:.4f}")
print(f"句子1和句子3的相似度: {similarity_matrix[0][2]:.4f}")