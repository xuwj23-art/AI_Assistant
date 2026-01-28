from sentence_transformers import SentenceTransformer
from pandas as pd
from config import RAW_DATA_DIR
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-mpnet-base-v2")

file_path = RAW_DATA_DIR

df = pd.read_csv(file_path)

paper = df["abstract"]

embeddings = model.encode(paper)

similarity_matrix = cosine_similarity(embeddings)

print(similarity_matrix)