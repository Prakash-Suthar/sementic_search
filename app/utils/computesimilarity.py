# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# def compute_similarity(text1: str, text2: str, model) -> float:
#     # Ensure they are encoded separately
#     emb1 = model.encode(text1, convert_to_tensor=False)
#     emb2 = model.encode(text2, convert_to_tensor=False)

#     # Compute cosine similarity manually
#     sim_score = cosine_similarity(
#         np.array(emb1).reshape(1, -1), 
#         np.array(emb2).reshape(1, -1)
#     )[0][0]

#     return {"similarity_score": round(float(sim_score), 3)}


import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityCache:
    def __init__(self, csv_path: str):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cache = {}
        self.load_data(csv_path)

    def load_data(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.head(50)  # Limit to first 50 for speed (optional)
        
        texts1 = df["text1"].astype(str).tolist()
        texts2 = df["text2"].astype(str).tolist()

        embeddings1 = self.model.encode(texts1, convert_to_tensor=True)
        embeddings2 = self.model.encode(texts2, convert_to_tensor=True)

        similarity_scores = cosine_similarity(embeddings1.cpu(), embeddings2.cpu()).diagonal()

        # Create cache with tuple key
        for t1, t2, score in zip(texts1, texts2, similarity_scores):
            key = (t1.strip(), t2.strip())
            self.cache[key] = round(float(score), 3)

    def get_similarity(self, text1: str, text2: str) -> float:
        return self.cache.get((text1.strip(), text2.strip()), -1.0)  # Return -1.0 if not found
