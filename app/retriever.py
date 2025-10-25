import os, json, faiss, numpy as np
from app.embedder import embed_text

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INDEXES_DIR = os.path.join(BASE_DIR, "Indexes")

class FaissRetriever:
    def __init__(self, index_name: str):
        self.index_dir = os.path.join(INDEXES_DIR, index_name)
        self.index_path = os.path.join(self.index_dir, "index.faiss")
        self.meta_path = os.path.join(self.index_dir, "meta.json")

        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Index or meta.json not found for {index_name} (expected at {self.index_dir})")

        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.docs = meta.get("docs", [])
            self.metas = meta.get("metas", [])

    def search(self, query: str, top_k: int = 5):
        q_emb = embed_text(query).astype("float32")
        q_emb = np.expand_dims(q_emb, axis=0)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.docs):
                results.append({
                    "score": float(dist),
                    "doc": self.docs[idx],
                    "meta": self.metas[idx]
                })
        return results
