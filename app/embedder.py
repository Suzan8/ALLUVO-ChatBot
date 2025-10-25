from sentence_transformers import SentenceTransformer
import numpy as np

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(texts):
    if not texts:
        return np.array([], dtype="float32").reshape(0, _model.get_sentence_embedding_dimension())
    embeddings = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype("float32")

def embed_text(text: str):
    return embed_texts([text])[0]
