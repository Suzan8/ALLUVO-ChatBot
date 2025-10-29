import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#from app.index_builder import build_faiss_index_for_json

from sentence_transformers import SentenceTransformer
import numpy as np

# ✅ استخدم نموذج متعدد اللغات (يدعم العربية والإنجليزية)
# هذا النموذج من أفضل النماذج المجانية لفهم النصوص بعدة لغات
EMBED_MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"

# تحميل النموذج مرة واحدة عند بدء التطبيق
_model = SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(texts):
    """
    🔹 تُحوّل قائمة من النصوص إلى تمثيلات عددية (Embeddings)
    🔹 تستخدم النموذج المحدد أعلاه لفهم المعنى اللغوي حتى لو كانت بالعربية أو الإنجليزية
    """
    if not texts:
        # إذا كانت القائمة فارغة، أرجع مصفوفة فارغة بنفس أبعاد النموذج
        return np.array([], dtype="float32").reshape(0, _model.get_sentence_embedding_dimension())

    # تحويل النصوص إلى تمثيلات رقمية
    embeddings = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype("float32")

def embed_text(text: str):
    """
    🔹 تُحوّل نصًا واحدًا إلى تمثيل عددي
    """
    return embed_texts([text])[0]
