
import os
import json
import faiss
import numpy as np
from typing import List
from app.embedder import embed_texts

# المسارات الأساسية
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
INDEXES_DIR = os.path.join(BASE_DIR, "Indexes")


# --------------------------------------------
# 🧩 دالة استخراج القيم المتداخلة (Nested Keys)
# --------------------------------------------
def get_nested_value(data: dict, key_path: str):
    """يدعم الوصول إلى الحقول المتداخلة مثل brand.displayName أو reels.videoUrl"""
    keys = key_path.split(".")
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, "")
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                data = data[0].get(key, "")
            else:
                return " ".join(map(str, data))
        else:
            return ""
    return str(data) if data else ""


# --------------------------------------------
# ✂️ دالة تقسيم النصوص إلى أجزاء (Chunks)
# --------------------------------------------
def chunk_text(text: str, size: int = 300, overlap: int = 50) -> List[str]:
    """
    تقسيم النص إلى مقاطع صغيرة لتقليل فقدان المعنى أثناء توليد الـ embeddings
    """
    if not text:
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += size - overlap
    return chunks


# --------------------------------------------
# 🏗️ بناء فهرس FAISS لملف JSON محدد
# --------------------------------------------
def build_faiss_index_for_json(json_filename: str, index_name: str, text_fields=None):
    """
    يبني فهرس متجهي باستخدام FAISS من بيانات JSON.
    text_fields: قائمة الحقول التي سيتم تحويلها إلى نصوص (تدعم المفاتيح المتداخلة)
    """
    path = os.path.join(DATA_DIR, json_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ الملف غير موجود: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs, metas = [], []

    for item in data:
        # دمج الحقول المطلوبة
        if text_fields:
            combined_text = " \n ".join([get_nested_value(item, f) for f in text_fields])
        else:
            combined_text = " \n ".join([
                str(item.get(k, "")) for k in ["name", "description", "caption", "brand_name"] if item.get(k)
            ])

        # تقطيع النصوص الطويلة
        for i, chunk in enumerate(chunk_text(combined_text)):
            if chunk.strip():
                docs.append(chunk)
                metas.append({
                    "source_id": item.get("id"),
                    "chunk_index": i,
                    "original": item
                })

    if not docs:
        raise ValueError(f"⚠️ لم يتم العثور على نصوص يمكن فهرستها في {json_filename}")

    # 🧠 إنشاء المتجهات باستخدام embedder
    embeddings = embed_texts(docs)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # حفظ الفهرس والبيانات
    index_dir = os.path.join(INDEXES_DIR, index_name)
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"docs": docs, "metas": metas}, f, ensure_ascii=False, indent=2)

    print(f"✅ تم بناء الفهرس: {json_filename} -> {index_dir} (عدد المتجهات={len(docs)}, الأبعاد={dim})")

