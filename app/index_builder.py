import os
import json
import re
import faiss
import numpy as np
from typing import List
from app.embedder import embed_texts  # ✅ الدالة الصحيحة من embedder.py

# المسارات الأساسية
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
INDEXES_DIR = os.path.join(BASE_DIR, "Indexes")


# --------------------------------------------
# 🧩 دالة استخراج القيم المتداخلة
# --------------------------------------------
def get_nested_value(data: dict, key_path: str):
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
# ✂️ دالة تقسيم النصوص
# --------------------------------------------
def chunk_text(text: str, size: int = 300, overlap: int = 50) -> List[str]:
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
# 🏗️ دالة بناء فهرس عام لأي نوع بيانات
# --------------------------------------------
def build_faiss_index_for_json(data_list, index_name, text_function):
    """
    data_list: قائمة البيانات (list of dicts)
    index_name: اسم الفهرس الناتج (مثل products_index)
    text_function: دالة لتحويل كل عنصر إلى نص
    """
    docs, metas = [], []

    for i, item in enumerate(data_list):
        combined_text = text_function(item)
        for j, chunk in enumerate(chunk_text(combined_text)):
            if chunk.strip():
                docs.append(chunk)
                metas.append({
                    "source_id": item.get("id", i),
                    "chunk_index": j,
                    "original": item
                })

    if not docs:
        raise ValueError(f"⚠️ لم يتم العثور على نصوص يمكن فهرستها في {index_name}")

    embeddings = embed_texts(docs)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # إنشاء مجلد الفهرس
    index_dir = os.path.join(INDEXES_DIR, index_name)
    os.makedirs(index_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"docs": docs, "metas": metas}, f, ensure_ascii=False, indent=2)

    print(f"✅ تم بناء الفهرس: {index_name} (عدد المقاطع={len(docs)}, أبعاد={dim})")


# --------------------------------------------
# 🧱 1. فهرس المنتجات Products
# --------------------------------------------
def make_product_text(item):
    text = f"""
    اسم المنتج: {item.get('name', '')}.
    الوصف: {item.get('description', '')}.
    الفئة: {item.get('category', '')}.
    السعر: {item.get('price', '')}.
    نسبة الخصم: {item.get('discountPercentage', '')}%.
    إمكانية التخصيص: {'نعم' if item.get('isCustomizable', False) else 'لا'}.
    """

    brand = item.get("brand", {})
    if brand:
        text += f"""
        البراند: {brand.get('displayName', '')}.
        وصف البراند: {brand.get('description', '')}.
        حالة التوثيق: {brand.get('verificationStatus', '')}.
        """

    reels = item.get("reels", [])
    for reel in reels:
        text += f"""
        فيديو ترويجي بعدد لايكات {reel.get('numOfLikes', 0)} 
        ومشاهدات {reel.get('numOfWatches', 0)}.
        رابط الفيديو: {reel.get('videoUrl', '')}.
        """

    return text


# --------------------------------------------
# 🧱 2. فهرس البراندات Brands
# --------------------------------------------
def make_brand_text(item):
    html_text = item.get("returnPolicyAsHtml", "")
    clean_policy = re.sub(r"<[^>]+>", " ", html_text)
    clean_policy = re.sub(r"\s+", " ", clean_policy).strip()

    text = f"""
    اسم البراند: {item.get('displayName', '')}.
    الوصف: {item.get('description', '')}.
    حالة التوثيق: {item.get('verificationStatus', '')}.
    شعار البراند: {item.get('logoUrl', '')}.
    سياسة الاسترجاع: {clean_policy}.
    """

    products = item.get("products", [])
    if products:
        text += f"\nيحتوي البراند على {len(products)} منتجًا:\n"
        for product in products:
            text += f"""
            🔸 المنتج: {product.get('name', '')}
            الوصف: {product.get('description', '')}
            الفئة: {product.get('category', '')}
            السعر: {product.get('price', '')}
            نسبة الخصم: {product.get('discountPercentage', 0)}%
            إمكانية التخصيص: {'نعم' if product.get('isCustomizable', False) else 'لا'}.
            """

            reels = product.get("reels", [])
            if reels:
                text += f"\n📹 عدد الريلز: {len(reels)}\n"
                for reel in reels:
                    text += f"""
                    عدد الإعجابات: {reel.get('numOfLikes', 0)}
                    عدد المشاهدات: {reel.get('numOfWatches', 0)}
                    رابط الفيديو: {reel.get('videoUrl', '')}
                    """

    return text


# --------------------------------------------
# 🧱 3. فهرس الريلز Reels
# --------------------------------------------
def make_reel_text(item):
    brand = item.get("brand", {})
    product = item.get("product", {})

    text = f"""
    🎥 فيديو ترويجي
    عدد الإعجابات: {item.get('numOfLikes', 0)}.
    عدد المشاهدات: {item.get('numOfWatches', 0)}.
    رابط الفيديو: {item.get('videoUrl', '')}.
    المنتج: {product.get('name', '')}.
    البراند: {brand.get('displayName', '')}.
    """
    return text


# --------------------------------------------
# 🚀 التنفيذ الرئيسي
# --------------------------------------------
if __name__ == "__main__":
    # 🛍️ المنتجات
    products_path = os.path.join(DATA_DIR, "products.json")
    if os.path.exists(products_path):
        with open(products_path, "r", encoding="utf-8") as f:
            products_data = json.load(f)
        build_faiss_index_for_json(products_data, "products_index", make_product_text)

    # 🏷️ البراندات
    brands_path = os.path.join(DATA_DIR, "brands.json")
    if os.path.exists(brands_path):
        with open(brands_path, "r", encoding="utf-8") as f:
            brands_data = json.load(f)
        build_faiss_index_for_json(brands_data, "brands_index", make_brand_text)

    # 🎥 الريلز
    reels_path = os.path.join(DATA_DIR, "reels.json")
    if os.path.exists(reels_path):
        with open(reels_path, "r", encoding="utf-8") as f:
            reels_data = json.load(f)
        build_faiss_index_for_json(reels_data, "reels_index", make_reel_text)

    print("\n🎯 تم بناء جميع الفهارس بنجاح!")
