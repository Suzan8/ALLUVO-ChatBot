import os
import json
from app.index_builder import build_faiss_index_for_json, make_product_text, make_brand_text, make_reel_text

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

# products
with open(os.path.join(DATA_DIR, "products.json"), "r", encoding="utf-8") as f:
    products_data = json.load(f)
build_faiss_index_for_json(products_data, "products_index", make_product_text)

# brands
with open(os.path.join(DATA_DIR, "brands.json"), "r", encoding="utf-8") as f:
    brands_data = json.load(f)
build_faiss_index_for_json(brands_data, "brands_index", make_brand_text)

# reels
with open(os.path.join(DATA_DIR, "reels.json"), "r", encoding="utf-8") as f:
    reels_data = json.load(f)
build_faiss_index_for_json(reels_data, "reels_index", make_reel_text)
