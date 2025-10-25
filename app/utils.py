import re

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"[^\w\s\u0600-\u06FF\-.,]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text

def safe_get(d: dict, key: str, default=""):
    return d.get(key, default) if isinstance(d, dict) else default
