import os
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "")  # optional override


def _default_gemini_url(model: str):
    # ✅ الصيغة الصحيحة لاستخدام واجهة generateContent الحديثة
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def call_gemini(prompt: str, max_tokens: int = 512, temperature: float = 0.7):
    if not GEMINI_API_KEY:
        raise EnvironmentError("⚠️ GEMINI_API_KEY is not set. Add it to your .env file.")

    # بناء رابط الطلب مع المفتاح في الـ URL
    url = GEMINI_API_URL.strip() or f"{_default_gemini_url(GEMINI_MODEL)}?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    # ✅ الصيغة الصحيحة لـ Gemini 2.5 REST API
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"You are an assistant for ALLUVO ecommerce site. Answer concisely in Arabic.\n\n{prompt}"}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        }
    }

    # إرسال الطلب
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()

    data = resp.json()

    # ✅ تحليل استجابة Gemini الحديثة
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return str(data)


def generate_answer(context: str, user_query: str) -> str:
    prompt = (
        f"المعلومات التالية تم استرجاعها من قاعدة بيانات ALLUVO:\n\n{context}\n\n"
        f"بناءً على المعلومات أعلاه، أجب على السؤال التالي بدقة وباللغة العربية: {user_query}\n"
        "- ابدأ برد قصير (1-3 جمل).\n"
        "- إن وُجد أكثر من منتج أعطِ جدول نصي: اسم | براند | سعر | رابط الصورة إن وُجد.\n"
        "- إن لم تكن المعلومات كافية اعترف واطلب توضيحًا."
    )
    return call_gemini(prompt)