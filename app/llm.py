import os
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "")  # optional override


def _default_gemini_url(model: str):
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def call_gemini(prompt: str, max_tokens: int = 512, temperature: float = 0.7):
    if not GEMINI_API_KEY:
        raise EnvironmentError("⚠️ GEMINI_API_KEY is not set. Add it to your .env file.")

    url = GEMINI_API_URL.strip() or f"{_default_gemini_url(GEMINI_MODEL)}?key={GEMINI_API_KEY}"

    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"أنت مساعد ذكي لمتجر ALLUVO الإلكتروني. أجب فقط باللغة العربية بدقة واختصار.\n\n{prompt}"}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()

    data = resp.json()

   
    try:
        text = data["candidates"][0]["content"]["parts"][0].get("text", "").strip()
        if text:
            return text
    except Exception:
        pass

    try:
        alt = data.get("candidates", [{}])[0].get("output", "")
        if alt:
            return alt.strip()
    except Exception:
        pass

    return "⚠️ لم أستطع تحليل الرد من Gemini.\n" + str(data)


def generate_answer(context: str, user_query: str) -> str:
    prompt = (
        f"المعلومات التالية تم استرجاعها من قاعدة بيانات ALLUVO:\n\n{context}\n\n"
        f"بناءً على المعلومات أعلاه، أجب على السؤال التالي بدقة وباللغة العربية فقط: {user_query}\n"
        "- استخدم ردًا مختصرًا وواضحًا.\n"
        "- لا تشرح خطواتك.\n"
        "- إذا كان هناك أكثر من منتج، اذكر اسم البراند المشترك.\n"
        "- إذا لم تكن المعلومة موجودة، قل 'لا توجد معلومات في قاعدة البيانات'."
    )
    return call_gemini(prompt)
