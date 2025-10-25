# ALLUVO-ChatBot (RAG) - Ready Project (empty Data folder)

## Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot scaffold for Alluvo.
Data files should be placed in the Data/ folder (brands.json, products.json, reels.json).
Indexes folder will be created after running the index builder.

## Environment
Create `.env` or set environment variables:
- GEMINI_API_KEY : Your Google Gemini / Generative AI API key (or use ADC)
- GEMINI_MODEL : the Gemini model name, e.g. gemini-1.5-mini
- GEMINI_API_URL : optional custom endpoint (defaults to Google Generative Language REST pattern)

Example `.env`:
```
GEMINI_API_KEY=ya29....your_key_here
GEMINI_MODEL=gemini-1.5-mini
GEMINI_API_URL=
```

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build indexes
After adding your JSON files into `Data/`, run:
```bash
python scripts/build_all_indexes.py
```
This will create `Indexes/products_index`, `Indexes/brands_index`, `Indexes/reels_index`.

## Run server
```bash
uvicorn app.main:app --reload --port 8000
```
Open http://127.0.0.1:8000/docs

## Notes about Gemini API
The included `app/llm.py` attempts to call the Google Generative Language REST endpoints.
If you are using Vertex AI or a different endpoint, set GEMINI_API_URL to the correct URL in your .env.
The default request tries to POST to:
https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateMessage
If your endpoint differs, update GEMINI_API_URL.



ALLUVO-ChatBot/
├── app/
│   ├── main.py
│   ├── retriever.py
│   ├── llm.py
│   ├── rag_pipeline.py
│   ├── utils.py
│   ├── embedder.py
│   ├── index_builder.py
│   └── api/
│       └── routes.py
│
├── Data/
│   ├── brands.json
│   ├── reels.json
│   ├── products.json
│
├── Indexes/
│   ├── brands_index/
│   ├── reels_index/
│   ├── products_index/
│
├── scripts/
│   └── build_all_indexes.py
│
├── .env
├── README.md
└── requirements.txt
