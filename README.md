# ALLUVO-ChatBot
A custom-built AI chatbot designed for Alluvo, a platform showcasing brand content, pricing, and offers. This chatbot enhances user experience by instantly answering visitor questions about products, prices, and promotions. It helps users navigate the platform efficiently and make informed decisions.

ALLUVO-ChatBot/
├── app/
│   ├── main.py                     # FastAPI entry point
│   ├── retriever.py                # Handles data retrieval from JSON
│   ├── llm.py                      # Interface for AI models (OpenAI or Gemini)
│   ├── rag_pipeline.py             # Combines retrieval and generation (RAG)
│   ├── utils.py                    # Utility functions (e.g., cleaning, filtering)
│   └── api/
│       └── routes.py               # API routes (e.g., /chat)
│
├── Data/
│   ├── brands.json                 
│   ├── reels.json       
│   ├── products.json 
│
├── Indexes/
│   ├── brands_index/               # Vector index for brand data
│   ├── products_index/             # Vector index for product data
│   ├── reels_index/                # Vector index for reels data
│
├── LICENSE 
├── README.md
└── requirements.txt
