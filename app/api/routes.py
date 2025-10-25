from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.rag_pipeline import rag_answer

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    index: Optional[str] = "products_index"

@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    answer = rag_answer(req.query, top_k=req.top_k, index_name=req.index)
    return {"answer": answer}
