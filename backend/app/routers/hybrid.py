from __future__ import annotations
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..core.hybrid import HybridRetriever, HybridHit

router = APIRouter()
INDEX_DIR = Path(__file__).resolve().parents[1] / "index"

class HybridRequest(BaseModel):
    question: str
    k: int = 5
    alpha: float = 0.5  # weight for embeddings
    emb_model: str = "all-MiniLM-L6-v2"

class HybridResponseHit(BaseModel):
    id: str
    text: str
    meta: dict
    score_tfidf: float
    score_embed: float
    score_final: float

class HybridResponse(BaseModel):
    hits: List[HybridResponseHit]

@router.post("/", response_model=HybridResponse)
def hybrid_search(req: HybridRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")
    retriever = HybridRetriever(INDEX_DIR, alpha=req.alpha, emb_model_name=req.emb_model)
    try:
        hits = retriever.retrieve(q, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return HybridResponse(hits=[HybridResponseHit(**h.__dict__) for h in hits])
