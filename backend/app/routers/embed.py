from __future__ import annotations
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..core.embedding_store import EmbeddingStore, EmbeddingDocument

router = APIRouter()
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INDEX_DIR = Path(__file__).resolve().parents[1] / "index"

class EmbedBuildResponse(BaseModel):
    count: int
    model: str

class EmbedQueryRequest(BaseModel):
    question: str
    k: int = 5

class EmbedHit(BaseModel):
    id: str
    score: float
    text: str
    meta: dict

class EmbedQueryResponse(BaseModel):
    hits: List[EmbedHit]

@router.post("/build", response_model=EmbedBuildResponse)
def build_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    docs: List[EmbeddingDocument] = []
    for p in sorted(DATA_DIR.glob("*.txt")):
        docs.append(EmbeddingDocument(id=p.name, text=p.read_text(encoding="utf-8"), meta={"path": str(p)}))
    store = EmbeddingStore(INDEX_DIR, model_name=model_name)
    try:
        store.add(docs)
        store.build()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return EmbedBuildResponse(count=len(docs), model=model_name)

@router.post("/query", response_model=EmbedQueryResponse)
def query_embeddings(req: EmbedQueryRequest):
    store = EmbeddingStore(INDEX_DIR)
    try:
        results = store.query(req.question, k=req.k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    hits = [EmbedHit(id=d.id, score=score, text=d.text, meta=d.meta) for d, score in results]
    return EmbedQueryResponse(hits=hits)
