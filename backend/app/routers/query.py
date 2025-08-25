from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.vector_store import TfidfStore

router = APIRouter()

INDEX_DIR = Path(__file__).resolve().parents[1] / "index"


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class Passage(BaseModel):
    id: str
    score: float
    text: str
    meta: dict


class QueryResponse(BaseModel):
    hits: List[Passage]


@router.post("/")
def query(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    store = TfidfStore(INDEX_DIR)
    results = store.query(req.question, k=req.k)
    hits = [Passage(id=d.id, score=score, text=d.text, meta=d.meta) for d, score in results]
    return QueryResponse(hits=hits)
