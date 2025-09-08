from __future__ import annotations
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ..core.vector_store import TfidfStore
from ..core import llm

router = APIRouter()
INDEX_DIR = Path(__file__).resolve().parents[1] / "index"

class StreamRequest(BaseModel):
    question: str
    top_k: int = 4

@router.post("/")
async def stream_generate(req: StreamRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")
    store = TfidfStore(INDEX_DIR)
    results = store.query(q, k=req.top_k)
    contexts = [d.text for d, _ in results]
    try:
        gen_iter = llm.stream_generate(q, contexts)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return StreamingResponse((chunk for chunk in gen_iter), media_type="text/plain")
