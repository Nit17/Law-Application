from __future__ import annotations
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import logging
from ..core.vector_store import TfidfStore
from ..core import llm

logger = logging.getLogger("generate")

router = APIRouter()
INDEX_DIR = Path(__file__).resolve().parents[1] / "index"

class GenerateRequest(BaseModel):
    question: str
    top_k: int = 4

class GenerateResponse(BaseModel):
    answer: str
    model: str
    contexts: List[str]
    tokens_in: int
    tokens_out: int
    prompt_tokens: int

@router.post("/")
def generate_answer(req: GenerateRequest) -> GenerateResponse:
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    # Retrieve
    store = TfidfStore(INDEX_DIR)
    results = store.query(q, k=req.top_k)
    contexts = [d.text for d, _ in results]

    try:
        gen = llm.generate(q, contexts)
    except RuntimeError as e:
        # Likely missing deps
        raise HTTPException(status_code=500, detail=str(e))

    return GenerateResponse(
        answer=gen.completion,
        model=gen.model,
        contexts=contexts,
        tokens_in=gen.tokens_in,
        tokens_out=gen.tokens_out,
        prompt_tokens=gen.usage["prompt_tokens"],
    )
