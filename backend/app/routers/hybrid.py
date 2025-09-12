from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.vector_store import TfidfStore
from ..core.hybrid import HybridRetriever

router = APIRouter()
INDEX_DIR = Path(__file__).resolve().parents[1] / "index"


class HybridRequest(BaseModel):
	question: str
	k: int = 5


class Hit(BaseModel):
	id: str
	score: float
	text: str
	meta: dict


class HybridResponse(BaseModel):
	hits: List[Hit]


@router.post("/")
def hybrid(req: HybridRequest) -> HybridResponse:
	q = req.question.strip()
	if not q:
		raise HTTPException(status_code=400, detail="Empty question")

	tf = TfidfStore(INDEX_DIR)
	hy = HybridRetriever(INDEX_DIR)
	results = hy.query(q, k=req.k)
	hits = [Hit(id=d.id, score=score, text=d.text, meta=d.meta) for d, score in results]
	return HybridResponse(hits=hits)

