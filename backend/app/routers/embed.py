from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.embedding_store import EmbeddingStore, TextRecord

router = APIRouter()

INDEX_DIR = Path(__file__).resolve().parents[1] / "index"


class EmbedRequest(BaseModel):
	force: bool = False


class EmbedResponse(BaseModel):
	count: int


@router.post("/")
def embed(req: EmbedRequest) -> EmbedResponse:
	store = EmbeddingStore(INDEX_DIR)
	try:
		n = store.build(force=req.force)
	except RuntimeError as e:
		raise HTTPException(status_code=500, detail=str(e))
	return EmbedResponse(count=n)

