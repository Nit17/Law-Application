from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

import logging
from ..core.vector_store import TfidfStore
from ..core.embedding_store import EmbeddingStore

logger = logging.getLogger("warm")

router = APIRouter()
INDEX_DIR = Path(__file__).resolve().parents[1] / "index"


class WarmRequest(BaseModel):
	queries: List[str]


class WarmResponse(BaseModel):
	warmed: int


@router.post("/")
def warm(req: WarmRequest) -> WarmResponse:
	# Ensure indexes exist
	tf = TfidfStore(INDEX_DIR)
	emb = EmbeddingStore(INDEX_DIR)
	count = 0
	for q in req.queries:
		_ = tf.query(q, k=3)
		_ = emb.search(q, k=3)
		count += 1
	return WarmResponse(warmed=count)

