from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..core.llm import generate, GenerationResult
from ..core.vector_store import TfidfStore

router = APIRouter()
INDEX_DIR = __import__("pathlib").Path(__file__).resolve().parents[1] / "index"


class StreamRequest(BaseModel):
	question: str
	top_k: int = 3


@router.post("/")
def stream_generate(req: StreamRequest):
	q = req.question.strip()
	if not q:
		raise HTTPException(status_code=400, detail="Empty question")

	store = TfidfStore(INDEX_DIR)
	results = store.query(q, k=req.top_k)
	contexts = [d.text for d, _ in results]

	# This is a simple non-incremental stream; for real streaming attach to model's streamer
	gen = generate(q, contexts)

	def iter_func():
		# Send a small SSE-like stream
		yield f"data: {gen.completion}\n\n"

	return StreamingResponse(iter_func(), media_type="text/event-stream")

