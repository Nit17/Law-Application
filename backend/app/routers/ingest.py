from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from ..core.vector_store import TfidfStore, Document

router = APIRouter()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INDEX_DIR = Path(__file__).resolve().parents[1] / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


class IngestResponse(BaseModel):
    count: int


@router.post("/")
def ingest() -> IngestResponse:
    # Read all .txt files under backend/data
    texts: List[Document] = []
    for p in sorted((DATA_DIR).glob("*.txt")):
        content = p.read_text(encoding="utf-8")
        texts.append(Document(id=p.name, text=content, meta={"path": str(p)}))

    store = TfidfStore(INDEX_DIR)
    store.add_texts(texts)
    store.build()

    return IngestResponse(count=len(texts))
