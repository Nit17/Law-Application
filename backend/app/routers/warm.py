from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel
from ..core import llm
from ..core.vector_store import TfidfStore
from ..core.embedding_store import EmbeddingStore

router = APIRouter()
INDEX_DIR = Path(__file__).resolve().parents[1] / "index"

class WarmResponse(BaseModel):
    model: str
    backend: str
    device: str
    loaded_embeddings: bool

@router.post("/")
def warm(embeddings: bool = False) -> WarmResponse:
    # trigger model load (if transformers) or ensure config accessible
    try:
        cfg = llm.get_config()
        # generating dummy prompt to force load without full generation (tokenizer + weights)
        if cfg["backend"] != "llamacpp":
            from backend.app.core.llm import _load_model  # type: ignore
            _load_model()  # cached
    except Exception:
        cfg = llm.get_config()

    emb_loaded = False
    if embeddings:
        try:
            store = EmbeddingStore(INDEX_DIR)
            # attempt a cheap load; will raise if missing
            store._load()  # type: ignore
            emb_loaded = store.embeddings is not None
        except Exception:
            emb_loaded = False

    return WarmResponse(model=cfg["model"], backend=cfg["backend"], device=cfg["device"], loaded_embeddings=emb_loaded)
