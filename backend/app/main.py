from fastapi import FastAPI
from pathlib import Path
import os

from .core import llm as _llm
from fastapi.middleware.cors import CORSMiddleware
from .routers import ingest, query, generate, embed, hybrid, warm, generate_stream

app = FastAPI(title="Legal RAG Backend", version="0.1.0")

# Allow local dev and mobile emulator
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(generate.router, prefix="/generate", tags=["generate"])
app.include_router(embed.router, prefix="/embed", tags=["embed"])
app.include_router(hybrid.router, prefix="/hybrid", tags=["hybrid"])
app.include_router(warm.router, prefix="/warm", tags=["warm"])
app.include_router(generate_stream.router, prefix="/generate_stream", tags=["generate_stream"])

@app.get("/")
def root():
    return {"status": "ok", "service": "legal-rag-backend"}


@app.get("/health")
def health():
    """Health endpoint for readiness checks.

    - index_ready: True if any files exist under `app/index`
    - model_ready: True if OPENAI_API_KEY is set or local HF tokenizer is available
    """
    idx = Path(__file__).resolve().parents[1] / "index"
    index_ready = idx.exists() and any(idx.iterdir())
    # model readiness: either OpenAI API key present or HF tokenizer available
    model_ready = bool(os.getenv("OPENAI_API_KEY")) or (getattr(_llm, "AutoTokenizer", None) is not None)
    status = "ok" if index_ready and model_ready else "degraded" if index_ready else "starting"
    return {"status": status, "index_ready": index_ready, "model_ready": model_ready}
