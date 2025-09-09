from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.middleware import TokenUsageMiddleware
from .routers import ingest, query, generate, embed, generate_stream, hybrid, warm

app = FastAPI(title="Legal RAG Backend", version="0.1.0")

# Allow local dev and mobile emulator
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TokenUsageMiddleware)

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(generate.router, prefix="/generate", tags=["generate"])
app.include_router(embed.router, prefix="/embed", tags=["embed"])
app.include_router(generate_stream.router, prefix="/generate-stream", tags=["generate-stream"])
app.include_router(hybrid.router, prefix="/hybrid", tags=["hybrid"])
app.include_router(warm.router, prefix="/warm", tags=["warm"])

@app.get("/")
def root():
    return {"status": "ok", "service": "legal-rag-backend"}
