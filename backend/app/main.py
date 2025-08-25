from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import ingest, query

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

@app.get("/")
def root():
    return {"status": "ok", "service": "legal-rag-backend"}
