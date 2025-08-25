# Backend: Legal RAG

Endpoints
- POST /ingest — index the text files from `backend/data/*.txt`
- POST /query — body: `{ "question": "...", "k": 5 }`

Dev
- Install deps (Python >=3.9): use `uv` or `pip`. See root README for commands.
- Run server: `uvicorn app.main:app --reload` from `backend/`.

Notes
- TF‑IDF is a starter. Swap to dense embeddings later.
- Index is persisted under `backend/app/index/`.
