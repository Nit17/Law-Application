# Backend: Legal RAG

This service provides a small retrieval-augmented generation (RAG) backend for legal Q&A.

Quick overview
- Ingest documents from `backend/data/*.txt` and index them using TF‑IDF (starter).
- Query the index to retrieve relevant passages.
- Generate answers using a local LLM (optional) with retrieved context.
- Optional dense-embedding support via `sentence-transformers` for improved retrieval.

Requirements
- Python 3.9+
- A virtual environment is recommended (project contains `.venv` in repo root).

Setup (recommended)

1. Activate the repo virtualenv (from repo root):

```bash
source .venv/bin/activate
```

2. Install core dependencies (FastAPI + minimal stack):

```bash
cd backend
pip install fastapi uvicorn scikit-learn joblib pydantic
```

3. Optional (recommended) - embeddings and LLMs

- For embeddings (dense retrieval):

```bash
pip install sentence-transformers
```

- For local LLM generation (transformers + torch):

```bash
pip install transformers accelerate torch
```

Notes on optional deps
- If `sentence-transformers` is not installed, embedding-based endpoints will return an explanatory error.
- If `transformers`/`torch` are not installed, the `/generate` and `/generate_stream` endpoints will return a 500 explaining how to install required packages.

Environment variables (optional)
- `LLM_MODEL` — HF model id to use (default in repo: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
- `LLM_DEVICE` — `auto|cpu|cuda|mps` (default `auto`).
- `LLM_MAX_INPUT_TOKENS`, `LLM_MAX_GENERATION_TOKENS`, `LLM_TEMPERATURE`, `LLM_TOP_P` — generation tuning.

Run the server

```bash
# from backend/ (while venv is active)
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Where data/index are stored
- Document files: `backend/data/*.txt` (sample files included)
- TF‑IDF index and metadata: `backend/app/index/` (files: `docs.json`, `texts.txt`, `vectorizer.joblib`, `matrix.joblib`)
- Embeddings (if computed): `backend/app/index/embeddings.joblib`

API endpoints

1) Ingest documents (creates/overwrites index files)

- POST /ingest/
- Response: `{ "count": <number_of_documents> }`

Example:

```bash
curl -X POST http://127.0.0.1:8000/ingest/
```

2) Query (TF‑IDF retrieval)

- POST /query/
- Body: `{ "question": "...", "k": 5 }`
- Response: `{ "hits": [ {"id":"...","score":0.9,"text":"...","meta":{}} ] }`

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/query/ -H "Content-Type: application/json" -d '{"question":"definition","k":3}'
```

3) Generate (LLM-backed answer using retrieved contexts)

- POST /generate/
- Body: `{ "question": "...", "top_k": 4 }`
- Response: `{ "answer": "...", "model": "...", "contexts": [...], "tokens_in": n, "tokens_out": n, "prompt_tokens": n }`
- Note: requires `transformers` + `torch` installed for local generation.

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/generate/ -H "Content-Type: application/json" -d '{"question":"What is the definition of negligence under Indian law?","top_k":3}'
```

4) Embed (compute dense embeddings for existing index texts)

- POST /embed/
- Body: `{ "force": false }` (set `force=true` to recompute)
- Response: `{ "count": <number_of_texts> }`
- Note: requires `sentence-transformers`.

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/embed/ -H "Content-Type: application/json" -d '{"force":false}'
```

5) Hybrid retrieval (TF‑IDF + embeddings)

- POST /hybrid/
- Body: `{ "question": "...", "k": 5 }`
- Response: same shape as `/query/` but uses a combined score when embeddings are present.

6) Warm (pre-run queries to warm caches / indexes)

- POST /warm/
- Body: `{ "queries": ["q1","q2"] }`
- Response: `{ "warmed": <count> }`

7) Generate stream (SSE-like streaming response)

- POST /generate_stream/
- Body: `{ "question": "...", "top_k": 3 }`
- Response: `text/event-stream` with incremental data events (basic non-incremental stub implemented).

Troubleshooting
- If you see 500 errors mentioning missing dependencies, install the optional extras described above.
- If your local Python executable is `python3` instead of `python`, use `python3 -m pip install ...` when following commands.
- For large LLMs, prefer using GPU with a compatible `torch` wheel and set `LLM_DEVICE=cuda`.

Development tips
- Use Postman or httpie for quick interactive testing. The app serves OpenAPI at `http://127.0.0.1:8000/docs` when running.
- To switch to an external vector DB (like FAISS or Pinecone) replace `backend/app/core/vector_store.py` and `embedding_store.py` with the desired backend implementation.

License & attribution
- See repository `LICENSE` for license details.



