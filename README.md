# Law-Application

Customised lightweight Retrieval-Augmented Generation (RAG) backend for Indian law. It indexes local legal text files and retrieves relevant snippets via a simple FastAPI service — a practical first step before exploring on-device or larger LLM deployments.

## What’s included
- FastAPI server with two endpoints: `POST /ingest` (build / rebuild index) and `POST /query` (retrieve relevant snippets)
- Lightweight TF‑IDF vector store (no heavy ML / GPU deps) stored under `backend/app/index/`
- Sample legal text in `backend/data/` to experiment quickly
- Persisted artifacts: `vectorizer.joblib`, `matrix.joblib`, raw `texts.txt`, and structured `docs.json`
- Documentation: project plan (`docs/project_plan.md`) and legal disclaimers (`docs/legal_disclaimers.md`)
- Placeholder `mobile-app/` folder for future client UI or offline features

## Tech stack (current)
Backend: FastAPI (Python)  |  Vectorization: scikit-learn `TfidfVectorizer`  |  Persistence: joblib + flat files

## Quick start
1. (Optional) Create & activate a virtual environment
2. Install backend dependencies: go to `backend/` and run your package installer (e.g. `pip install -e .` or `pip install -r requirements.txt` if you export one)
3. Run the FastAPI app (e.g. `uvicorn backend.app.main:app --reload` from repo root)
4. Ingest sample data: `POST /ingest`
5. Query: `POST /query {"query": "contract termination"}`

### Example session
```
uvicorn backend.app.main:app --reload
curl -X POST http://localhost:8000/ingest
curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{"query": "contract termination"}'
```

See `docs/project_plan.md` for detailed roadmap and next phases (improved retrieval, embeddings, evaluation, mobile integration).

## Roadmap snapshot
- Phase 1: Local TF‑IDF prototype (current)
- Phase 2: Add embeddings + hybrid search
- Phase 3: Response synthesis + citation traces
- Phase 4: Mobile client + offline packs

## Disclaimer
All content and generated outputs are for informational purposes only and are NOT legal advice. Consult a qualified professional for legal matters. See `docs/legal_disclaimers.md` for full disclaimer text.

## License
TBD (add a license file before public distribution).

---
Feel free to extend, refactor, or request features. PRs and issues welcome once licensing is finalized.
