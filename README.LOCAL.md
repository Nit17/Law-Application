# Local Run Instructions (short)

These commands assume you're on macOS, using the project's venv at `./.venv` and working from the repository root.

1) Activate the project virtualenv (if you haven't created it, create and activate one):

```bash
# create venv (only if you don't have it already)
python -m venv .venv
# activate
source .venv/bin/activate
```

2) Install backend & frontend deps:

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

3) Start the backend (FastAPI / uvicorn):

```bash
# from repo root
source .venv/bin/activate
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info
```

4) Start the Streamlit UI (in a separate terminal):

```bash
source .venv/bin/activate
streamlit run frontend/streamlit_app.py --server.port 8501
# open http://localhost:8501 in your browser
```

5) Quick smoke tests (can be run from another terminal while servers run):

```bash
# ingest and warm
curl -X POST http://127.0.0.1:8000/ingest/
curl -X POST -H "Content-Type: application/json" -d '{"queries":["What is negligence under Indian law?"]}' http://127.0.0.1:8000/warm/

# query
curl -s -X POST -H "Content-Type: application/json" -d '{"question":"What is negligence under Indian law?","top_k":3}' http://127.0.0.1:8000/query/ | jq

# generate
curl -s -X POST -H "Content-Type: application/json" -d '{"question":"What is negligence under Indian law?","top_k":3}' http://127.0.0.1:8000/generate/ | jq
```

Notes & troubleshooting:
- If `uvicorn` reports "address already in use", find/kill the process on port 8000: `lsof -i :8000` then `kill <PID>`.
- If the Streamlit app doesn't load, check `/tmp/law_app_streamlit.log` for errors.
- The `/warm/` endpoint expects a JSON body: `{ "queries": ["...", ...] }`.
- To use a remote OpenAI backend instead of the local HF model, set `OPENAI_API_KEY` in your environment.

