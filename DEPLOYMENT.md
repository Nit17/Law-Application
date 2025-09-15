# Deployment Guide

This project has two parts:
- Backend (FastAPI): folder `backend/`
- Frontend (Streamlit): `frontend/streamlit_app.py`

You can deploy them separately: host the backend on a general PaaS/container host, and host the Streamlit UI on Streamlit Community Cloud.

## 1) Backend deployment (Docker image)

Build locally and push to a registry (Docker Hub, GHCR, etc.). From repo root:

```bash
docker build -t your-docker-user/law-backend:latest ./backend
docker push your-docker-user/law-backend:latest
```

Run on a VM:

```bash
docker run -p 8000:8000 \
  -e PYTHONUNBUFFERED=1 \
  -e LLM_DEVICE=cpu \
  your-docker-user/law-backend:latest
```

Optional: map a volume for persistent index files:

```bash
docker run -p 8000:8000 -v law_index:/app/app/index your-docker-user/law-backend:latest
```

Render deployment (Blueprint):

1. Push your repo to GitHub.
2. In Render, click "New +" → "Blueprint" and point it at your repo.
3. Render reads `render.yaml` and provisions the service:
  - Name: law-app-backend
  - Health check: /health
  - Plan: free (change as needed)
  - Persistent disk mounted at `/app/app/index` to keep index files
4. After deploy, copy the public URL (e.g., https://law-app-backend.onrender.com)

Fly.io example (Dockerfile):
- `fly launch` in `backend/`, expose port 8000, deploy

Security: restrict CORS in `backend/app/main.py` to your Streamlit domain for production.

## 2) Frontend deployment (Streamlit Community Cloud)

In Streamlit Cloud:
- Repository: this repo
- Branch: main
- Main file path: `frontend/streamlit_app.py`
- Python dependencies: auto-detected from `frontend/requirements.txt`

App secrets (App Settings → Secrets):

```toml
backend_url = "https://your-backend.example.com"
```

You can copy `frontend/.streamlit/secrets.toml.example` locally for reference.

## 3) End-to-end checklist

- Backend `/health` returns JSON with `status` ok/degraded.
- Streamlit secret `backend_url` is set and reachable over HTTPS.
- Try a query:
  - Open the Streamlit app, ensure Status shows green.
  - Ask a question using streaming; if issues, toggle off streaming to use `/generate`.

Quick share link

- Streamlit Cloud URL: shown on your app page (e.g., https://your-app.streamlit.app)
- Backend URL: the Render URL from step 1
- Share only the Streamlit Cloud URL with users; ensure the backend is up (free tier may sleep).

## 4) Troubleshooting

- 404/connection errors: confirm `backend_url` (no trailing slash), the backend is public, and CORS allows the Streamlit domain.
- 500 on `/generate`: install LLM deps or set a smaller `LLM_MODEL` and ensure sufficient RAM/CPU.
- Blank stream: some hosts buffer SSE; disable streaming in the UI or switch host.
