Streamlit frontend for Law-Application

Install dependencies (in venv or separate):

```bash
pip install -r frontend/requirements.txt
```

Run the app:

```bash
streamlit run frontend/streamlit_app.py
```

The app will call the backend at `http://127.0.0.1:8000` by default. To change, set `secrets.toml` or modify `st.secrets["backend_url"]`.

---

Deploy to Streamlit Community Cloud

1) Push this repo to GitHub (public or private with access granted to Streamlit Cloud).

2) Create a new app on Streamlit Cloud and point it to:
	- Repository: this repo
	- Branch: main
	- Main file path: `frontend/streamlit_app.py`
	- Working directory: `frontend` (optional; not required if specifying full path above)

3) Python dependencies

	Streamlit Cloud will auto-install from `frontend/requirements.txt`. Ensure it includes:

	- streamlit
	- requests

4) Secrets and configuration

	In the app settings > Secrets, add:

	```toml
	# frontend/.streamlit/secrets.toml
	backend_url = "https://your-backend.example.com"
	```

	You can copy the example file at `frontend/.streamlit/secrets.toml.example`.

5) Backend deployment

	The Streamlit app needs a reachable backend (FastAPI) URL. Options:

	- Local dev: expose via `ngrok` or `cloudflared` and paste the public URL into `backend_url`.
	- Docker on a VM: open port 8000 and use a domain with HTTPS (recommended behind a reverse proxy like Caddy or Nginx).
	- PaaS: Deploy FastAPI on Fly.io/Render/Heroku/Azure Web Apps. Ensure CORS allows the Streamlit domain.

	Health check endpoint: `GET /health` should return status `ok` or `degraded` for the UI to display green.

6) Environment and CORS

	The backend already enables permissive CORS. For production, restrict `allow_origins` to your Streamlit app domain to improve security.

7) Troubleshooting

	- If the UI shows "offline", verify `backend_url` and that `/health` responds.
	- If responses hang with streaming enabled, try disabling "Stream response" in the sidebar (uses `/generate` instead of `/generate_stream`).
	- Check Streamlit app logs (Cloud dashboard) for exceptions.
