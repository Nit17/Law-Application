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
