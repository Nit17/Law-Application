import streamlit as st
import requests

BACKEND_URL = st.secrets.get("backend_url", "http://127.0.0.1:8000")

st.title("Law-Application — RAG Demo")

question = st.text_area("Question", value="What is negligence under Indian law?", height=120)
top_k = st.slider("Top-k contexts", 1, 8, 3)

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("Ingest & Warm"):
        with st.spinner("Ingesting documents and warming embeddings..."):
            try:
                r1 = requests.post(f"{BACKEND_URL}/ingest/", timeout=30)
                r1.raise_for_status()
                ingest_resp = r1.json()
            except Exception as e:
                st.error(f"Ingest failed: {e}")
                ingest_resp = None

            try:
                # warm expects a body: {"queries": ["...", ...]}
                r2 = requests.post(f"{BACKEND_URL}/warm/", json={"queries": [question]}, timeout=60)
                r2.raise_for_status()
                warm_resp = r2.json()
            except Exception as e:
                st.error(f"Warm failed: {e}")
                warm_resp = None

        if ingest_resp is not None:
            st.success(f"Ingested {ingest_resp.get('count')} documents")
        if warm_resp is not None:
            st.success(f"Warmed: {warm_resp}")

with col2:
    if st.button("Check Health"):
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=5)
            r.raise_for_status()
            st.json(r.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")

if st.button("Generate"):
    with st.spinner("Calling backend..."):
        try:
            resp = requests.post(f"{BACKEND_URL}/generate/", json={"question": question, "top_k": top_k}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
        else:
            st.subheader("Answer")
            st.write(data.get("answer"))
            st.subheader("Model")
            st.write(data.get("model"))
            st.subheader("Contexts")
            for i, c in enumerate(data.get("contexts", []), start=1):
                st.markdown(f"**Context {i}:**\n\n{c}")
