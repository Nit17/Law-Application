import os
import streamlit as st
import requests

# determine backend URL
try:
    BACKEND_URL = st.secrets.get("backend_url", None)
except Exception:
    BACKEND_URL = None
if not BACKEND_URL:
    BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Law RAG — Ask", layout="centered")
st.title("Ask a legal question")

question = st.text_input("Your question", value="What is negligence under Indian law?")
top_k = 3

if st.button("Ask"):
    if not question.strip():
        st.error("Please enter a question")
    else:
        with st.spinner("Generating answer..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/generate/", json={"question": question, "top_k": top_k}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                data = None

        if data:
            st.subheader("Answer")
            st.write(data.get("answer"))
            st.subheader("Contexts")
            for i, c in enumerate(data.get("contexts", []), start=1):
                st.markdown(f"**Context {i}:**\n\n{c}")
