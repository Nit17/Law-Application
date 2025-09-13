import streamlit as st
import requests

BACKEND_URL = st.secrets.get("backend_url", "http://127.0.0.1:8000")

st.title("Law-Application — RAG Demo")

question = st.text_area("Question", value="What is negligence under Indian law?", height=120)
top_k = st.slider("Top-k contexts", 1, 8, 3)

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
