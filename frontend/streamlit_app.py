import os
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional

import requests
import streamlit as st


# Determine backend URL from Streamlit secrets or env; default to local
def get_backend_url() -> str:
    try:
        url = st.secrets.get("backend_url", None)  # type: ignore[attr-defined]
    except Exception:
        url = None
    return url or os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")


BACKEND_URL = get_backend_url()


st.set_page_config(page_title="Law RAG — Ask", layout="wide")
st.title("Ask a legal question")


@contextmanager
def _spinner(msg: str):
    with st.spinner(msg):
        yield


def get_health() -> Dict:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"status": "offline", "index_ready": False, "model_ready": False}


def call_generate(question: str, top_k: int) -> Optional[Dict]:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/generate/",
            json={"question": question, "top_k": top_k},
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def stream_generate(question: str, top_k: int) -> Generator[str, None, None]:
    """Yield chunks of assistant text by reading Server-Sent Events from backend.

    Backend emits lines like: 'data: <text>\n\n'. This function parses and yields the data content.
    """
    try:
        with requests.post(
            f"{BACKEND_URL}/generate_stream/",
            json={"question": question, "top_k": top_k},
            stream=True,
            timeout=90,
        ) as r:
            r.raise_for_status()
            buffer = ""
            for raw in r.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line[len("data:") :].strip()
                    buffer += data
                    # Yield incrementally if backend sends multiple chunks
                    yield data
            # Ensure final newline
            if buffer and not buffer.endswith("\n"):
                yield "\n"
    except Exception as e:
        st.error(f"Streaming failed: {e}")
        return


# Sidebar controls
with st.sidebar:
    st.header("Settings")
    st.caption(f"Backend: {BACKEND_URL}")
    health = get_health()
    status = health.get("status", "unknown")
    color = "green" if status in {"ok", "degraded"} else "red"
    st.markdown(f"Status: <span style='color:{color}'><strong>{status}</strong></span>", unsafe_allow_html=True)
    st.checkbox("Index ready", value=bool(health.get("index_ready")), disabled=True)
    st.checkbox("Model ready", value=bool(health.get("model_ready")), disabled=True)

    top_k = st.slider("Top K contexts", min_value=1, max_value=8, value=4)
    use_stream = st.toggle("Stream response", value=True)
    show_contexts = st.toggle("Show contexts", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []


# Initialize chat history
if "messages" not in st.session_state:
    # Initialize messages list in session state
    st.session_state.messages = []  # type: ignore[assignment]


# Render existing chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# Chat input
if prompt := st.chat_input("Type your legal question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant container
    with st.chat_message("assistant"):
        placeholder = st.empty()

        if use_stream:
            # Stream tokens
            acc = ""
            for chunk in stream_generate(prompt, top_k=top_k):
                acc += chunk
                placeholder.markdown(acc)
            answer_text = acc.strip()
            meta = None  # no metadata from stream endpoint today
            contexts = []
        else:
            with _spinner("Thinking..."):
                data = call_generate(prompt, top_k=top_k)
            if not data:
                st.stop()
            answer_text = data.get("answer", "")
            meta = {
                "model": data.get("model"),
                "tokens_in": data.get("tokens_in"),
                "tokens_out": data.get("tokens_out"),
                "prompt_tokens": data.get("prompt_tokens"),
            }
            contexts = data.get("contexts", [])
            placeholder.markdown(answer_text)

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer_text})

        # Show contexts and metadata
        if show_contexts and contexts:
            with st.expander("Show retrieved contexts"):
                for i, c in enumerate(contexts, start=1):
                    st.markdown(f"**Context {i}**")
                    st.write(c)
                    st.divider()

        if meta:
            st.caption(
                f"Model: {meta.get('model')} • Tokens in: {meta.get('tokens_in')} • Tokens out: {meta.get('tokens_out')} • Prompt tokens: {meta.get('prompt_tokens')}"
            )

