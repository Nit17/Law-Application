# Project Plan (Practical RAG First)

Phases

1. RAG Prototype (this repo)
   - Ingest local text, TF‑IDF retrieval, FastAPI endpoints.
   - Swap TF‑IDF with a dense model later (e.g., sentence-transformers) when ready.
   - Add citation snippets in answers.

2. Cloud LLM completion
   - Add an LLM gateway (OpenAI, Gemini, or local server Llama via llama.cpp/ollama) and stitch retrieved context into prompts.
   - Add rate limiting and logging.

3. Data pipeline
   - Scrapers for public judgments/acts, dedupe, clean, and normalize citations.
   - Build evaluation sets (Q&A pairs) and a small golden set.

4. Mobile app (client)
   - Build a React Native minimal chat UI that calls the backend.
   - Add first-launch disclaimer screen and an About page.

5. On-device exploration (optional later)
   - Quantize a small model (e.g., 3B–8B) to GGUF and test on-device runtimes.
   - Gate large downloads via Wi‑Fi only; checksum verification.

Risk & compliance

- Prominent disclaimer; do not present outputs as legal advice.
- Respect source copyrights and licenses; store URLs.
- PII handling and logs redaction.
