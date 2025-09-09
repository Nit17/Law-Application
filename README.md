# Law-Application

Customised open-source legal Retrieval-Augmented Generation (RAG) backend for Indian law with optional local LLM generation and LoRA fine‑tuning scaffold.

## Features
- FastAPI endpoints:
  - `POST /ingest/` – build TF‑IDF index from `backend/data/*.txt`
  - `POST /query/` – retrieve top‑k relevant passages
  - `POST /generate/` – retrieve + generate answer with local OSS LLM (TinyLlama by default)
  - `POST /embed/build` – build sentence-transformer embedding index (optional)
  - `POST /embed/query` – embedding similarity search (optional)
  - `POST /generate-stream/` – streaming token generation (transformers backend)
  - `POST /hybrid/` – hybrid TF‑IDF + embedding score fusion
  - `POST /warm/` – preload model (and optionally embeddings)
- Lightweight TF‑IDF vector store (scikit‑learn)
- Pluggable LLM abstraction (`backend/app/core/llm.py`) – future backends (llama.cpp, vLLM) can be added
- Optional LoRA fine‑tuning script (`backend/scripts/finetune_lora.py`)
- Documentation & disclaimers

## Tech stack
FastAPI, scikit‑learn (retrieval), Transformers (optional for LLM), PEFT (optional for LoRA)

## Setup
```bash
cd backend
pip install -e .            # base (retrieval only)
pip install -e .[llm]       # add local LLM + fine-tune deps
```

## Running the API
```bash
uvicorn backend.app.main:app --reload
```
Visit: http://localhost:8000/docs

### 1. Ingest
```bash
curl -X POST http://localhost:8000/ingest/
```

### 2. Query
```bash
curl -X POST http://localhost:8000/query/ -H 'Content-Type: application/json' \
  -d '{"question": "What is force majeure?"}'
```

### 3. Generate (retrieval + LLM answer)
```bash
curl -X POST http://localhost:8000/generate/ -H 'Content-Type: application/json' \
  -d '{"question": "Explain arbitration clause basics"}'
```
If LLM deps missing, endpoint returns 500 with guidance.

## Environment variables (optional)
`LLM_MODEL` (default TinyLlama/TinyLlama-1.1B-Chat-v1.0) – for transformers OR path to GGUF for llama.cpp
`LLM_BACKEND` (default transformers) – one of: transformers | llamacpp
`LLM_LORA_ADAPTER` (optional) – path to a PEFT LoRA adapter directory
`LLM_LORA_MERGE` (0/1) – if 1 merges adapter weights into base model in memory
`LLM_MAX_INPUT_TOKENS` (default 2048)
`LLM_MAX_GENERATION_TOKENS` (default 256)
`LLM_TEMPERATURE` (default 0.4)
`LLM_TOP_P` (default 0.95)

Example:
```bash
export LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
```

## LoRA Fine‑tuning (SFT) Scaffold
Dataset format: JSONL lines with keys: instruction, context (optional), response.
```bash
python -m backend.scripts.finetune_lora \
  --dataset backend/data/instruct_train.jsonl \
  --output-dir models/lora/run1 \
  --epochs 1 --batch-size 2
```
Adapters saved under `models/lora/run1`.

### Using a LoRA adapter at inference
```bash
export LLM_LORA_ADAPTER=models/lora/run1
uvicorn backend.app.main:app --reload
```
To merge adapter weights permanently:
```bash
python -m backend.scripts.merge_lora_adapter \
  --base TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter models/lora/run1 \
  --out models/merged/tinyllama-run1
export LLM_MODEL=models/merged/tinyllama-run1
```

## Additional Endpoints
- `/embed/build`, `/embed/query` – dense embeddings
- `/hybrid/` – fused ranking
- `/warm/` – warm start (model + embeddings load)
- `/generate-stream/` – streaming tokens

## Logging
Token/request metadata appended to `backend/logs/token_usage.jsonl` by middleware.

## Extending Models
- Add new backend: create `core/llm_<backend>.py` and dispatch in env var
- Add custom embed model: build with `embed/build?model_name=<hf_model>`
- Add streaming improvements: implement callback for llama.cpp

## Roadmap
- Reranker integration (cross-encoder or ColBERT-lite)
- Dense legal-specific embedding model
- Citation grounding & evaluation harness
- User/session scoped usage metrics & quotas
- Mobile client integration + offline packs

## Disclaimer
Informational only – not legal advice. See `docs/legal_disclaimers.md`.

## License
Apache 2.0. See `LICENSE` file.

---
Contributions welcome once license is defined.
