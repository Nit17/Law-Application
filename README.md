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

## Extending Models
- Add llama.cpp: create new backend in `core/llm_<backend>.py` and route selection env var
- Add embedding/hybrid retrieval: introduce `EmbeddingsStore` side-by-side with TF‑IDF
- Add streaming: wrap generate with token iterator (transformers `streamer=True`)

## Roadmap
- Hybrid ranking (TF‑IDF + embeddings)
- Dense embedding upgrade (legal-specific model)
- Citation grounding & answer evaluation
- Model adapter loading (LoRA merge at runtime)
- Mobile client integration + offline packs

## Disclaimer
Informational only – not legal advice. See `docs/legal_disclaimers.md`.

## License
Apache 2.0. See `LICENSE` file.

---
Contributions welcome once license is defined.
