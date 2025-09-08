"""LLM loading abstraction with pluggable backends.

Current default: TinyLlama or any causal instruction model from Hugging Face.
Optional future backends: llama.cpp, vLLM, OpenAI-compatible gateways.

Design goals:
- Lazy singleton load (first generate call)
- Minimal dependency footprint if user doesn't install LLM extras
- Clear error guidance when extras missing
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Dict, Any, List, Generator
import os

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:  # pragma: no cover - we handle absence gracefully
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    torch = None  # type: ignore

DEFAULT_MODEL = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
BACKEND = os.getenv("LLM_BACKEND", "transformers")  # transformers|llamacpp (future: vllm)
LORA_ADAPTER = os.getenv("LLM_LORA_ADAPTER")  # path to LoRA adapter directory (PEFT)
LORA_MERGE = os.getenv("LLM_LORA_MERGE", "0") in {"1", "true", "True"}
MAX_INPUT_TOKENS = int(os.getenv("LLM_MAX_INPUT_TOKENS", "2048"))
MAX_GENERATION_TOKENS = int(os.getenv("LLM_MAX_GENERATION_TOKENS", "256"))
DEVICE = os.getenv("LLM_DEVICE", "auto")  # auto|cpu|cuda|mps
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))


@dataclass
class GenerationResult:
    prompt: str
    completion: str
    model: str
    tokens_in: int
    tokens_out: int
    usage: Dict[str, Any]


def _select_device():
    if DEVICE != "auto":
        return DEVICE
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _load_model():  # returns (tokenizer, model, device)
    if BACKEND == "llamacpp":
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("llama-cpp backend selected but llama-cpp-python not installed. Install with: pip install .[llamacpp]") from e
        # For llama.cpp we treat DEFAULT_MODEL as path to GGUF file
        model_path = DEFAULT_MODEL
        if not os.path.exists(model_path):
            raise RuntimeError(f"GGUF model file not found: {model_path}")
        llm = Llama(model_path=model_path, n_ctx=MAX_INPUT_TOKENS)
        return None, llm, "cpu"

    # transformers backend
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError(
            "Transformers backend selected but dependencies missing. Install with: pip install .[llm]"
        )
    device = _select_device()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        device_map="auto" if device in ("cuda", "mps") else None,
        torch_dtype=(torch.float16 if device == "cuda" else None),
    )
    # Optional LoRA adapter
    if LORA_ADAPTER:
        try:
            from peft import PeftModel
            base_loaded = model
            model = PeftModel.from_pretrained(model, LORA_ADAPTER)
            if LORA_MERGE:
                model = model.merge_and_unload()
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Failed to load LoRA adapter at {LORA_ADAPTER}: {e}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, device


def build_prompt(question: str, contexts: List[str]) -> str:
    ctx_block = "\n\n".join(f"[CONTEXT {i+1}]\n{c}" for i, c in enumerate(contexts))
    system = (
        "You are a legal assistant for Indian law. Answer the user question using ONLY the provided context snippets. "
        "If the answer cannot be found, say you do not have enough information. Provide concise, factual responses with section references when possible."
    )
    return f"<|system|>\n{system}\n<|context|>\n{ctx_block}\n<|question|>\n{question}\n<|answer|>"


def generate(question: str, contexts: List[str]) -> GenerationResult:
    tokenizer, model, device = _load_model()
    prompt = build_prompt(question, contexts)

    if BACKEND == "llamacpp":
        # llama.cpp API
        output = model(prompt, max_tokens=MAX_GENERATION_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
        completion = output["choices"][0]["text"].strip()
        usage_meta = output.get("usage", {})
        return GenerationResult(
            prompt=prompt,
            completion=completion,
            model=f"llamacpp:{DEFAULT_MODEL}",
            tokens_in=usage_meta.get("prompt_tokens", 0),
            tokens_out=usage_meta.get("completion_tokens", 0),
            usage=usage_meta,
        )

    # transformers path
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    input_ids = inputs["input_ids"].to(model.device)
    attn_mask = inputs["attention_mask"].to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=MAX_GENERATION_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    completion = full_text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)) :].strip()
    return GenerationResult(
        prompt=prompt,
        completion=completion,
        model=DEFAULT_MODEL,
        tokens_in=input_ids.shape[1],
        tokens_out=gen_ids.shape[1] - input_ids.shape[1],
        usage={"total_tokens": gen_ids.shape[1], "prompt_tokens": input_ids.shape[1], "completion_tokens": gen_ids.shape[1] - input_ids.shape[1]},
    )

def stream_generate(question: str, contexts: List[str]) -> Generator[str, None, None]:
    """Yield tokens/chunks incrementally. Transformers only for now."""
    if BACKEND == "llamacpp":
        # Basic streaming via llama.cpp token callback is more involved; placeholder for future.
        result = generate(question, contexts)
        yield result.completion
        return
    tokenizer, model, device = _load_model()
    prompt = build_prompt(question, contexts)
    if AutoTokenizer is None:
        raise RuntimeError("Streaming requires transformers backend")
    streamer = None
    try:
        from transformers import TextIteratorStreamer  # type: ignore
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        import threading
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(model.device)
        def _work():
            model.generate(**inputs, max_new_tokens=MAX_GENERATION_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, streamer=streamer, pad_token_id=tokenizer.eos_token_id)
        t = threading.Thread(target=_work)
        t.start()
        for token in streamer:
            yield token
        t.join()
    except Exception as e:  # pragma: no cover
        yield f"[streaming error: {e}]"
