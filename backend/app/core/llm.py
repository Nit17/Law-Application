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
from typing import Optional, Dict, Any, List
import os

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:  # pragma: no cover - we handle absence gracefully
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    torch = None  # type: ignore

DEFAULT_MODEL = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
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
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError(
            "LLM dependencies not installed. Install with: pip install 'transformers' 'accelerate' 'torch' 'peft' 'datasets'"
        )
    device = _select_device()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        device_map="auto" if device in ("cuda", "mps") else None,
        torch_dtype=(torch.float16 if device == "cuda" else None),
    )
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
