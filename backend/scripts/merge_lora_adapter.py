"""Merge a LoRA adapter into a base model weights directory for standalone inference.

Example:
python -m backend.scripts.merge_lora_adapter \
  --base TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter models/lora/run1 \
  --out models/merged/tinyllama-run1

Install deps: pip install -e .[llm]
"""
from __future__ import annotations
import argparse
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError as e:  # pragma: no cover
    raise SystemExit("Missing deps. Install with: pip install -e .[llm]") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='Base model name or path')
    ap.add_argument('--adapter', required=True, help='LoRA adapter directory')
    ap.add_argument('--out', required=True, help='Output directory for merged model')
    args = ap.parse_args()

    print(f"Loading base model: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base)

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging...")
    merged = model.merge_and_unload()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Merged model saved to {out_dir}")

if __name__ == '__main__':  # pragma: no cover
    main()
