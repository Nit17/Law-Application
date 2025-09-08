"""LoRA fine-tuning script scaffold.

Usage example:
  LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  python -m backend.scripts.finetune_lora \
    --dataset data/instruct_train.jsonl \
    --output-dir models/lora/run1

Dataset format (JSONL):
{"instruction": "...", "context": "...", "response": "..."}
Context is optional; prompt will concatenate when present.

This is a minimal starting point; adjust hyperparameters for real training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import os

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    import datasets
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError as e:  # pragma: no cover
    raise SystemExit("Missing dependencies. Install extras: pip install .[llm]") from e


def build_prompt(example: dict) -> str:
    instruction = example.get("instruction", "").strip()
    context = example.get("context", "").strip()
    response = example.get("response", "").strip()
    parts = ["<|system|> You are a legal assistant for Indian law."]
    if context:
        parts.append(f"<|context|>\n{context}")
    parts.append(f"<|instruction|>\n{instruction}")
    parts.append(f"<|response|>\n{response}")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    ap.add_argument("--output-dir", required=True, help="Directory to store adapter")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    model_name = args.model

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)

    ds = datasets.load_dataset("json", data_files=str(Path(args.dataset)), split="train")

    def _map(ex):
        prompt = build_prompt(ex)
        out = tok(prompt, truncation=True, max_length=2048)
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = ds.map(_map, remove_columns=ds.column_names)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        gradient_accumulation_steps=1,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter + tokenizer to {args.output_dir}")

if __name__ == "__main__":  # pragma: no cover
    main()
