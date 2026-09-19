"""
finetune_lora.py
==================
LoRA-fine-tunes a small open base model on the output of
synthetic_data_generator.py, producing the Tier-1 model that
tiered_sql_agent.py calls first.

Why LoRA on (retrieved context + question -> SQL), not (question -> SQL)
alone: the point of this fine-tune is to teach the base model to reliably
follow whatever schema context HybridRetriever hands it — which is the
skill that survives new questions and schema changes — rather than to
memorize a fixed schema baked into the weights. Every training row's
`messages` field already has that context inlined by the generator, so
this script does no prompt engineering of its own; it only teaches the
model to produce `messages[-1]` given `messages[:-1]`.

This is a scaffold, in the same spirit as the Bitnet1BitRouter /
KeywordFallbackRouter split in agent_orchestrator.py: it is a real,
runnable SFT recipe (transformers + peft + trl), but the base model,
batch size, and epoch count are exactly the knobs you're expected to
tune for your own GPU budget and base-model choice. Heavy deps
(torch/transformers/peft/trl/accelerate/bitsandbytes) are intentionally
kept OUT of requirements.txt — see requirements-finetune.txt — so the
rest of the pipeline (which targets a no-GPU/no-API-key baseline) never
pulls them in by accident.

Usage:
  pip install -r requirements-finetune.txt
  python finetune_lora.py \\
      --data synthetic_training_data.jsonl \\
      --base-model meta-llama/Llama-3.2-3B-Instruct \\
      --output-dir ./sql-lora-adapter \\
      --merge --merged-output-dir ./sql-lora-merged

Then serve the merged weights locally with Ollama (see the printed
Modelfile snippet, or FINE_TUNING.md) so tiered_sql_agent.py's
FineTunedSQLGenerationAgent can call it exactly like the baseline
SQLGenerationAgent calls llama3.2:latest today.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_dataset_split(data_path: str, eval_fraction: float, seed: int):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=data_path, split="train")
    if eval_fraction > 0:
        split = ds.train_test_split(test_size=eval_fraction, seed=seed)
        return split["train"], split["test"]
    return ds, None


def build_lora_config(r: int, alpha: int, dropout: float):
    from peft import LoraConfig, TaskType

    # Reasonable default target modules for Llama/Mistral/Qwen-family
    # decoder blocks. If you pick a base model with a different attention
    # module naming scheme, override --target-modules.
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def print_ollama_modelfile(merged_dir: str, tag: str) -> None:
    """SQLGenerationAgent (and FineTunedSQLGenerationAgent, which is the
    same class pointed at a different tag) calls Ollama's /api/chat with a
    plain model tag — this is the one manual step to get merged HF weights
    behind that same tag. GGUF conversion (llama.cpp's convert_hf_to_gguf.py)
    happens outside this script; this just prints the Modelfile you point
    `ollama create` at once you have a GGUF file."""
    print("\n--- Next step: serve the fine-tuned model via Ollama ---")
    print(f"1. Convert {merged_dir} to GGUF with llama.cpp's convert_hf_to_gguf.py")
    print("2. Write a Modelfile next to the .gguf file:")
    print(f"""
    FROM ./sql-lora-ft.gguf
    PARAMETER temperature 0
    SYSTEM \"\"\"You are a SQL generation engine for a brokerage platform.\"\"\"
    """)
    print(f"3. ollama create {tag} -f Modelfile")
    print(f"4. python tiered_sql_agent.py --dsn ... --tier1-model {tag} \"your question\"")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="JSONL from synthetic_data_generator.py (uses the `messages` field).")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--output-dir", default="./sql-lora-adapter")
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--merge", action="store_true",
                         help="After training, merge the LoRA adapter into the base weights (needed before GGUF conversion for Ollama).")
    parser.add_argument("--merged-output-dir", default="./sql-lora-merged")
    parser.add_argument("--ollama-tag", default="sql-lora-ft")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer, SFTConfig
        from peft import PeftModel
    except ImportError as e:
        print(
            f"Missing fine-tuning dependency ({e}). Install them first:\n"
            "  pip install -r requirements-finetune.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    train_ds, eval_ds = load_dataset_split(args.data, args.eval_fraction, args.seed)
    print(f"Loaded {len(train_ds)} training rows"
          + (f", {len(eval_ds)} eval rows" if eval_ds is not None else " (no eval split)"))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
    )

    lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        max_seq_length=args.max_seq_length,
        packing=False,  # rows have varying schema-context length; packing would blur example boundaries
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")

    if args.merge:
        print("Merging LoRA adapter into base weights...")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        )
        merged = PeftModel.from_pretrained(base, args.output_dir)
        merged = merged.merge_and_unload()
        Path(args.merged_output_dir).mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(args.merged_output_dir)
        tokenizer.save_pretrained(args.merged_output_dir)
        print(f"Merged model saved to {args.merged_output_dir}")
        print_ollama_modelfile(args.merged_output_dir, args.ollama_tag)


if __name__ == "__main__":
    main()
