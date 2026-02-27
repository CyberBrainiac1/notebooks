#!/usr/bin/env python
"""Train a Pranav profile LoRA adapter that fits 8GB VRAM."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("UNSLOTH_SKIP_TORCHVISION_CHECK", "1")

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from build_pranav_profile_dataset import build_and_write_dataset


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


PROMPT_TEMPLATE = """You are PranavProfileGPT, a profile-grounded assistant.
Rules:
- Use only known profile facts.
- If a fact is unknown, say you do not have that detail yet.
- Do not invent personal details.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a personal profile assistant on 8GB VRAM."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/pranav_profile_qa.jsonl"),
        help="Path to JSONL with instruction/input/output fields.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pranav_8gb"),
        help="Directory for checkpoints and final adapter.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=250,
        help="Training steps. Increase to 600+ for stronger fitting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (8/16/32 typical).",
    )
    parser.add_argument(
        "--save-merged-16bit",
        action="store_true",
        help="Also export a merged 16-bit model (larger disk usage).",
    )
    parser.add_argument(
        "--auto-max-steps",
        action="store_true",
        help="Automatically derive max-steps from dataset size and effective batch size.",
    )
    parser.add_argument(
        "--target-epochs",
        type=float,
        default=3.0,
        help="Approximate epochs used when --auto-max-steps is enabled.",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=120,
        help="Lower clamp for auto-derived max-steps.",
    )
    parser.add_argument(
        "--max-steps-cap",
        type=int,
        default=700,
        help="Upper clamp for auto-derived max-steps.",
    )
    return parser.parse_args()


def ensure_dataset(path: Path) -> None:
    if path.exists():
        return
    print(f"Dataset not found at {path}. Building it now...")
    count = build_and_write_dataset(path)
    print(f"Built dataset with {count} examples.")


def format_dataset(raw_dataset, eos_token: str):
    def _fmt(batch):
        texts = []
        for inst, user_input, out in zip(
            batch["instruction"], batch["input"], batch["output"]
        ):
            text = PROMPT_TEMPLATE.format(inst, user_input, out) + eos_token
            texts.append(text)
        return {"text": texts}

    return raw_dataset.map(_fmt, batched=True)


def dedupe_dataset(dataset):
    seen: set[tuple[str, str, str]] = set()
    keep: list[int] = []
    for idx, row in enumerate(dataset):
        key = (
            str(row["instruction"]).strip().lower(),
            str(row["input"]).strip().lower(),
            str(row["output"]).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    if len(keep) == len(dataset):
        return dataset, 0
    deduped = dataset.select(keep)
    return deduped, len(dataset) - len(keep)


def resolve_max_steps(args: argparse.Namespace, row_count: int) -> tuple[int, bool]:
    if args.auto_max_steps or args.max_steps <= 0:
        effective_batch = max(1, args.batch_size * args.grad_accum)
        derived = math.ceil((row_count * max(0.5, args.target_epochs)) / effective_batch)
        steps = max(args.min_steps, min(args.max_steps_cap, derived))
        return steps, True
    return args.max_steps, False


def main() -> None:
    args = parse_args()
    ensure_dataset(args.dataset_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print("Loading and formatting dataset...")
    dataset = load_dataset("json", data_files=str(args.dataset_path), split="train")
    dataset, removed = dedupe_dataset(dataset)
    if removed:
        print(f"Deduplicated dataset: removed {removed} duplicate rows.")
    row_count = len(dataset)
    max_steps, auto_used = resolve_max_steps(args, row_count)
    warmup_steps = max(5, min(50, int(max_steps * 0.08)))
    print(
        "Training config:",
        f"rows={row_count}, max_steps={max_steps}",
        f"(auto={auto_used}), batch={args.batch_size}, grad_accum={args.grad_accum}",
    )
    dataset = format_dataset(dataset, tokenizer.eos_token)

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=SFTConfig(
            output_dir=str(args.output_dir / "checkpoints"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=args.learning_rate,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=3407,
            report_to="none",
            fp16=not bf16_supported,
            bf16=bf16_supported,
            save_strategy="steps",
            save_steps=50,
        ),
    )

    print("Starting training...")
    stats = trainer.train()
    print(f"Training complete. Runtime seconds: {stats.metrics.get('train_runtime')}")

    adapter_dir = args.output_dir / "pranav_lora"
    print(f"Saving LoRA adapter to {adapter_dir} ...")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if args.save_merged_16bit:
        merged_dir = args.output_dir / "pranav_merged_16bit"
        print(f"Saving merged 16-bit model to {merged_dir} ...")
        model.save_pretrained_merged(
            str(merged_dir), tokenizer, save_method="merged_16bit"
        )

    print("Done.")


if __name__ == "__main__":
    main()
