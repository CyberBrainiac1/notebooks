#!/usr/bin/env python
"""Interactive chat runner for the trained Pranav profile LoRA adapter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("UNSLOTH_SKIP_TORCHVISION_CHECK", "1")

import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer


PROMPT_TEMPLATE = """You are PranavProfileGPT, a profile-grounded assistant.
Rules:
- Use only known profile facts.
- If a fact is unknown, say you do not have that detail yet.
- Do not invent personal details.

### Instruction:
Answer the user question about Pranav's profile.

### Input:
{}

### Response:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with Pranav profile model.")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("outputs/pranav_8gb/pranav_lora"),
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length for inference.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=220,
        help="Maximum generated tokens per answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter path not found: {args.adapter_dir}. "
            "Run scripts/train_pranav_8gb.py first."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter_dir),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("PranavProfileGPT ready. Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        prompt = PROMPT_TEMPLATE.format(question)
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        print("Model:", end=" ")
        _ = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            use_cache=True,
            streamer=streamer,
        )


if __name__ == "__main__":
    main()
