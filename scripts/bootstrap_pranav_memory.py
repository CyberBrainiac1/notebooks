#!/usr/bin/env python
"""One-shot bootstrap to ingest all Pranav datasets into vector memory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

try:
    from ollama_vector_memory_assistant import (
        DEFAULT_SEED_DATASETS,
        MemoryStore,
        OllamaClient,
        ensure_base_dataset,
        resolve_model_name,
    )
except ModuleNotFoundError:
    from scripts.ollama_vector_memory_assistant import (  # type: ignore
        DEFAULT_SEED_DATASETS,
        MemoryStore,
        OllamaClient,
        ensure_base_dataset,
        resolve_model_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap vector memory from profile datasets."
    )
    parser.add_argument("--host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("runtime/ollama_vector_memory"),
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path("data/pranav_profile_qa_v4.jsonl"),
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text",
    )
    parser.add_argument("--auto-start-ollama", action="store_true")
    parser.add_argument("--auto-pull-embed-model", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--extra-dataset",
        action="append",
        default=[],
        help="Extra dataset JSONL path to ingest. Can be used multiple times.",
    )
    return parser.parse_args()


def gather_existing_paths(base_dataset: Path, extra: List[str]) -> List[Path]:
    candidates = [base_dataset, *DEFAULT_SEED_DATASETS, *[Path(x) for x in extra if x.strip()]]
    out: List[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str((p.resolve() if p.exists() else p)).lower()
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            out.append(p)
    return out


def main() -> None:
    args = parse_args()
    ensure_base_dataset(args.base_dataset)

    client = OllamaClient(host=args.host, timeout=40)
    client.ensure_running(auto_start=args.auto_start_ollama)
    models = client.list_models()
    embed_model = resolve_model_name(args.embed_model, models)
    if embed_model is None and args.auto_pull_embed_model:
        print(f"System: pulling embedding model `{args.embed_model}` ...")
        client.pull_model(args.embed_model)
        models = client.list_models()
        embed_model = resolve_model_name(args.embed_model, models)
    if embed_model is None:
        raise RuntimeError(
            f"Embedding model `{args.embed_model}` not found.\n"
            f"Run: ollama pull {args.embed_model}"
        )

    store = MemoryStore(args.runtime_dir / "memory.db")
    dataset_paths = gather_existing_paths(args.base_dataset, args.extra_dataset)
    if not dataset_paths:
        raise RuntimeError("No dataset files found for memory bootstrap.")

    print("System: bootstrapping vector memory from datasets...")
    total_added = 0
    for path in dataset_paths:
        added = store.seed_from_dataset(
            dataset_path=path,
            embed_fn=lambda texts: client.embed_texts(embed_model, texts),
            batch_size=max(1, args.batch_size),
        )
        total_added += added
        print(f"System: {path} -> added {added}")

    print(
        f"System: bootstrap complete. added_total={total_added}, "
        f"memory_count={store.memory_count()}"
    )


if __name__ == "__main__":
    main()
