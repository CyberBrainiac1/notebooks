#!/usr/bin/env python
"""Colab-only end-to-end runner for Pranav model training and GGUF export."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def run(cmd: list[str], cwd: Path | None = None) -> None:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"> {pretty}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def install_dependencies() -> None:
    print("Installing dependencies...")
    cmds = [
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers==4.56.2",
            "datasets==4.3.0",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "trl",
            "peft",
            "accelerate",
            "bitsandbytes",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "unsloth==2026.2.1",
            "unsloth-zoo==2026.2.1",
            "sentencepiece",
            "protobuf",
            "huggingface_hub",
            "hf_transfer",
        ],
    ]
    for cmd in cmds:
        run(cmd)


def maybe_clone_llama_cpp(dst: Path) -> None:
    convert_script = dst / "convert_hf_to_gguf.py"
    if convert_script.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(dst)])


def read_eval_ratio(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("overall", {}).get("ratio", 0.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full Colab pipeline: dataset -> train -> eval -> gguf export."
    )
    parser.add_argument(
        "--allow-local",
        action="store_true",
        help="Allow running outside Colab (disabled by default).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pranav_colab_3b"),
        help="Training output directory.",
    )
    parser.add_argument(
        "--dataset-v2",
        type=Path,
        default=Path("data/pranav_profile_qa_v2.jsonl"),
        help="Intermediate dataset path.",
    )
    parser.add_argument(
        "--dataset-v4",
        type=Path,
        default=Path("data/pranav_profile_qa_v4.jsonl"),
        help="Final enriched dataset path.",
    )
    parser.add_argument(
        "--repeat-scale",
        type=int,
        default=1,
        help="Booster repeat scale for hard-cases.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        help="Base model name.",
    )
    parser.add_argument(
        "--target-epochs",
        type=float,
        default=2.8,
        help="Target epochs used by auto-step sizing.",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=160,
        help="Minimum auto steps.",
    )
    parser.add_argument(
        "--max-steps-cap",
        type=int,
        default=460,
        help="Maximum auto steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.5e-4,
        help="Training learning rate.",
    )
    parser.add_argument(
        "--min-eval-ratio",
        type=float,
        default=0.90,
        help="Minimum eval ratio required before export.",
    )
    parser.add_argument(
        "--build-gguf",
        action="store_true",
        help="Convert merged model to f16 GGUF.",
    )
    parser.add_argument(
        "--gguf-out",
        type=Path,
        default=Path("outputs/pranav_colab_3b/pranav-assistant-f16.gguf"),
        help="GGUF output path.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=Path("runtime/llama.cpp"),
        help="llama.cpp directory for conversion.",
    )
    parser.add_argument(
        "--download-gguf",
        action="store_true",
        help="Attempt Colab files.download on exported GGUF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not in_colab() and not args.allow_local:
        raise SystemExit(
            "Refusing to run outside Colab to avoid local GPU usage. "
            "Use this in Colab, or pass --allow-local explicitly."
        )

    repo_root = args.repo_root.resolve()
    os.chdir(repo_root)
    print(f"Working directory: {repo_root}")

    if not args.skip_install:
        install_dependencies()

    # Build datasets.
    run(
        [
            sys.executable,
            "scripts/build_pranav_profile_dataset.py",
            "--output",
            str(args.dataset_v2),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/build_pranav_booster_dataset.py",
            "--input",
            str(args.dataset_v2),
            "--output",
            str(args.dataset_v4),
            "--repeat-scale",
            str(args.repeat_scale),
        ]
    )

    # Train.
    run(
        [
            sys.executable,
            "scripts/train_pranav_8gb.py",
            "--dataset-path",
            str(args.dataset_v4),
            "--output-dir",
            str(args.output_dir),
            "--model-name",
            args.model_name,
            "--batch-size",
            "1",
            "--grad-accum",
            "8",
            "--max-seq-length",
            "1024",
            "--max-steps",
            "0",
            "--auto-max-steps",
            "--target-epochs",
            str(args.target_epochs),
            "--min-steps",
            str(args.min_steps),
            "--max-steps-cap",
            str(args.max_steps_cap),
            "--learning-rate",
            str(args.learning_rate),
            "--save-merged-16bit",
        ]
    )

    # Evaluate adapter.
    eval_report = args.output_dir / "eval_report.json"
    run(
        [
            sys.executable,
            "scripts/eval_pranav_adapter.py",
            "--adapter-dir",
            str(args.output_dir / "pranav_lora"),
            "--report",
            str(eval_report),
            "--min-overall-pass",
            str(args.min_eval_ratio),
        ]
    )
    eval_ratio = read_eval_ratio(eval_report)
    print(f"Eval ratio: {eval_ratio:.3f}")

    if not args.build_gguf:
        print("Skipping GGUF export. Use --build-gguf to export for Ollama.")
        return

    merged_dir = args.output_dir / "pranav_merged_16bit"
    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged model not found: {merged_dir}")

    maybe_clone_llama_cpp(args.llama_cpp_dir)
    args.gguf_out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(args.llama_cpp_dir / "convert_hf_to_gguf.py"),
            str(merged_dir),
            "--outfile",
            str(args.gguf_out),
            "--outtype",
            "f16",
        ]
    )
    print(f"GGUF ready: {args.gguf_out}")

    if args.download_gguf and in_colab():
        try:
            from google.colab import files  # type: ignore

            files.download(str(args.gguf_out))
        except Exception as exc:  # noqa: BLE001
            print(f"Download failed: {exc}")


if __name__ == "__main__":
    main()
